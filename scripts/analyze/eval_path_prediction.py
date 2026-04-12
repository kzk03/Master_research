"""
パスごとの貢献者数予測評価スクリプト（Step 1）

時点 T で各ディレクトリの Δ ヶ月後の貢献者数を予測し、
実際の貢献者数と比較する。

手法:
  - Variant A（単純合計）: Σ continuation_prob_d
  - Variant B（親和度加重）: Σ continuation_prob_d × affinity(d, D)

ベースライン:
  - Naive: T 時点の貢献者数がそのまま続くと仮定
  - Linear: 過去の貢献者数推移から線形外挿

使い方:
    python scripts/analyze/eval_path_prediction.py \
        --data data/nova_raw.csv \
        --raw-json data/raw_json/openstack__nova.json \
        --irl-model outputs/cross_temporal_v39/train_0-3m/irl_model.pt \
        --prediction-time 2014-01-01 \
        --delta-months 3 \
        --window-days 180
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from review_predictor.IRL.features.directory_contributors import (
    count_actual_contributors,
    get_all_directories,
    get_directory_developers,
)
from review_predictor.IRL.features.path_features import (
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map,
)
from review_predictor.IRL.model.batch_predictor import BatchContinuationPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── 親和度スコア ──────────────────────────────────────────────

# path features: [path_review_count, path_recency, path_acceptance_rate]
AFFINITY_WEIGHTS = np.array([0.5, 0.3, 0.2], dtype=np.float32)


def compute_affinity_score(path_features: np.ndarray) -> float:
    """path features (3-dim) を scalar affinity score に変換。"""
    return float(np.dot(path_features, AFFINITY_WEIGHTS))


# ── 予測関数 ──────────────────────────────────────────────────


def predict_contributor_counts(
    dir_developers: Dict[str, Set[str]],
    continuation_probs: Dict[str, float],
    path_extractor: PathFeatureExtractor,
    prediction_time: datetime,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    各ディレクトリについて Variant A / Variant B の予測貢献者数を算出。

    Returns:
        (variant_a, variant_b): {directory: predicted_count}
    """
    variant_a: Dict[str, float] = {}
    variant_b: Dict[str, float] = {}

    for dir_path, developers in dir_developers.items():
        sum_a = 0.0
        sum_b = 0.0
        task_dirs = frozenset({dir_path})

        for dev in developers:
            prob = continuation_probs.get(dev, 0.5)
            sum_a += prob

            pf = path_extractor.compute(dev, task_dirs, prediction_time)
            affinity = compute_affinity_score(pf)
            sum_b += prob * affinity

        variant_a[dir_path] = sum_a
        variant_b[dir_path] = sum_b

    return variant_a, variant_b


# ── ベースライン ──────────────────────────────────────────────


def baseline_naive(
    df: pd.DataFrame,
    prediction_time: datetime,
    window_days: int,
) -> Dict[str, int]:
    """Naive ベースライン: T 時点の貢献者数がそのまま続くと仮定。"""
    start = prediction_time - pd.Timedelta(days=window_days)
    return count_actual_contributors(df, start, prediction_time)


def baseline_linear(
    df: pd.DataFrame,
    prediction_time: datetime,
    delta_months: int,
    n_periods: int = 3,
    period_months: int = 3,
) -> Dict[str, float]:
    """
    線形トレンドベースライン:
    過去 n_periods 期間の貢献者数から線形外挿する。
    """
    # 過去の各期間の貢献者数を集計
    period_counts: Dict[str, List[float]] = {}
    for i in range(n_periods, 0, -1):
        end = prediction_time - pd.DateOffset(months=(i - 1) * period_months)
        start = end - pd.DateOffset(months=period_months)
        counts = count_actual_contributors(df, start, end)
        for d, c in counts.items():
            period_counts.setdefault(d, []).append(float(c))

    # 線形外挿
    result: Dict[str, float] = {}
    for d, counts in period_counts.items():
        if len(counts) < 2:
            result[d] = counts[-1] if counts else 0.0
            continue
        # 単純線形回帰: y = a*x + b
        x = np.arange(len(counts), dtype=np.float64)
        y = np.array(counts, dtype=np.float64)
        a, b = np.polyfit(x, y, 1)
        # delta_months 先を予測
        future_x = len(counts) - 1 + delta_months / period_months
        result[d] = max(0.0, float(a * future_x + b))

    return result


# ── 評価指標 ──────────────────────────────────────────────────


def compute_metrics(
    predicted: Dict[str, float],
    actual: Dict[str, int],
    method_name: str,
) -> Dict[str, float]:
    """MAE, RMSE, 相関を計算。"""
    # 両方に存在するディレクトリだけで比較
    common_dirs = set(predicted.keys()) & set(actual.keys())
    if not common_dirs:
        logger.warning(f"[{method_name}] 共通ディレクトリが0件")
        return {"mae": float("nan"), "rmse": float("nan"), "n_dirs": 0}

    pred_vals = np.array([predicted[d] for d in common_dirs])
    actual_vals = np.array([float(actual[d]) for d in common_dirs])

    mae = float(np.mean(np.abs(pred_vals - actual_vals)))
    rmse = float(np.sqrt(np.mean((pred_vals - actual_vals) ** 2)))

    # 相関
    if len(common_dirs) >= 3 and np.std(pred_vals) > 0 and np.std(actual_vals) > 0:
        from scipy import stats

        pearson_r, _ = stats.pearsonr(pred_vals, actual_vals)
        spearman_r, _ = stats.spearmanr(pred_vals, actual_vals)
    else:
        pearson_r = float("nan")
        spearman_r = float("nan")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "n_dirs": float(len(common_dirs)),
    }
    logger.info(
        f"[{method_name}] "
        + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    )
    return metrics


def compute_danger_detection(
    predicted: Dict[str, float],
    actual: Dict[str, int],
    threshold: float = 1.0,
    danger_actual_threshold: int = 1,
    method_name: str = "",
) -> Dict[str, float]:
    """
    危険パス検知: actual_count <= danger_actual_threshold を「危険」として
    predicted < threshold で検知できるかを評価。
    """
    common_dirs = set(predicted.keys()) & set(actual.keys())
    if not common_dirs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = fp = fn = tn = 0
    for d in common_dirs:
        is_danger_actual = actual[d] <= danger_actual_threshold
        is_danger_pred = predicted[d] < threshold
        if is_danger_actual and is_danger_pred:
            tp += 1
        elif not is_danger_actual and is_danger_pred:
            fp += 1
        elif is_danger_actual and not is_danger_pred:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
    logger.info(
        f"[{method_name} danger] "
        + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    )
    return metrics


# ── メイン ────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="パスごとの貢献者数予測評価")
    p.add_argument("--data", type=Path, default=Path("data/nova_raw.csv"))
    p.add_argument(
        "--raw-json",
        type=Path,
        default=Path("data/raw_json/openstack__nova.json"),
    )
    p.add_argument(
        "--irl-model",
        type=Path,
        default=Path("outputs/cross_temporal_v39/train_0-3m/irl_model.pt"),
    )
    p.add_argument("--prediction-time", type=str, default="2014-01-01")
    p.add_argument("--delta-months", type=int, default=3)
    p.add_argument("--window-days", type=int, default=180)
    p.add_argument("--dir-depth", type=int, default=2)
    p.add_argument("--path-window-days", type=int, default=180)
    p.add_argument("--danger-threshold", type=float, default=1.5)
    p.add_argument(
        "--danger-actual-threshold",
        type=int,
        default=1,
        help="actual_count がこの値以下のディレクトリを「危険」とみなす",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--multi-timepoint",
        action="store_true",
        help="複数時点で評価（3ヶ月刻み）",
    )
    return p.parse_args()


def evaluate_single_timepoint(
    df: pd.DataFrame,
    path_extractor: PathFeatureExtractor,
    predictor: BatchContinuationPredictor,
    prediction_time: datetime,
    delta_months: int,
    window_days: int,
    danger_threshold: float,
    danger_actual_threshold: int = 1,
) -> Dict[str, Dict[str, float]]:
    """単一時点の評価を実行し、全手法の結果を返す。"""

    logger.info(f"=== 予測時点 T={prediction_time}, Δ={delta_months}ヶ月 ===")

    pred_time_pd = pd.Timestamp(prediction_time)

    # 1. 各ディレクトリの過去の貢献者を取得
    window_start = prediction_time - pd.Timedelta(days=window_days)
    dir_developers = get_directory_developers(df, window_start, prediction_time)
    logger.info(f"対象ディレクトリ数: {len(dir_developers)}")

    # 2. 全開発者の continuation_prob をバッチ推論
    all_devs = set()
    for devs in dir_developers.values():
        all_devs.update(devs)
    logger.info(f"推論対象開発者数: {len(all_devs)}")

    continuation_probs = predictor.predict_batch(
        list(all_devs), prediction_time
    )

    # 3. IRL 予測 (Variant A / B)
    variant_a, variant_b = predict_contributor_counts(
        dir_developers, continuation_probs, path_extractor, prediction_time
    )

    # 3.5 スケーリング補正版 (Variant A-scaled)
    # IRL の continuation_prob は絶対値が低い傾向があるため、
    # 過去の実績（Naive）との比率で補正する
    naive_pred_raw = baseline_naive(df, prediction_time, window_days)
    naive_pred = {d: float(c) for d, c in naive_pred_raw.items()}

    # Variant A-scaled: naive と共通のディレクトリで平均スケール比を計算
    common_for_scale = set(variant_a.keys()) & set(naive_pred.keys())
    if common_for_scale:
        naive_vals = np.array([naive_pred[d] for d in common_for_scale])
        irl_vals = np.array([variant_a[d] for d in common_for_scale])
        # IRL の合計値 / Naive の合計値 の逆数がスケール係数
        irl_sum = irl_vals.sum()
        naive_sum = naive_vals.sum()
        scale_factor = naive_sum / irl_sum if irl_sum > 0 else 1.0
    else:
        scale_factor = 1.0
    variant_a_scaled = {d: v * scale_factor for d, v in variant_a.items()}
    logger.info(f"スケール係数: {scale_factor:.2f}")

    # 4. ベースライン（Linear）
    linear_pred = baseline_linear(df, prediction_time, delta_months)

    # 5. Ground truth
    future_start = pred_time_pd + pd.DateOffset(months=0)
    future_end = pred_time_pd + pd.DateOffset(months=delta_months)
    actual = count_actual_contributors(df, future_start, future_end)
    logger.info(
        f"Ground truth: {len(actual)} dirs, "
        f"avg={np.mean(list(actual.values())):.1f} devs/dir"
    )

    # 6. 評価
    all_results: Dict[str, Dict[str, float]] = {}

    methods = {
        "Naive": naive_pred,
        "Linear": linear_pred,
        "IRL_VariantA": variant_a,
        "IRL_VariantA_scaled": variant_a_scaled,
        "IRL_VariantB": variant_b,
    }

    for name, pred in methods.items():
        metrics = compute_metrics(pred, actual, name)
        danger = compute_danger_detection(
            pred,
            actual,
            threshold=danger_threshold,
            danger_actual_threshold=danger_actual_threshold,
            method_name=name,
        )
        all_results[name] = {**metrics, **{f"danger_{k}": v for k, v in danger.items()}}

    # 7. ディレクトリ別詳細テーブル
    rows = []
    for d in sorted(set(actual.keys()) | set(variant_a.keys())):
        rows.append(
            {
                "directory": d,
                "actual": actual.get(d, 0),
                "naive": naive_pred.get(d, 0.0),
                "linear": linear_pred.get(d, 0.0),
                "irl_a": variant_a.get(d, 0.0),
                "irl_a_s": variant_a_scaled.get(d, 0.0),
                "irl_b": variant_b.get(d, 0.0),
                "n_past_devs": len(dir_developers.get(d, set())),
            }
        )
    if rows:
        detail_df = pd.DataFrame(rows).sort_values("actual", ascending=False)
        logger.info("\n=== ディレクトリ別詳細（上位15件） ===")
        logger.info(
            detail_df.head(15).to_string(
                index=False, float_format=lambda x: f"{x:.2f}"
            )
        )
    else:
        logger.info("ディレクトリ別詳細: データなし")

    return all_results


def main() -> None:
    args = parse_args()

    # データロード
    logger.info(f"データ読み込み: {args.data}")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ディレクトリマッピング
    cdm = load_change_dir_map(args.raw_json, depth=args.dir_depth)
    df = attach_dirs_to_df(df, cdm)

    # PathFeatureExtractor
    path_extractor = PathFeatureExtractor(df, window_days=args.path_window_days)

    # BatchContinuationPredictor
    prediction_time = datetime.fromisoformat(args.prediction_time)
    # history_start: データの最初から
    history_start = df["timestamp"].min().to_pydatetime()
    predictor = BatchContinuationPredictor(
        model_path=args.irl_model,
        df=df,
        history_start=history_start,
        device=args.device,
    )

    if args.multi_timepoint:
        # 複数時点で評価
        # 月100件以上のデータがある期間を対象にする
        monthly = df.set_index("timestamp").resample("MS").size()
        active_months = monthly[monthly >= 100].index
        if len(active_months) < 2:
            logger.error("十分なデータ量の月が不足しています")
            return
        data_max = df["timestamp"].max()
        # 最低 6ヶ月の履歴 + delta_months の将来データが必要
        min_start = active_months[0] + pd.DateOffset(months=6)
        max_start = data_max - pd.DateOffset(months=args.delta_months)

        timepoints = pd.date_range(min_start, max_start, freq="3MS")
        logger.info(f"複数時点評価: {len(timepoints)} 時点")

        all_timepoint_results: Dict[str, List[Dict[str, float]]] = {}
        for t in timepoints:
            t_dt = t.to_pydatetime()
            results = evaluate_single_timepoint(
                df,
                path_extractor,
                predictor,
                t_dt,
                args.delta_months,
                args.window_days,
                args.danger_threshold,
                args.danger_actual_threshold,
            )
            for method, metrics in results.items():
                all_timepoint_results.setdefault(method, []).append(metrics)

        # 集約
        logger.info("\n=== 複数時点の平均指標 ===")
        for method, metrics_list in all_timepoint_results.items():
            avg = {}
            for key in metrics_list[0]:
                vals = [m[key] for m in metrics_list if not np.isnan(m[key])]
                avg[key] = np.mean(vals) if vals else float("nan")
            logger.info(
                f"[{method}] "
                + " ".join(f"{k}={v:.3f}" for k, v in avg.items() if "danger_t" not in k)
            )
    else:
        # 単一時点
        evaluate_single_timepoint(
            df,
            path_extractor,
            predictor,
            prediction_time,
            args.delta_months,
            args.window_days,
            args.danger_threshold,
            args.danger_actual_threshold,
        )


if __name__ == "__main__":
    main()
