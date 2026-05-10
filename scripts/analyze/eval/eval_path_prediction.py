"""
パスごとの貢献者数予測評価スクリプト（Step 1）

特徴:
  - joblib による並列化（RF推論, RF_Dir推論）
  - GPU 対応（--device cuda）

使い方:
    python scripts/analyze/eval/eval_path_prediction.py \
        --data data/nova_raw.csv \
        --raw-json data/raw_json/openstack__nova.json \
        --irl-model outputs/cross_temporal_v39/train_0-3m/irl_model.pt \
        --prediction-time 2014-01-01 \
        --delta-months 3 \
        --window-days 180 \
        --device cuda \
        --n-jobs -1
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from review_predictor.IRL.features.common_features import (
    FEATURE_NAMES,
    extract_common_features,
)
from review_predictor.IRL.model.rf_predictor import (
    extract_features_for_window,
    extract_features_for_window_directory,
    prepare_rf_features,
    prepare_rf_features_directory,
)
from review_predictor.IRL.features.directory_contributors import (
    count_actual_contributors,
    get_all_directories,
    get_directory_developers,
)
from review_predictor.IRL.features.path_features import (
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map,
    load_change_dir_map_multi,
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


# ── RF ベースライン（IRL と同じ 25 次元特徴量・同じ集約方法） ──────
#
# IRL: 開発者ごとに 25 次元特徴量の月次時系列 → LSTM → continuation_prob → Σ per dir
# RF:  開発者ごとに 25 次元特徴量のスナップショット → RF → continuation_prob → Σ per dir
#
# 特徴量・集約方法が同じなので、差分は「時系列モデリング（LSTM）の有無」だけ。
# rf_predictor.py の学習方式をそのまま使い、個人の継続確率を予測してから集約する。


def _train_rf_classifier(
    df: pd.DataFrame,
    prediction_time: datetime,
    window_days: int,
    delta_months: int,
    future_start_months: int = 0,
    rf_train_end: Optional[datetime] = None,
):
    """
    卒論と同じ方式で RF を学習する。

    rf_train_end が指定された場合、IRL と同じ train_end を基準に
    ラベル期間を決定する（公平な比較）。
    """
    from sklearn.ensemble import RandomForestClassifier

    if rf_train_end is not None:
        train_feat_end = pd.Timestamp(rf_train_end)
    else:
        train_feat_end = prediction_time - pd.DateOffset(months=future_start_months + delta_months)
    train_feat_start = train_feat_end - pd.Timedelta(days=window_days)
    train_label_start = train_feat_end + pd.DateOffset(months=future_start_months)
    train_label_end = train_label_start + pd.DateOffset(months=delta_months)

    features_df = extract_features_for_window(
        df,
        pd.Timestamp(train_feat_start),
        pd.Timestamp(train_feat_end),
        pd.Timestamp(train_label_start),
        pd.Timestamp(train_label_end),
    )

    if len(features_df) < 10:
        logger.warning(f"RF: 学習データが少なすぎます ({len(features_df)} samples)")
        return None

    X_train, y_train = prepare_rf_features(features_df)

    if len(np.unique(y_train)) < 2:
        logger.warning("RF: 学習データに正例/負例の両方が必要")
        return None

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    logger.info(
        f"RF: 学習完了 ({len(X_train)} samples, "
        f"pos={int(y_train.sum())}, neg={int(len(y_train) - y_train.sum())})"
    )
    return clf


def baseline_rf(
    df: pd.DataFrame,
    prediction_time: datetime,
    delta_months: int,
    window_days: int,
    target_dirs: Dict[str, Set[str]],
    future_start_months: int = 0,
    rf_train_end: Optional[datetime] = None,
    n_jobs: int = -1,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    RF ベースライン: 卒論の RF で個人の continuation_prob を予測し、
    IRL Variant A と同じように Σ prob でディレクトリ単位に集約する。

    Returns:
        (dir_predictions, developer_probs):
            dir_predictions: {directory: predicted_count}
            developer_probs: {email: continuation_prob}
    """
    clf = _train_rf_classifier(df, prediction_time, window_days, delta_months, future_start_months, rf_train_end)

    if clf is None:
        naive = baseline_naive(df, prediction_time, window_days)
        return {d: float(c) for d, c in naive.items()}, {}

    feature_start = prediction_time - pd.Timedelta(days=window_days)

    # 全開発者の continuation_prob を RF で推論（並列）
    all_devs = set()
    for devs in target_dirs.values():
        all_devs.update(devs)

    def _rf_predict_one(email):
        feat_dict = extract_common_features(
            df, email, feature_start, prediction_time, normalize=False,
        )
        X = np.array([[feat_dict[f] for f in FEATURE_NAMES]], dtype=np.float64)
        return email, float(clf.predict_proba(X)[0, 1])

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_rf_predict_one)(email) for email in all_devs
    )
    rf_probs: Dict[str, float] = dict(results)

    logger.info(
        f"RF: {len(rf_probs)} devs 推論完了, "
        f"avg_prob={np.mean(list(rf_probs.values())):.3f}"
    )

    # ディレクトリ単位で Σ prob（IRL Variant A と同じ集約）
    result: Dict[str, float] = {}
    for d, devs in target_dirs.items():
        result[d] = sum(rf_probs.get(dev, 0.5) for dev in devs)

    return result, rf_probs


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


# ── ディレクトリ×個人の二値分類評価 ──────────────────────────────


def compute_per_developer_classification(
    dir_developers: Dict[str, Set[str]],
    probs: Dict[str, float],
    actual_dir_developers: Dict[str, Set[str]],
    method_name: str = "",
) -> Dict[str, float]:
    """
    卒論と同じ指標（AUC-ROC, AUC-PR, F1 等）を
    ディレクトリ × 個人レベルで計算する。

    各 (developer, directory) ペアについて:
      - 予測: continuation_prob（0〜1）
      - 正解: T+Δ 期間にそのディレクトリに実際に貢献したか（0/1）

    Args:
        dir_developers: 予測時点 T での {directory: {dev1, dev2, ...}}
        probs: {email: continuation_prob}
        actual_dir_developers: T+Δ 期間での {directory: {dev1, dev2, ...}}
    """
    from scipy import stats
    from sklearn.metrics import (
        auc,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_list: List[int] = []
    y_prob_list: List[float] = []

    for d, devs in dir_developers.items():
        actual_devs = actual_dir_developers.get(d, set())
        for dev in devs:
            label = 1 if dev in actual_devs else 0
            prob = probs.get(dev, 0.5)
            y_true_list.append(label)
            y_prob_list.append(prob)

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)

    n_pairs = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_pairs - n_pos

    if n_pos == 0 or n_neg == 0:
        logger.warning(f"[{method_name}] 正例 or 負例が 0 件 (pos={n_pos}, neg={n_neg})")
        return {
            "auc_roc": float("nan"), "auc_pr": float("nan"),
            "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "n_pairs": float(n_pairs), "n_pos": float(n_pos), "n_neg": float(n_neg),
        }

    # AUC-ROC
    try:
        auc_roc_val = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_roc_val = float("nan")

    # AUC-PR
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_true, y_prob)
    auc_pr_val = float(auc(rec_curve, prec_curve))

    # 最適閾値（F1 最大化）
    f1_scores = 2 * (prec_curve * rec_curve) / (prec_curve + rec_curve + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "auc_roc": auc_roc_val,
        "auc_pr": auc_pr_val,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(best_threshold),
        "n_pairs": float(n_pairs),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
    }
    logger.info(
        f"[{method_name} classification] "
        + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    )
    return metrics


# ── 校正 ──────────────────────────────────────────────────────


def calibrate_predictions(
    pair_results: List[Tuple[str, str, float]],
    actual_dir_devs: Dict[str, Set[str]],
    calibration_fraction: float = 0.3,
    random_seed: int = 777,
) -> Tuple[List[Tuple[str, str, float]], pd.DataFrame]:
    """
    IsotonicRegression で IRL_Dir の (dev, dir) 確率を事後校正する。

    Args:
        pair_results: [(directory, developer, raw_prob), ...]
        actual_dir_devs: {directory: {dev_who_actually_contributed, ...}}
        calibration_fraction: フィット用に使うペアの割合
        random_seed: 再現性用シード

    Returns:
        calibrated_pair_results: 校正済み確率に置き換えた pair_results
        calibration_curve: 校正曲線データ (DataFrame)
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedShuffleSplit

    # y_true / y_score を構築
    y_true = np.array([
        1 if dev in actual_dir_devs.get(d, set()) else 0
        for d, dev, _ in pair_results
    ])
    y_score = np.array([prob for _, _, prob in pair_results])

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos < 10 or n_neg < 10:
        logger.warning(f"校正スキップ: 正例={n_pos}, 負例={n_neg} が少なすぎる")
        return pair_results, pd.DataFrame()

    # Stratified split: calibration_fraction をフィット用に使う
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=calibration_fraction,
        random_state=random_seed,
    )
    cal_idx, _ = next(sss.split(y_score, y_true))

    # Isotonic regression をフィット
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(y_score[cal_idx], y_true[cal_idx])

    # 全ペアを変換
    calibrated_scores = iso.transform(y_score)

    calibrated_pair_results = [
        (d, dev, float(calibrated_scores[i]))
        for i, (d, dev, _) in enumerate(pair_results)
    ]

    # 校正曲線データ（reliability diagram 用）
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    curve_rows = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask_raw = (y_score >= lo) & (y_score < hi)
        mask_cal = (calibrated_scores >= lo) & (calibrated_scores < hi)
        curve_rows.append({
            "bin_center": (lo + hi) / 2,
            "fraction_positive_raw": float(y_true[mask_raw].mean()) if mask_raw.sum() > 0 else np.nan,
            "mean_predicted_raw": float(y_score[mask_raw].mean()) if mask_raw.sum() > 0 else np.nan,
            "count_raw": int(mask_raw.sum()),
            "fraction_positive_cal": float(y_true[mask_cal].mean()) if mask_cal.sum() > 0 else np.nan,
            "mean_predicted_cal": float(calibrated_scores[mask_cal].mean()) if mask_cal.sum() > 0 else np.nan,
            "count_cal": int(mask_cal.sum()),
        })
    cal_curve = pd.DataFrame(curve_rows)

    logger.info(
        f"校正完了: フィット={len(cal_idx)}ペア, "
        f"raw_mean={y_score.mean():.3f} → cal_mean={calibrated_scores.mean():.3f}"
    )
    return calibrated_pair_results, cal_curve


# ── メイン ────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="パスごとの貢献者数予測評価")
    p.add_argument("--data", type=Path, default=Path("data/nova_raw.csv"))
    p.add_argument(
        "--raw-json",
        type=str,
        nargs="+",
        default=["data/raw_json/openstack__nova.json"],
    )
    p.add_argument(
        "--irl-model",
        type=Path,
        default=Path("outputs/cross_temporal_v39/train_0-3m/irl_model.pt"),
    )
    p.add_argument(
        "--irl-dir-model",
        type=Path,
        default=None,
        help="ディレクトリ単位IRLモデルのパス（state_dim=23）",
    )
    p.add_argument("--prediction-time", type=str, default="2014-01-01")
    p.add_argument("--delta-months", type=int, default=3)
    p.add_argument("--future-start-months", type=int, default=0,
                   help="将来窓の開始オフセット（卒論: 0,3,6,9）")
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
    p.add_argument("--rf-future-start-months", type=int, default=None,
                   help="RFの訓練ラベル窓の開始オフセット（未指定時は--future-start-monthsと同じ）")
    p.add_argument("--rf-train-end", type=str, default=None,
                   help="RF_Dirの学習ラベル基準日（IRL train_endと揃える）")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="joblib 並列数（-1=全コア）")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--multi-timepoint",
        action="store_true",
        help="複数時点で評価（3ヶ月刻み）",
    )
    p.add_argument(
        "--save-importance",
        action="store_true",
        help="IRL gradient importance と RF_Dir Gini importance を保存する",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="IsotonicRegression で IRL_Dir 出力を事後校正する",
    )
    p.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.3,
        help="校正フィットに使う (dev, dir) ペアの割合（残りで評価）",
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
    irl_dir_model_path: Optional[Path] = None,
    future_start_months: int = 0,
    rf_train_end: Optional[datetime] = None,
    rf_future_start_months: Optional[int] = None,
    n_jobs: int = -1,
    save_importance: bool = False,
    output_dir: Optional[Path] = None,
    calibrate: bool = False,
    calibration_fraction: float = 0.3,
) -> Dict[str, Dict[str, float]]:
    """単一時点の評価を実行し、全手法の結果を返す。"""
    # RF訓練ラベル窓（未指定なら評価窓と同じ = 従来動作）
    if rf_future_start_months is None:
        rf_future_start_months = future_start_months

    logger.info(f"=== 予測時点 T={prediction_time}, 将来窓={future_start_months}-{future_start_months+delta_months}ヶ月 ===")

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

    continuation_probs: Dict[str, float] = {}
    variant_a: Dict[str, float] = {}
    variant_b: Dict[str, float] = {}
    if predictor is not None:
        continuation_probs = predictor.predict_batch(
            list(all_devs), prediction_time
        )
        # 3. IRL 予測 (Variant A / B)
        variant_a, variant_b = predict_contributor_counts(
            dir_developers, continuation_probs, path_extractor, prediction_time
        )
    else:
        logger.info("グローバルIRLモデルなし: IRL_VariantA/B をスキップ")

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

    # 4. ベースライン（Linear, RF）
    linear_pred = baseline_linear(df, prediction_time, delta_months)
    rf_pred, rf_probs = baseline_rf(
        df, prediction_time, delta_months, window_days, dir_developers,
        future_start_months=rf_future_start_months,
        rf_train_end=rf_train_end,
        n_jobs=n_jobs,
    )

    # 5. Ground truth
    future_start = pred_time_pd + pd.DateOffset(months=future_start_months)
    future_end = pred_time_pd + pd.DateOffset(months=future_start_months + delta_months)
    actual = count_actual_contributors(df, future_start, future_end)
    actual_dir_devs = get_directory_developers(df, future_start, future_end)
    logger.info(
        f"Ground truth: {len(actual)} dirs, "
        f"avg={np.mean(list(actual.values())):.1f} devs/dir"
    )

    # 6. 評価
    all_results: Dict[str, Dict[str, float]] = {}

    methods = {
        "Naive": naive_pred,
        "Linear": linear_pred,
        "RF": rf_pred,
        # NOTE: グローバルIRLモデル未配線のため一旦除外
        # "IRL_VariantA": variant_a,
        # "IRL_VariantA_scaled": variant_a_scaled,
        # "IRL_VariantB": variant_b,
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

    # 6.5 ディレクトリ×個人の二値分類評価（卒論と同じ指標）
    clf_methods = {
        # NOTE: グローバルIRLモデル未配線のため一旦除外
        # "IRL_VariantA": continuation_probs,
        "RF": rf_probs,
    }

    # ── IRL_Dir: ディレクトリ単位 IRL モデル ──
    irl_dir_probs: Dict[str, Dict[str, float]] = {}  # {dir: {email: prob}}
    if irl_dir_model_path and irl_dir_model_path.exists():
        logger.info("IRL_Dir: ディレクトリ単位モデルで推論...")
        # 訓練時と同じ窓幅（24ヶ月）で推論する
        # 訓練: 2021-01 ~ 2023-01 (24ヶ月)
        # 推論: prediction_time - 24ヶ月 ~ prediction_time
        dir_history_start = (
            pd.Timestamp(prediction_time) - pd.DateOffset(months=24)
        ).to_pydatetime()
        dir_predictor = BatchContinuationPredictor(
            model_path=irl_dir_model_path,
            df=df,
            history_start=dir_history_start,
        )
        # (dev, dir) ペアをフラット化して一括推論
        head_index = future_start_months // 3
        pairs = [(d, dev) for d, devs in dir_developers.items() for dev in devs]
        logger.info(f"IRL_Dir: {len(pairs)} ペアを推論...")

        pair_results = []
        for d, dev in pairs:
            prob = dir_predictor.predict_developer_directory(
                dev, d, prediction_time, path_extractor=path_extractor,
                head_index=head_index,
            )
            pair_results.append((d, dev, prob))

        irl_dir_agg: Dict[str, float] = {}
        for d, dev, prob in pair_results:
            irl_dir_probs.setdefault(d, {})[dev] = prob
            irl_dir_agg[d] = irl_dir_agg.get(d, 0.0) + prob

        methods["IRL_Dir"] = irl_dir_agg
        metrics = compute_metrics(irl_dir_agg, actual, "IRL_Dir")
        danger = compute_danger_detection(
            irl_dir_agg, actual,
            threshold=danger_threshold,
            danger_actual_threshold=danger_actual_threshold,
            method_name="IRL_Dir",
        )
        all_results["IRL_Dir"] = {**metrics, **{f"danger_{k}": v for k, v in danger.items()}}
        # IRL_Dir の (dev, dir) 確率を clf_methods に追加
        irl_dir_flat: Dict[str, float] = {}
        for d, dev_probs in irl_dir_probs.items():
            for dev, prob in dev_probs.items():
                # ディレクトリごとに異なるので、キーを (dev, dir) タプルにする必要がある
                # → 専用の分類評価を行う
                pass
        # IRL_Dir 用の分類評価（ディレクトリごとに prob が異なるので専用処理）
        y_true_list_dir: List[int] = []
        y_prob_list_dir: List[float] = []
        for d, devs in dir_developers.items():
            actual_devs = actual_dir_devs.get(d, set())
            for dev in devs:
                label = 1 if dev in actual_devs else 0
                prob = irl_dir_probs.get(d, {}).get(dev, 0.5)
                y_true_list_dir.append(label)
                y_prob_list_dir.append(prob)
        if y_true_list_dir:
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
            from sklearn.metrics import auc as sk_auc
            y_true_arr = np.array(y_true_list_dir)
            y_prob_arr = np.array(y_prob_list_dir)
            n_pos = int(y_true_arr.sum())
            n_neg = len(y_true_arr) - n_pos
            if n_pos > 0 and n_neg > 0:
                auc_roc = float(roc_auc_score(y_true_arr, y_prob_arr))
                prec_c, rec_c, thr = precision_recall_curve(y_true_arr, y_prob_arr)
                auc_pr = float(sk_auc(rec_c, prec_c))
                f1s = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-10)
                bi = np.argmax(f1s)
                bt = thr[bi] if bi < len(thr) else 0.5
                yp = (y_prob_arr >= bt).astype(int)
                clf_m = {
                    "auc_roc": auc_roc, "auc_pr": auc_pr,
                    "f1": float(f1_score(y_true_arr, yp, zero_division=0)),
                    "precision": float(precision_score(y_true_arr, yp, zero_division=0)),
                    "recall": float(recall_score(y_true_arr, yp, zero_division=0)),
                    "threshold": float(bt),
                    "n_pairs": float(len(y_true_arr)),
                    "n_pos": float(n_pos), "n_neg": float(n_neg),
                }
                all_results["IRL_Dir"] = {
                    **all_results.get("IRL_Dir", {}),
                    **{f"clf_{k}": v for k, v in clf_m.items()},
                }
                logger.info(
                    f"[IRL_Dir classification] "
                    + " ".join(f"{k}={v:.3f}" for k, v in clf_m.items())
                )

        # ── IRL_Dir 校正版 ──
        if calibrate and pair_results:
            calibrated_pair_results, cal_curve = calibrate_predictions(
                pair_results, actual_dir_devs, calibration_fraction,
            )
            # 校正済みで集計
            irl_dir_probs_cal: Dict[str, Dict[str, float]] = {}
            irl_dir_agg_cal: Dict[str, float] = {}
            for d, dev, prob in calibrated_pair_results:
                irl_dir_probs_cal.setdefault(d, {})[dev] = prob
                irl_dir_agg_cal[d] = irl_dir_agg_cal.get(d, 0.0) + prob

            methods["IRL_Dir_calibrated"] = irl_dir_agg_cal
            metrics_cal = compute_metrics(irl_dir_agg_cal, actual, "IRL_Dir_calibrated")
            danger_cal = compute_danger_detection(
                irl_dir_agg_cal, actual,
                threshold=danger_threshold,
                danger_actual_threshold=danger_actual_threshold,
                method_name="IRL_Dir_calibrated",
            )
            all_results["IRL_Dir_calibrated"] = {
                **metrics_cal,
                **{f"danger_{k}": v for k, v in danger_cal.items()},
            }
            # 校正版の分類評価
            y_true_cal: List[int] = []
            y_prob_cal: List[float] = []
            for d, devs in dir_developers.items():
                actual_devs_d = actual_dir_devs.get(d, set())
                for dev in devs:
                    y_true_cal.append(1 if dev in actual_devs_d else 0)
                    y_prob_cal.append(irl_dir_probs_cal.get(d, {}).get(dev, 0.5))
            if y_true_cal:
                from sklearn.metrics import roc_auc_score as _roc, f1_score as _f1
                from sklearn.metrics import precision_score as _ps, recall_score as _rs
                from sklearn.metrics import precision_recall_curve as _prc
                from sklearn.metrics import auc as _auc
                _yt = np.array(y_true_cal)
                _yp = np.array(y_prob_cal)
                _np = int(_yt.sum()); _nn = len(_yt) - _np
                if _np > 0 and _nn > 0:
                    _ar = float(_roc(_yt, _yp))
                    _pc, _rc, _th = _prc(_yt, _yp)
                    _ap = float(_auc(_rc, _pc))
                    _f1s = 2 * (_pc * _rc) / (_pc + _rc + 1e-10)
                    _bi = np.argmax(_f1s)
                    _bt = _th[_bi] if _bi < len(_th) else 0.5
                    _pred = (_yp >= _bt).astype(int)
                    _cm = {
                        "auc_roc": _ar, "auc_pr": _ap,
                        "f1": float(_f1(_yt, _pred, zero_division=0)),
                        "precision": float(_ps(_yt, _pred, zero_division=0)),
                        "recall": float(_rs(_yt, _pred, zero_division=0)),
                        "threshold": float(_bt),
                        "n_pairs": float(len(_yt)),
                        "n_pos": float(_np), "n_neg": float(_nn),
                    }
                    all_results["IRL_Dir_calibrated"] = {
                        **all_results.get("IRL_Dir_calibrated", {}),
                        **{f"clf_{k}": v for k, v in _cm.items()},
                    }
                    logger.info(
                        f"[IRL_Dir_calibrated classification] "
                        + " ".join(f"{k}={v:.3f}" for k, v in _cm.items())
                    )
            # 校正曲線の保存
            if output_dir is not None and len(cal_curve) > 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                cal_path = Path(output_dir) / "calibration_curve.csv"
                cal_curve.to_csv(cal_path, index=False)
                logger.info(f"校正曲線を保存: {cal_path}")

        # IRL gradient importance を保存
        if save_importance and output_dir is not None:
            try:
                import torch as _torch
                from review_predictor.IRL.features.common_features import (
                    STATE_FEATURES as _SF, ACTION_FEATURES as _AF,
                )
                from review_predictor.IRL.features.path_features import PATH_FEATURE_NAMES as _PFN
                _irl = dir_predictor._irl_system
                _is_dir = _irl.state_dim > len(_SF)
                _all_grads_abs = []
                _all_grads_signed = []
                _sample_devs = list(all_devs)[:300]
                for _dev in _sample_devs:
                    _md = dir_predictor._build_monthly_data(_dev, prediction_time)
                    _mh, _sd, _sr, _di = _md
                    if not _mh or _di is None:
                        continue
                    # 代表ディレクトリ1つを選択
                    _dev_dirs = [d for d, devs in dir_developers.items() if _dev in devs]
                    _d = _dev_dirs[0] if _dev_dirs else None
                    _spf = None
                    if _is_dir and _d:
                        _task = frozenset({_d})
                        _spf = [path_extractor.compute(_dev, _task, cd) for cd in _sd]
                    _email = _di.get("email", _di.get("developer_id", _dev))
                    _st, _at = [], []
                    for _i, (_mhi, _mctx) in enumerate(zip(_mh, _sd)):
                        if not _mhi:
                            _st.append(_torch.zeros(_irl.state_dim))
                            _at.append(_torch.zeros(_irl.action_dim))
                            continue
                        _tp = _sr[_i] if _sr and _i < len(_sr) else 0
                        _pf = _spf[_i] if _spf and _i < len(_spf) else None
                        _s, _a = _irl.extract_features_tensor(
                            _email, _mhi, _mctx, total_project_reviews=_tp,
                            path_features_vec=_pf,
                        )
                        _st.append(_s); _at.append(_a)
                    if not _st:
                        continue
                    _ss = _torch.stack(_st).unsqueeze(0).requires_grad_(True)
                    _as = _torch.stack(_at).unsqueeze(0).requires_grad_(True)
                    _ln = _torch.tensor([len(_st)], dtype=_torch.long)
                    _, _cont = _irl.network(_ss, _as, _ln)
                    _cont.backward()
                    _sg_signed = _ss.grad.mean(dim=(0, 1)).detach().cpu().numpy()
                    _ag_signed = _as.grad.mean(dim=(0, 1)).detach().cpu().numpy()
                    _all_grads_signed.append(np.concatenate([_sg_signed, _ag_signed]))
                    _sg_abs = _ss.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
                    _ag_abs = _as.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
                    _all_grads_abs.append(np.concatenate([_sg_abs, _ag_abs]))
                if _all_grads_abs:
                    _sn = list(_SF) + (list(_PFN) if _is_dir else [])
                    _names = _sn + list(_AF)
                    import json as _json
                    # 絶対値版
                    _mg = np.mean(_all_grads_abs, axis=0)
                    _tot = _mg.sum()
                    if _tot > 0:
                        _mg = _mg / _tot
                    _imp = {n: float(v) for n, v in zip(_names, _mg)}
                    _imp_path = Path(output_dir) / "irl_feature_importance.json"
                    with open(_imp_path, "w") as f:
                        _json.dump(_imp, f, indent=2, ensure_ascii=False)
                    logger.info(f"IRL importance (abs) を保存: {_imp_path} ({len(_all_grads_abs)} samples)")
                    # 符号付き版
                    _mg_s = np.mean(_all_grads_signed, axis=0)
                    _imp_s = {n: float(v) for n, v in zip(_names, _mg_s)}
                    _imp_s_path = Path(output_dir) / "irl_feature_importance_signed.json"
                    with open(_imp_s_path, "w") as f:
                        _json.dump(_imp_s, f, indent=2, ensure_ascii=False)
                    logger.info(f"IRL importance (signed) を保存: {_imp_s_path}")
            except Exception as e:
                logger.warning(f"IRL importance 計算で例外: {e}")

    # ── RF_Dir: ディレクトリ単位 RF ──
    logger.info("RF_Dir: ディレクトリ単位 RF で推論...")
    if rf_train_end is not None:
        # IRL と同じ train_end を使う（公平な比較）
        _rf_base = pd.Timestamp(rf_train_end)
    else:
        # デフォルト: prediction_time から逆��
        _rf_base = pred_time_pd - pd.DateOffset(months=future_start_months + delta_months)
    rf_dir_train_end = _rf_base
    rf_dir_train_start = rf_dir_train_end - pd.Timedelta(days=window_days)
    rf_dir_label_start = rf_dir_train_end + pd.DateOffset(months=rf_future_start_months)
    rf_dir_label_end = rf_dir_label_start + pd.DateOffset(months=delta_months)

    rf_dir_train_df = extract_features_for_window_directory(
        df,
        pd.Timestamp(rf_dir_train_start),
        pd.Timestamp(rf_dir_train_end),
        pd.Timestamp(rf_dir_label_start),
        pd.Timestamp(rf_dir_label_end),
        path_extractor=path_extractor,
    )
    if len(rf_dir_train_df) >= 10 and len(rf_dir_train_df["label"].unique()) >= 2:
        from sklearn.ensemble import RandomForestClassifier as RFC
        X_train_dir, y_train_dir = prepare_rf_features_directory(rf_dir_train_df)
        clf_dir = RFC(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        clf_dir.fit(X_train_dir, y_train_dir)

        # RF_Dir Gini importance を保存
        if save_importance and output_dir is not None:
            from review_predictor.IRL.features.common_features import FEATURE_NAMES_WITH_PATH as _FN_PATH
            rf_importance = {
                name: float(val)
                for name, val in zip(_FN_PATH, clf_dir.feature_importances_)
            }
            import json as _json
            rf_imp_path = Path(output_dir) / "rf_dir_feature_importance.json"
            with open(rf_imp_path, "w") as f:
                _json.dump({"feature_importance": rf_importance}, f, indent=2, ensure_ascii=False)
            logger.info(f"RF_Dir importance を保存: {rf_imp_path}")

        # 推論: 各 (dev, dir) ペアの continuation_prob（並列）
        from review_predictor.IRL.features.common_features import FEATURE_NAMES_WITH_PATH
        from review_predictor.IRL.features.path_features import PATH_FEATURE_NAMES

        feature_start = prediction_time - pd.Timedelta(days=window_days)
        rf_dir_agg: Dict[str, float] = {}
        rf_dir_dev_probs: Dict[str, Dict[str, float]] = {}

        rf_dir_pairs = [(d, dev) for d, devs in dir_developers.items() for dev in devs]
        logger.info(f"RF_Dir: {len(rf_dir_pairs)} ペアを並列推論 (n_jobs={n_jobs})...")

        def _rf_dir_predict_one(d, dev):
            common_feats = extract_common_features(
                df, dev, feature_start, prediction_time, normalize=False,
            )
            pf = path_extractor.compute(dev, frozenset({d}), prediction_time)
            path_feats = {
                name: float(val) for name, val in zip(PATH_FEATURE_NAMES, pf)
            }
            feat_row = {**common_feats, **path_feats}
            X = np.array([[feat_row.get(f, 0.0) for f in FEATURE_NAMES_WITH_PATH]], dtype=np.float64)
            prob = float(clf_dir.predict_proba(X)[0, 1])
            return d, dev, prob

        rf_dir_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_rf_dir_predict_one)(d, dev) for d, dev in rf_dir_pairs
        )
        for d, dev, prob in rf_dir_results:
            rf_dir_dev_probs.setdefault(d, {})[dev] = prob
            rf_dir_agg[d] = rf_dir_agg.get(d, 0.0) + prob

        methods["RF_Dir"] = rf_dir_agg
        metrics = compute_metrics(rf_dir_agg, actual, "RF_Dir")
        danger = compute_danger_detection(
            rf_dir_agg, actual,
            threshold=danger_threshold,
            danger_actual_threshold=danger_actual_threshold,
            method_name="RF_Dir",
        )
        all_results["RF_Dir"] = {**metrics, **{f"danger_{k}": v for k, v in danger.items()}}

        # RF_Dir 分類評価
        y_true_rf: List[int] = []
        y_prob_rf: List[float] = []
        for d, devs in dir_developers.items():
            actual_devs = actual_dir_devs.get(d, set())
            for dev in devs:
                y_true_rf.append(1 if dev in actual_devs else 0)
                y_prob_rf.append(rf_dir_dev_probs.get(d, {}).get(dev, 0.5))
        if y_true_rf:
            clf_m_rf = compute_per_developer_classification(
                dir_developers,
                # compute_per_developer_classification はグローバルprobを期待するので
                # 直接 y_true/y_prob を使う
                {}, actual_dir_devs, method_name="RF_Dir",
            )
            # 上の関数はグローバルprob前提なので、直接計算する
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
            from sklearn.metrics import auc as sk_auc
            y_t = np.array(y_true_rf)
            y_p = np.array(y_prob_rf)
            n_p = int(y_t.sum())
            n_n = len(y_t) - n_p
            if n_p > 0 and n_n > 0:
                auc_roc = float(roc_auc_score(y_t, y_p))
                prec_c, rec_c, thr = precision_recall_curve(y_t, y_p)
                auc_pr = float(sk_auc(rec_c, prec_c))
                f1s = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-10)
                bi = np.argmax(f1s)
                bt = thr[bi] if bi < len(thr) else 0.5
                yp = (y_p >= bt).astype(int)
                clf_m = {
                    "auc_roc": auc_roc, "auc_pr": auc_pr,
                    "f1": float(f1_score(y_t, yp, zero_division=0)),
                    "precision": float(precision_score(y_t, yp, zero_division=0)),
                    "recall": float(recall_score(y_t, yp, zero_division=0)),
                    "threshold": float(bt),
                    "n_pairs": float(len(y_t)),
                    "n_pos": float(n_p), "n_neg": float(n_n),
                }
                all_results["RF_Dir"] = {
                    **all_results.get("RF_Dir", {}),
                    **{f"clf_{k}": v for k, v in clf_m.items()},
                }
                logger.info(
                    f"[RF_Dir classification] "
                    + " ".join(f"{k}={v:.3f}" for k, v in clf_m.items())
                )
    else:
        logger.warning(f"RF_Dir: 学習データ不足 ({len(rf_dir_train_df)} samples)")

    # 既存手法のグローバル prob による分類評価
    for name, probs in clf_methods.items():
        if not probs:
            continue
        clf_metrics = compute_per_developer_classification(
            dir_developers, probs, actual_dir_devs, method_name=name,
        )
        all_results[name] = {
            **all_results.get(name, {}),
            **{f"clf_{k}": v for k, v in clf_metrics.items()},
        }

    # 7. ディレクトリ別詳細テーブル
    rows = []
    for d in sorted(set(actual.keys()) | set(variant_a.keys())):
        rows.append(
            {
                "directory": d,
                "actual": actual.get(d, 0),
                "naive": naive_pred.get(d, 0.0),
                "linear": linear_pred.get(d, 0.0),
                "rf": rf_pred.get(d, 0.0),
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

    # 8. (developer, directory) ペアごとの予測結果を CSV 保存
    # rf_dir_dev_probs がスコープ内に存在しない場合は空辞書
    _rf_dir_dev_probs = rf_dir_dev_probs if 'rf_dir_dev_probs' in dir() else {}
    _irl_dir_probs_cal = irl_dir_probs_cal if 'irl_dir_probs_cal' in dir() else {}
    pair_rows: List[Dict] = []
    for d, devs in dir_developers.items():
        actual_devs = actual_dir_devs.get(d, set())
        for dev in devs:
            row = {
                "developer": dev,
                "directory": d,
                "label": 1 if dev in actual_devs else 0,
                "irl_dir_prob": irl_dir_probs.get(d, {}).get(dev, None),
                "irl_dir_prob_calibrated": _irl_dir_probs_cal.get(d, {}).get(dev, None),
                "rf_dir_prob": _rf_dir_dev_probs.get(d, {}).get(dev, None),
                "irl_global_prob": continuation_probs.get(dev, None),
                "rf_global_prob": rf_probs.get(dev, None),
            }
            pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)
    all_results["_pair_predictions"] = pair_df

    return all_results


def main() -> None:
    args = parse_args()

    # データロード
    logger.info(f"データ読み込み: {args.data}")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ディレクトリマッピング
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=args.dir_depth)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=args.dir_depth)
    df = attach_dirs_to_df(df, cdm)

    # PathFeatureExtractor
    path_extractor = PathFeatureExtractor(df, window_days=args.path_window_days)

    # BatchContinuationPredictor
    prediction_time = datetime.fromisoformat(args.prediction_time)
    # history_start: データの最初から
    history_start = df["timestamp"].min().to_pydatetime()
    if args.irl_model.exists():
        predictor = BatchContinuationPredictor(
            model_path=args.irl_model,
            df=df,
            history_start=history_start,
            device=args.device,
        )
    else:
        predictor = None
        logger.warning(f"グローバルIRLモデルが見つからない: {args.irl_model}（スキップ）")

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
                irl_dir_model_path=args.irl_dir_model,
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
        rf_train_end_dt = (
            datetime.fromisoformat(args.rf_train_end)
            if args.rf_train_end else None
        )
        rf_fsm = args.rf_future_start_months if args.rf_future_start_months is not None else args.future_start_months
        results = evaluate_single_timepoint(
            df,
            path_extractor,
            predictor,
            prediction_time,
            args.delta_months,
            args.window_days,
            args.danger_threshold,
            args.danger_actual_threshold,
            irl_dir_model_path=args.irl_dir_model,
            future_start_months=args.future_start_months,
            rf_train_end=rf_train_end_dt,
            rf_future_start_months=rf_fsm,
            n_jobs=args.n_jobs,
            save_importance=args.save_importance,
            output_dir=args.output_dir,
            calibrate=args.calibrate,
            calibration_fraction=args.calibration_fraction,
        )

        # CSV 保存
        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            # ペアごとの予測結果
            pair_df = results.pop("_pair_predictions", None)
            if pair_df is not None and len(pair_df) > 0:
                pair_path = out / "pair_predictions.csv"
                pair_df.to_csv(pair_path, index=False)
                logger.info(f"ペア予測結果を保存: {pair_path} ({len(pair_df)} 行)")

            # サマリ指標
            import json
            summary = {k: v for k, v in results.items() if isinstance(v, dict)}
            summary_path = out / "summary_metrics.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"サマリ指標を保存: {summary_path}")


if __name__ == "__main__":
    main()
