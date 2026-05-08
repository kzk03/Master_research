#!/usr/bin/env python3
"""マルチクラス accept action 版 MCE-IRL の (dev, dir) / dev レベル評価
(Plan B-1 Phase 4)

`MCEEventMulticlassBatchPredictor` を使い、

  - (dev, dir) 二値分類 AUC: π(a = class_id(target_dir) | s_last)
    (本当に T 〜 T+Δ にその dir で承諾したかが正解)
  - dev レベル AUC: π(a ≠ 0 | s_last)
    (T 〜 T+Δ に少なくとも 1 つ accept したかが正解)

を 1 度の実行で出す。

既存 eval_mce_event_irl_path_prediction.py (1300行) の RF / Naive / Linear などは含めず、
IRL_Dir のみに絞った薄い評価スクリプトとして実装する。

使い方:
    uv run python scripts/analyze/eval_mce_event_irl_multiclass_prediction.py \\
        --data data/combined_raw.csv \\
        --raw-json data/raw_json/openstack__*.json \\
        --model outputs/mce_pilot_event_dev_multiclass/event_cold/mce_event_irl_model.pt \\
        --dir-class-mapping outputs/dir_class_mapping_K15.json \\
        --prediction-time 2023-01-01 \\
        --delta-months 3 \\
        --window-days 180 \\
        --output-dir outputs/mce_pilot_event_dev_multiclass/eval_event_cold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCE-IRL multiclass モデルの (dev, dir) / dev レベル評価"
    )
    p.add_argument("--data", type=str, required=True, help="combined_raw.csv")
    p.add_argument(
        "--raw-json", type=str, nargs="+", required=True,
        help="dirs マッピング用の Gerrit raw JSON",
    )
    p.add_argument("--model", type=str, required=True, help="mce_event_irl_model.pt")
    p.add_argument(
        "--dir-class-mapping",
        type=str,
        default=None,
        help="dir → class_id JSON (省略時は model_metadata.json から読み取り)",
    )
    p.add_argument(
        "--prediction-time", type=str, required=True,
        help="予測時点 T (例: 2023-01-01)",
    )
    p.add_argument("--delta-months", type=int, default=3)
    p.add_argument("--future-start-months", type=int, default=0)
    p.add_argument("--window-days", type=int, default=180)
    p.add_argument("--history-months", type=int, default=24)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--skip-pair-eval", action="store_true",
        help="(dev, dir) 評価をスキップし dev レベルのみ計算",
    )
    p.add_argument(
        "--skip-dev-eval", action="store_true",
        help="dev レベル評価をスキップし (dev, dir) のみ計算",
    )
    p.add_argument(
        "--use-beta", action="store_true",
        help="β 戦略: 最終 step の path_features を target_dir で上書きして推論",
    )
    return p.parse_args()


def _binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method_name: str,
) -> Dict[str, float]:
    from sklearn.metrics import (
        auc as sk_auc,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    out: Dict[str, float] = {
        "n_pairs": float(n),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
    }
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "[%s] 正例 or 負例が 0 件 (pos=%d, neg=%d) → AUC スキップ",
            method_name, n_pos, n_neg,
        )
        out.update({
            "clf_auc_roc": float("nan"),
            "clf_auc_pr": float("nan"),
            "clf_f1": float("nan"),
            "clf_precision": float("nan"),
            "clf_recall": float("nan"),
            "clf_threshold": float("nan"),
        })
        return out
    auc_roc = float(roc_auc_score(y_true, y_prob))
    prec_c, rec_c, thr = precision_recall_curve(y_true, y_prob)
    auc_pr = float(sk_auc(rec_c, prec_c))
    f1s = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-10)
    bi = int(np.argmax(f1s))
    bt = float(thr[bi]) if bi < len(thr) else 0.5
    yp = (y_prob >= bt).astype(int)
    out.update({
        "clf_auc_roc": auc_roc,
        "clf_auc_pr": auc_pr,
        "clf_f1": float(f1_score(y_true, yp, zero_division=0)),
        "clf_precision": float(precision_score(y_true, yp, zero_division=0)),
        "clf_recall": float(recall_score(y_true, yp, zero_division=0)),
        "clf_threshold": bt,
        "prob_mean": float(y_prob.mean()),
        "prob_pos_mean": float(y_prob[y_true == 1].mean()) if n_pos > 0 else float("nan"),
        "prob_neg_mean": float(y_prob[y_true == 0].mean()) if n_neg > 0 else float("nan"),
    })
    logger.info(
        "[%s] auc_roc=%.4f auc_pr=%.4f f1=%.3f thr=%.3f n=%d (pos=%d, neg=%d)",
        method_name, out["clf_auc_roc"], out["clf_auc_pr"],
        out["clf_f1"], out["clf_threshold"], n, n_pos, n_neg,
    )
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from review_predictor.IRL.features.path_features import (
        PathFeatureExtractor,
        attach_dirs_to_df,
        load_change_dir_map,
        load_change_dir_map_multi,
    )
    from review_predictor.IRL.features.directory_contributors import (
        count_actual_contributors,
        get_directory_developers,
    )
    from review_predictor.IRL.model.mce_event_irl_batch_predictor_multiclass import (
        MCEEventMulticlassBatchPredictor,
    )

    pred_time = pd.Timestamp(args.prediction_time)
    delta = pd.DateOffset(months=args.delta_months)
    future_start = pred_time + pd.DateOffset(months=args.future_start_months)
    future_end = future_start + delta
    window_start = pred_time - pd.Timedelta(days=args.window_days)
    history_start = (
        pred_time - pd.DateOffset(months=args.history_months)
    ).to_pydatetime()

    # ── データ読み込み ──
    logger.info("データ読み込み: %s", args.data)
    df = pd.read_csv(args.data, low_memory=False)
    if "request_time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"request_time": "timestamp"})
    if "reviewer_email" in df.columns and "email" not in df.columns:
        df = df.rename(columns={"reviewer_email": "email"})
    if "timestamp" not in df.columns:
        raise ValueError("data に timestamp 列がありません")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── ディレクトリマッピング (depth=2 軌跡用) ──
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    df = attach_dirs_to_df(df, cdm, column="dirs")
    path_extractor = PathFeatureExtractor(df, window_days=180)

    # ── predictor 初期化 ──
    predictor = MCEEventMulticlassBatchPredictor(
        model_path=args.model,
        df=df,
        history_start=history_start,
        device=args.device,
        reviewer_col="email",
        date_col="timestamp",
        label_col="label",
        dirs_column="dirs",
        dir_class_mapping_path=args.dir_class_mapping,
    )

    summary: Dict[str, Dict] = {}

    # ════════════════════════════════════════════════════════════════
    # Part 1: (dev, dir) 評価
    # ════════════════════════════════════════════════════════════════

    pair_results: List[Dict] = []
    if not args.skip_pair_eval:
        logger.info("=" * 70)
        logger.info("(dev, dir) 評価")
        logger.info("=" * 70)
        # 候補 (dev, dir) ペアを抽出: T 時点で過去 window_days に活動したペア
        dir_developers: Dict[str, Set[str]] = get_directory_developers(
            df, window_start, pred_time
        )
        actual_dir_developers: Dict[str, Set[str]] = get_directory_developers(
            df, future_start, future_end
        )
        n_pairs_total = sum(len(v) for v in dir_developers.values())
        logger.info(
            "対象 dirs=%d, 候補 (dev, dir) ペア=%d",
            len(dir_developers), n_pairs_total,
        )

        # 重複 dev 推論を抑えるため、reviewer 単位で 1 回モデル推論し全クラス確率を取得。
        # (dev, target_dir) → π(a = cid(target_dir) | s_last)。
        # ただしこれは per-dev=True 専用の最適化。non-per-dev は従来どおり個別推論する。
        flat_pairs: List[Tuple[str, str]] = []
        for d, devs in dir_developers.items():
            for dev in devs:
                flat_pairs.append((d, dev))

        # 推論実行: per-dev 最適化のために先にモデルロードし、確定した per_dev を見る
        # （コンストラクタ直後の self.per_dev は初期値 False のままなので注意）
        predictor._load_model()
        per_dev_mode = predictor.per_dev

        if args.use_beta and per_dev_mode:
            logger.info(
                "β 戦略モード: %d ペアを個別に推論 (target_dir を path_features に注入)",
                len(flat_pairs),
            )
            for i, (d, dev) in enumerate(flat_pairs):
                p = predictor.predict_developer_directory_with_beta(
                    dev, d, pred_time.to_pydatetime(),
                    path_extractor=path_extractor,
                )
                pair_results.append({
                    "directory": d,
                    "email": dev,
                    "target_class_id": int(predictor._dir_to_class_id(d)),
                    "prob": float(p),
                    "label": int(dev in actual_dir_developers.get(d, set())),
                })
                if (i + 1) % 500 == 0:
                    logger.info("  β 推論 進捗: %d/%d", i + 1, len(flat_pairs))
        elif per_dev_mode:
            unique_devs = sorted({dev for _, dev in flat_pairs})
            logger.info(
                "per-dev モード: %d 個の reviewer で 1 回ずつ推論し全クラス確率を計算します",
                len(unique_devs),
            )
            dev_probs: Dict[str, List[float]] = {}
            for i, dev in enumerate(unique_devs):
                probs = predictor.predict_developer_directory_distribution(
                    dev, pred_time.to_pydatetime(),
                    path_extractor=path_extractor,
                )
                dev_probs[dev] = probs
                if (i + 1) % 200 == 0:
                    logger.info("  分布推論 進捗: %d/%d", i + 1, len(unique_devs))

            for d, dev in flat_pairs:
                cid = predictor._dir_to_class_id(d)
                probs = dev_probs.get(dev, [])
                if not probs:
                    p = 0.5
                else:
                    p = probs[cid] if 0 <= cid < len(probs) else 0.0
                pair_results.append({
                    "directory": d,
                    "email": dev,
                    "target_class_id": int(cid),
                    "prob": float(p),
                    "label": int(dev in actual_dir_developers.get(d, set())),
                })
        else:
            logger.info("(dev, dir) ペアモデル: 全ペアを個別に推論")
            for i, (d, dev) in enumerate(flat_pairs):
                p = predictor.predict_developer_directory(
                    dev, d, pred_time.to_pydatetime(),
                    path_extractor=path_extractor,
                )
                pair_results.append({
                    "directory": d,
                    "email": dev,
                    "target_class_id": int(predictor._dir_to_class_id(d)),
                    "prob": float(p),
                    "label": int(dev in actual_dir_developers.get(d, set())),
                })
                if (i + 1) % 1000 == 0:
                    logger.info("  ペア推論 進捗: %d/%d", i + 1, len(flat_pairs))

        # メトリクス計算
        if pair_results:
            y_true = np.array([r["label"] for r in pair_results])
            y_prob = np.array([r["prob"] for r in pair_results])
            metrics_pair = _binary_metrics(y_true, y_prob, "IRL_Dir (multiclass)")
            summary["IRL_Dir"] = metrics_pair

            pair_df = pd.DataFrame(pair_results)
            pair_path = output_dir / "pair_predictions.csv"
            pair_df.to_csv(pair_path, index=False)
            logger.info("ペア予測保存: %s", pair_path)

    # ════════════════════════════════════════════════════════════════
    # Part 2: dev レベル評価
    # ════════════════════════════════════════════════════════════════

    if not args.skip_dev_eval:
        logger.info("=" * 70)
        logger.info("dev レベル評価")
        logger.info("=" * 70)

        candidate_mask = (
            (df["timestamp"] >= window_start) & (df["timestamp"] < pred_time)
        )
        candidate_devs = sorted(df.loc[candidate_mask, "email"].dropna().unique().tolist())
        future_mask = (
            (df["timestamp"] >= future_start) & (df["timestamp"] < future_end)
        )
        future_df = df.loc[future_mask]
        accepted_devs = set(
            future_df.loc[future_df["label"] == 1, "email"].dropna().unique().tolist()
        )
        logger.info(
            "候補 dev=%d, 将来窓で承諾あり=%d",
            len(candidate_devs), len(accepted_devs),
        )

        dev_rows: List[Dict] = []
        for i, dev in enumerate(candidate_devs):
            try:
                p = predictor.predict_developer(
                    dev, pred_time.to_pydatetime(),
                    path_extractor=path_extractor,
                )
            except ValueError as e:
                logger.error("predict_developer 失敗: %s", e)
                p = 0.5
            label = 1 if dev in accepted_devs else 0
            dev_rows.append({"email": dev, "label": label, "prob": float(p)})
            if (i + 1) % 200 == 0:
                logger.info("  dev 推論 進捗: %d/%d", i + 1, len(candidate_devs))

        if dev_rows:
            y_true_dev = np.array([r["label"] for r in dev_rows])
            y_prob_dev = np.array([r["prob"] for r in dev_rows])
            metrics_dev = _binary_metrics(y_true_dev, y_prob_dev, "IRL_dev")
            summary["IRL_dev"] = metrics_dev

            dev_df = pd.DataFrame(dev_rows)
            dev_path = output_dir / "dev_predictions.csv"
            dev_df.to_csv(dev_path, index=False)
            logger.info("dev 予測保存: %s", dev_path)

    # ════════════════════════════════════════════════════════════════
    # 保存
    # ════════════════════════════════════════════════════════════════

    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("サマリ保存: %s", summary_path)

    logger.info("=" * 70)
    logger.info("評価結果サマリ")
    logger.info("=" * 70)
    for grp, m in summary.items():
        if "clf_auc_roc" in m:
            logger.info(
                "  %s: clf_auc_roc=%.4f clf_auc_pr=%.4f n_pairs=%d (pos=%d, neg=%d)",
                grp, m["clf_auc_roc"], m["clf_auc_pr"],
                int(m["n_pairs"]), int(m["n_pos"]), int(m["n_neg"]),
            )


if __name__ == "__main__":
    main()
