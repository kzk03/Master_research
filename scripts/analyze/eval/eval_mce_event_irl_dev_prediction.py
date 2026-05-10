#!/usr/bin/env python3
"""per-dev MCE-IRL モデルの dev レベル評価スクリプト。

per-dev で学習されたモデル (model_metadata.json: per_dev=True) は本質的に
「dev が継続するか」を予測している。target_dir 条件付きの (dev, dir) ペア
評価は学習目標と乖離するため、本スクリプトでは:

  - 候補 dev = 予測時点 T で過去 window_days 日に活動した reviewer 全員
  - ラベル  = dev が T~T+Δ ヶ月内に label=1 のレビューを承諾したか
  - 予測   = predictor.predict_developer(dev, T) (β 戦略なし、素の dev 軌跡)

として AUC を計算する。

使い方:
    uv run python scripts/analyze/eval/eval_mce_event_irl_dev_prediction.py \\
        --data data/combined_raw.csv \\
        --raw-json data/raw_json/openstack__*.json \\
        --model outputs/mce_pilot_event_dev/event_cold/mce_event_irl_model.pt \\
        --prediction-time 2023-01-01 \\
        --delta-months 3 \\
        --output-dir outputs/mce_pilot_event_dev/eval_dev_level
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    p = argparse.ArgumentParser(description="per-dev MCE-IRL の dev レベル評価")
    p.add_argument("--data", type=str, required=True, help="combined_raw.csv")
    p.add_argument(
        "--raw-json", type=str, nargs="+", required=True,
        help="dirs マッピング用の Gerrit raw JSON",
    )
    p.add_argument("--model", type=str, required=True, help="mce_event_irl_model.pt")
    p.add_argument(
        "--prediction-time", type=str, required=True,
        help="予測時点 T (例: 2023-01-01)",
    )
    p.add_argument(
        "--delta-months", type=int, default=3,
        help="将来窓 Δ (T~T+Δ ヶ月で承諾有無を判定)",
    )
    p.add_argument(
        "--future-start-months", type=int, default=0,
        help="head_index 計算用 (head_index = future_start_months // 3)",
    )
    p.add_argument(
        "--window-days", type=int, default=180,
        help="候補 dev 抽出用の過去活動窓 (日)",
    )
    p.add_argument(
        "--history-months", type=int, default=24,
        help="モデル推論用の dev 履歴開始時点 (T - history_months)",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="モデル実行デバイス",
    )
    p.add_argument("--output-dir", type=str, required=True, help="評価結果の出力先")
    return p.parse_args()


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
    from review_predictor.IRL.model.mce_event_irl_batch_predictor import (
        MCEEventBatchContinuationPredictor,
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
    logger.info(f"データ読み込み: {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    if "request_time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"request_time": "timestamp"})
    if "reviewer_email" in df.columns and "email" not in df.columns:
        df = df.rename(columns={"reviewer_email": "email"})
    if "timestamp" not in df.columns:
        raise ValueError("data に timestamp 列がありません")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── ディレクトリマッピング ──
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    df = attach_dirs_to_df(df, cdm, column="dirs")

    path_extractor = PathFeatureExtractor(df, window_days=180)

    # ── predictor 初期化 ──
    predictor = MCEEventBatchContinuationPredictor(
        model_path=args.model,
        df=df,
        history_start=history_start,
        device=args.device,
        reviewer_col="email",
        date_col="timestamp",
        label_col="label",
        dirs_column="dirs",
    )

    # ── 候補 dev 抽出 (T 時点で過去 window_days 内に活動) ──
    candidate_mask = (
        (df["timestamp"] >= window_start) & (df["timestamp"] < pred_time)
    )
    candidate_devs = sorted(df.loc[candidate_mask, "email"].dropna().unique().tolist())
    logger.info(
        f"候補 dev 数: {len(candidate_devs)} "
        f"(過去 {args.window_days} 日に活動)"
    )

    # ── ラベル: T~T+Δ で label=1 を持つ dev ──
    future_mask = (
        (df["timestamp"] >= future_start) & (df["timestamp"] < future_end)
    )
    future_df = df.loc[future_mask]
    accepted_devs = set(
        future_df.loc[future_df["label"] == 1, "email"].dropna().unique().tolist()
    )
    logger.info(
        f"将来窓 [{future_start.date()}, {future_end.date()}) で "
        f"label=1 を持つ dev: {len(accepted_devs)}"
    )

    # ── 予測 ──
    head_index = args.future_start_months // 3
    y_true: List[int] = []
    y_prob: List[float] = []
    pair_rows: List[Dict[str, float]] = []

    logger.info(f"dev レベル予測開始 ({len(candidate_devs)} dev)...")
    for i, dev in enumerate(candidate_devs):
        prob = predictor.predict_developer(
            dev, pred_time.to_pydatetime(),
            path_extractor=path_extractor, head_index=head_index,
        )
        label = 1 if dev in accepted_devs else 0
        y_true.append(label)
        y_prob.append(float(prob))
        pair_rows.append({"email": dev, "label": label, "prob": float(prob)})
        if (i + 1) % 200 == 0:
            logger.info(f"  進捗: {i + 1}/{len(candidate_devs)}")

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    n_devs = len(y_true_arr)
    n_pos = int(y_true_arr.sum())
    n_neg = n_devs - n_pos
    logger.info(f"評価対象: {n_devs} dev (pos={n_pos}, neg={n_neg})")

    # ── メトリクス計算 ──
    metrics: Dict[str, float] = {
        "n_devs": float(n_devs),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "pos_rate": float(n_pos / n_devs) if n_devs > 0 else float("nan"),
    }
    if n_pos > 0 and n_neg > 0:
        from sklearn.metrics import (
            auc as sk_auc,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        metrics["auc_roc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        prec_c, rec_c, thr = precision_recall_curve(y_true_arr, y_prob_arr)
        metrics["auc_pr"] = float(sk_auc(rec_c, prec_c))
        f1s = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-10)
        bi = int(np.argmax(f1s))
        bt = float(thr[bi]) if bi < len(thr) else 0.5
        y_pred = (y_prob_arr >= bt).astype(int)
        metrics["threshold"] = bt
        metrics["f1"] = float(f1_score(y_true_arr, y_pred, zero_division=0))
        metrics["precision"] = float(
            precision_score(y_true_arr, y_pred, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true_arr, y_pred, zero_division=0)
        )
        # 確率分布の参考統計
        metrics["prob_mean"] = float(y_prob_arr.mean())
        metrics["prob_pos_mean"] = float(y_prob_arr[y_true_arr == 1].mean())
        metrics["prob_neg_mean"] = float(y_prob_arr[y_true_arr == 0].mean())
    else:
        logger.warning("正例 or 負例が 0 件のため AUC をスキップ")

    # ── 保存 ──
    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"サマリ保存: {summary_path}")

    pair_df = pd.DataFrame(pair_rows)
    pair_path = output_dir / "dev_predictions.csv"
    pair_df.to_csv(pair_path, index=False)
    logger.info(f"dev 予測保存: {pair_path}")

    # ── 結果ログ ──
    logger.info("=" * 70)
    logger.info("dev レベル評価結果")
    logger.info("=" * 70)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
