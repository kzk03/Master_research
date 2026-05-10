#!/usr/bin/env python3
"""エンゲージメント版 MCE-IRL の評価

`MCEEventMulticlassBatchPredictor` を流用し、評価ラベルを
"未来 N ヶ月以内に何らかのイベント発生" (engagement) に切り替える。

評価軸:
  (a) (dev, dir) 評価: π(a = class_id(target_dir) | s_last) を確率とし、
      label = 1 if dev が future window 内に target_dir 周辺で何らかの
      activity を発生させたか
  (b) dev レベル評価: π(a ≠ 0 | s_last) を確率とし、
      label = 1 if dev が future window 内に何らかの activity を発生させたか

参考: 既存 eval_mce_event_irl_multiclass_prediction.py は
"label==1 (accept) があったか" で判定。本スクリプトは
"何らかの event があったか (label を問わない)" で判定する。

使い方:
    uv run python scripts/analyze/eval/eval_mce_engagement_prediction.py \\
        --events data/combined_events.csv \\
        --data data/combined_raw.csv \\
        --raw-json data/raw_json/openstack__*.json \\
        --model outputs/mce_pilot_engagement/event_cold/mce_event_irl_model.pt \\
        --dir-class-mapping outputs/dir_class_mapping_K15.json \\
        --prediction-time 2022-01-01 \\
        --delta-months 3 \\
        --window-days 180 \\
        --output-dir outputs/mce_pilot_engagement/eval_event_cold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--events", type=str, required=True,
        help="combined_events.csv (engagement label 計算用)",
    )
    p.add_argument("--data", type=str, required=True, help="combined_raw.csv")
    p.add_argument("--raw-json", type=str, nargs="+", required=True)
    p.add_argument("--model", type=str, required=True, help="mce_event_irl_model.pt")
    p.add_argument(
        "--dir-class-mapping", type=str, default=None,
        help="省略時は model_metadata.json から読み取り",
    )
    p.add_argument("--prediction-time", type=str, required=True)
    p.add_argument("--delta-months", type=int, default=3)
    p.add_argument("--future-start-months", type=int, default=0)
    p.add_argument("--window-days", type=int, default=180)
    p.add_argument("--history-months", type=int, default=24)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--skip-pair-eval", action="store_true")
    p.add_argument("--skip-dev-eval", action="store_true")
    p.add_argument(
        "--use-beta", action="store_true",
        help="β 戦略を使う: 最終 step の path_features を target_dir で上書きしてから推論",
    )
    return p.parse_args()


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Dict[str, float]:
    from sklearn.metrics import (
        auc as sk_auc, f1_score, precision_recall_curve,
        precision_score, recall_score, roc_auc_score,
    )
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    out: Dict[str, float] = {
        "n_pairs": float(n), "n_pos": float(n_pos), "n_neg": float(n_neg),
    }
    if n_pos == 0 or n_neg == 0:
        logger.warning("[%s] 正例 or 負例 0 → AUC スキップ", name)
        out.update({k: float("nan") for k in [
            "clf_auc_roc", "clf_auc_pr", "clf_f1",
            "clf_precision", "clf_recall", "clf_threshold",
        ]})
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
        "[%s] auc_roc=%.4f auc_pr=%.4f f1=%.3f thr=%.3f n=%d (pos=%d, neg=%d, pos_rate=%.3f)",
        name, out["clf_auc_roc"], out["clf_auc_pr"],
        out["clf_f1"], out["clf_threshold"], n, n_pos, n_neg, n_pos / n,
    )
    return out


def _load_events_with_dirs(events_csv: str) -> pd.DataFrame:
    df = pd.read_csv(events_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    def _parse(s):
        if not isinstance(s, str) or s == "[]":
            return []
        try:
            return [d for d in json.loads(s) if d and d != "."]
        except Exception:
            return []

    df["dirs_list"] = df["dirs"].map(_parse)
    return df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from review_predictor.IRL.features.path_features import (
        PathFeatureExtractor, attach_dirs_to_df,
        load_change_dir_map, load_change_dir_map_multi,
    )
    from review_predictor.IRL.features.directory_contributors import (
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
    history_start = (pred_time - pd.DateOffset(months=args.history_months)).to_pydatetime()

    # ── データ読み込み (combined_raw / events) ──
    logger.info("loading: %s", args.data)
    df = pd.read_csv(args.data, low_memory=False)
    if "request_time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"request_time": "timestamp"})
    if "reviewer_email" in df.columns and "email" not in df.columns:
        df = df.rename(columns={"reviewer_email": "email"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    df = attach_dirs_to_df(df, cdm, column="dirs")
    path_extractor = PathFeatureExtractor(df, window_days=180)

    logger.info("loading events: %s", args.events)
    events_df = _load_events_with_dirs(args.events)

    # engagement label set 構築 (future window 内に何らかの event 発生)
    in_future = events_df[
        (events_df["timestamp"] >= future_start) & (events_df["timestamp"] < future_end)
    ]
    devs_engaged_future: Set[str] = set(in_future["email"].dropna().unique().tolist())
    # (dev, dir) engagement: dev が future window で any event を dir に対して発生させたか
    pair_engaged_future: Set[Tuple[str, str]] = set()
    for _, row in in_future.iterrows():
        for d in row["dirs_list"]:
            pair_engaged_future.add((row["email"], d))
    logger.info(
        "future window [%s, %s): events=%d, engaged_devs=%d, engaged_pairs=%d",
        future_start.date(), future_end.date(),
        len(in_future), len(devs_engaged_future), len(pair_engaged_future),
    )

    # ── predictor ──
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
    # Part 1: (dev, dir) engagement 評価
    # ════════════════════════════════════════════════════════════════
    if not args.skip_pair_eval:
        logger.info("=" * 70)
        logger.info("(dev, dir) engagement 評価")
        logger.info("=" * 70)
        # 候補 (dev, dir) ペアは combined_raw.csv ベース (歴史 window で活動した dev × その dir)
        dir_developers: Dict[str, Set[str]] = get_directory_developers(
            df, window_start, pred_time
        )
        n_pairs_total = sum(len(v) for v in dir_developers.values())
        logger.info(
            "対象 dirs=%d, 候補 (dev, dir) ペア=%d", len(dir_developers), n_pairs_total,
        )
        flat_pairs: List[Tuple[str, str]] = [
            (d, dev) for d, devs in dir_developers.items() for dev in devs
        ]

        predictor._load_model()
        per_dev_mode = predictor.per_dev

        pair_results: List[Dict] = []
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
                    "label": int((dev, d) in pair_engaged_future),
                })
                if (i + 1) % 500 == 0:
                    logger.info("  β 推論 進捗: %d/%d", i + 1, len(flat_pairs))
        elif per_dev_mode:
            unique_devs = sorted({dev for _, dev in flat_pairs})
            logger.info("per-dev mode: %d 個の reviewer で 1 回ずつ推論", len(unique_devs))
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
                p = probs[cid] if probs and 0 <= cid < len(probs) else 0.5
                pair_results.append({
                    "directory": d,
                    "email": dev,
                    "target_class_id": int(cid),
                    "prob": float(p),
                    "label": int((dev, d) in pair_engaged_future),
                })
        else:
            logger.info("(dev, dir) ペアモード: 個別推論")
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
                    "label": int((dev, d) in pair_engaged_future),
                })
                if (i + 1) % 1000 == 0:
                    logger.info("  ペア推論 進捗: %d/%d", i + 1, len(flat_pairs))

        if pair_results:
            y_true = np.array([r["label"] for r in pair_results])
            y_prob = np.array([r["prob"] for r in pair_results])
            metrics_pair = _binary_metrics(y_true, y_prob, "IRL_Dir_engagement")
            summary["IRL_Dir"] = metrics_pair
            pd.DataFrame(pair_results).to_csv(
                output_dir / "pair_predictions.csv", index=False
            )
            logger.info("ペア予測保存: %s", output_dir / "pair_predictions.csv")

    # ════════════════════════════════════════════════════════════════
    # Part 2: dev レベル engagement 評価
    # ════════════════════════════════════════════════════════════════
    if not args.skip_dev_eval:
        logger.info("=" * 70)
        logger.info("dev レベル engagement 評価")
        logger.info("=" * 70)

        # 候補 dev は combined_raw でも events でも OK だが、訓練と整合させて events ベースで定義
        cand_mask = (
            (events_df["timestamp"] >= window_start)
            & (events_df["timestamp"] < pred_time)
        )
        candidate_devs = sorted(events_df.loc[cand_mask, "email"].dropna().unique().tolist())
        logger.info(
            "候補 dev (events で活動)=%d, future engaged=%d",
            len(candidate_devs), len(devs_engaged_future),
        )

        dev_rows: List[Dict] = []
        for i, dev in enumerate(candidate_devs):
            try:
                p = predictor.predict_developer(
                    dev, pred_time.to_pydatetime(),
                    path_extractor=path_extractor,
                )
            except (ValueError, KeyError) as e:
                logger.error("predict_developer 失敗 dev=%s: %s", dev, e)
                p = 0.5
            label = 1 if dev in devs_engaged_future else 0
            dev_rows.append({"email": dev, "label": label, "prob": float(p)})
            if (i + 1) % 200 == 0:
                logger.info("  dev 推論 進捗: %d/%d", i + 1, len(candidate_devs))

        if dev_rows:
            y_true_dev = np.array([r["label"] for r in dev_rows])
            y_prob_dev = np.array([r["prob"] for r in dev_rows])
            metrics_dev = _binary_metrics(y_true_dev, y_prob_dev, "IRL_dev_engagement")
            summary["IRL_dev"] = metrics_dev
            pd.DataFrame(dev_rows).to_csv(output_dir / "dev_predictions.csv", index=False)
            logger.info("dev 予測保存: %s", output_dir / "dev_predictions.csv")

    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("サマリ保存: %s", summary_path)

    logger.info("=" * 70)
    logger.info("評価結果サマリ (engagement)")
    logger.info("=" * 70)
    for grp, m in summary.items():
        if "clf_auc_roc" in m:
            logger.info(
                "  %s: clf_auc_roc=%.4f clf_auc_pr=%.4f n=%d (pos=%d, neg=%d)",
                grp, m["clf_auc_roc"], m["clf_auc_pr"],
                int(m["n_pairs"]), int(m["n_pos"]), int(m["n_neg"]),
            )


if __name__ == "__main__":
    main()
