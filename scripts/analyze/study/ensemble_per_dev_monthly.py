#!/usr/bin/env python3
"""per-dev event MCE-IRL × 月次 MCE-IRL のアンサンブル評価。

両モデルの (dev, dir) ペア予測を融合し、AUC ROC / PR を最大化する重みを
探索する。fine-tune を使わず、推論結果の単純な組み合わせのみで
月次 cold (0.792) を超えられるかを検証する。

入力:
  - outputs/mce_pilot/eval_monthly_cold/pair_predictions.csv
  - outputs/mce_pilot_event_dev/eval_event_cold/pair_predictions.csv

出力:
  - outputs/mce_pilot_event_dev/eval_ensemble/ensemble_summary.json
  - outputs/mce_pilot_event_dev/eval_ensemble/ensemble_predictions.csv
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="per-dev × 月次 MCE-IRL アンサンブル")
    p.add_argument(
        "--monthly-csv", type=str,
        default="outputs/mce_pilot/eval_monthly_cold/pair_predictions.csv",
    )
    p.add_argument(
        "--event-dev-csv", type=str,
        default="outputs/mce_pilot_event_dev/eval_event_cold/pair_predictions.csv",
    )
    p.add_argument(
        "--output-dir", type=str,
        default="outputs/mce_pilot_event_dev/eval_ensemble",
    )
    p.add_argument(
        "--prob-col", type=str, default="irl_dir_prob",
        help="ベースとなる確率列名 (default: irl_dir_prob, 校正前)",
    )
    return p.parse_args()


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def auc_metrics(y_true, y_prob, name=""):
    from sklearn.metrics import (
        auc as sk_auc,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"auc_roc": float("nan"), "auc_pr": float("nan")}
    auc_roc = float(roc_auc_score(y_true, y_prob))
    prec_c, rec_c, thr = precision_recall_curve(y_true, y_prob)
    auc_pr = float(sk_auc(rec_c, prec_c))
    f1s = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-10)
    bi = int(np.argmax(f1s))
    bt = float(thr[bi]) if bi < len(thr) else 0.5
    yp = (y_prob >= bt).astype(int)
    out = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": float(f1_score(y_true, yp, zero_division=0)),
        "precision": float(precision_score(y_true, yp, zero_division=0)),
        "recall": float(recall_score(y_true, yp, zero_division=0)),
        "threshold": bt,
        "n_pairs": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    logger.info(
        f"[{name}] AUC ROC={auc_roc:.4f} PR={auc_pr:.4f} "
        f"F1={out['f1']:.4f} (P={out['precision']:.4f} R={out['recall']:.4f})"
    )
    return out


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"月次 cold 読み込み: {args.monthly_csv}")
    df_m = pd.read_csv(args.monthly_csv)
    logger.info(f"per-dev event 読み込み: {args.event_dev_csv}")
    df_e = pd.read_csv(args.event_dev_csv)

    key_cols = ["developer", "directory"]
    df_m = df_m[key_cols + ["label", args.prob_col]].rename(
        columns={args.prob_col: "p_monthly"}
    )
    df_e = df_e[key_cols + ["label", args.prob_col]].rename(
        columns={args.prob_col: "p_event_dev"}
    )

    df = df_m.merge(df_e, on=key_cols, suffixes=("_m", "_e"))
    if "label_m" in df.columns:
        # ラベルは両方一致するはずだが念のため整合確認
        mismatch = (df["label_m"] != df["label_e"]).sum()
        if mismatch:
            logger.warning(f"ラベル不一致 {mismatch} 件")
        df["label"] = df["label_m"]
        df = df.drop(columns=["label_m", "label_e"])

    n_pairs = len(df)
    n_pos = int(df["label"].sum())
    logger.info(f"join 後 ペア数: {n_pairs} (pos={n_pos}, neg={n_pairs - n_pos})")

    y = df["label"].values.astype(int)
    p_m = df["p_monthly"].values
    p_e = df["p_event_dev"].values

    results = {}

    # ── 個別モデル ──
    results["monthly_cold"] = auc_metrics(y, p_m, "monthly_cold")
    results["per_dev_event"] = auc_metrics(y, p_e, "per_dev_event")

    # ── 単純平均 ──
    results["mean"] = auc_metrics(y, (p_m + p_e) / 2.0, "mean")

    # ── logit 平均 ──
    z_avg = (logit(p_m) + logit(p_e)) / 2.0
    results["logit_mean"] = auc_metrics(y, sigmoid(z_avg), "logit_mean")

    # ── 重み付け sweep (確率空間) ──
    best_alpha = 0.5
    best_auc = -1.0
    sweep_results = []
    for alpha in np.linspace(0.0, 1.0, 21):
        p_mix = alpha * p_m + (1 - alpha) * p_e
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p_mix))
        sweep_results.append({"alpha": float(alpha), "auc_roc": auc})
        if auc > best_auc:
            best_auc = auc
            best_alpha = float(alpha)
    logger.info(
        f"確率重み付け最良: alpha={best_alpha:.2f} (p = {best_alpha:.2f}*p_m + {1-best_alpha:.2f}*p_e), "
        f"AUC={best_auc:.4f}"
    )
    p_best_prob = best_alpha * p_m + (1 - best_alpha) * p_e
    results["weighted_prob_best"] = auc_metrics(y, p_best_prob, f"weighted_prob α={best_alpha:.2f}")
    results["weighted_prob_best"]["alpha"] = best_alpha
    results["sweep_prob"] = sweep_results

    # ── 重み付け sweep (logit 空間) ──
    best_alpha_l = 0.5
    best_auc_l = -1.0
    sweep_logit = []
    z_m = logit(p_m)
    z_e = logit(p_e)
    for alpha in np.linspace(0.0, 1.0, 21):
        z_mix = alpha * z_m + (1 - alpha) * z_e
        p_mix = sigmoid(z_mix)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p_mix))
        sweep_logit.append({"alpha": float(alpha), "auc_roc": auc})
        if auc > best_auc_l:
            best_auc_l = auc
            best_alpha_l = float(alpha)
    logger.info(
        f"logit 重み付け最良: alpha={best_alpha_l:.2f}, AUC={best_auc_l:.4f}"
    )
    p_best_logit = sigmoid(best_alpha_l * z_m + (1 - best_alpha_l) * z_e)
    results["weighted_logit_best"] = auc_metrics(y, p_best_logit, f"weighted_logit α={best_alpha_l:.2f}")
    results["weighted_logit_best"]["alpha"] = best_alpha_l
    results["sweep_logit"] = sweep_logit

    # ── ロジスティック回帰での fusion (2 features → 1 prob) ──
    # 注: held-out なしで fit したらリーク。ここでは 5-fold CV で OOF AUC を測定
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    X = np.stack([z_m, z_e], axis=1)
    oof = np.zeros(len(y))
    kf = KFold(n_splits=5, shuffle=True, random_state=777)
    for tr, te in kf.split(X):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    results["lr_fusion_oof"] = auc_metrics(y, oof, "lr_fusion_oof (5-fold)")
    # 全データで fit した係数も参考保存
    clf_full = LogisticRegression(max_iter=1000).fit(X, y)
    results["lr_fusion_oof"]["coef_logit_monthly"] = float(clf_full.coef_[0, 0])
    results["lr_fusion_oof"]["coef_logit_event"] = float(clf_full.coef_[0, 1])
    results["lr_fusion_oof"]["intercept"] = float(clf_full.intercept_[0])

    # ── 保存 ──
    df_out = df.copy()
    df_out["p_mean"] = (p_m + p_e) / 2.0
    df_out["p_logit_mean"] = sigmoid(z_avg)
    df_out["p_weighted_prob"] = p_best_prob
    df_out["p_weighted_logit"] = p_best_logit
    df_out["p_lr_oof"] = oof
    pred_path = output_dir / "ensemble_predictions.csv"
    df_out.to_csv(pred_path, index=False)
    logger.info(f"予測 CSV 保存: {pred_path}")

    summary_path = output_dir / "ensemble_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"サマリ保存: {summary_path}")

    # ── 比較表 ──
    print("\n" + "=" * 80)
    print(f"{'method':<35} {'AUC ROC':>10} {'AUC PR':>10}")
    print("-" * 80)
    for name in [
        "monthly_cold", "per_dev_event", "mean", "logit_mean",
        "weighted_prob_best", "weighted_logit_best", "lr_fusion_oof",
    ]:
        r = results[name]
        suffix = ""
        if "alpha" in r:
            suffix = f"  (α={r['alpha']:.2f})"
        print(
            f"{name:<35} {r.get('auc_roc', float('nan')):>10.4f} "
            f"{r.get('auc_pr', float('nan')):>10.4f}{suffix}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
