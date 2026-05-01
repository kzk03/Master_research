# -*- coding: utf-8 -*-
"""
Quantitative comparison of IRL (gradient) vs RF (Gini) feature importance.

Reads per-window importance JSONs and produces:
    - irl_vs_rf_importance.csv  (full table: per-window + averaged shares/ranks)
    - irl_vs_rf_scatter.pdf     (scatter: IRL share vs RF share, labeled, by category)
    - irl_vs_rf_divergence.pdf  (horizontal bar: features sorted by share difference)
    - irl_vs_rf_summary.json    (Spearman correlation per window + average)

Usage:
    uv run python scripts/analyze/compare_irl_rf_importance.py \
        --input-dir outputs/variant_comparison_server/lstm_baseline \
        --output-dir outputs/variant_comparison_server/figures/feature_importance
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

FEATURE_META = [
    ("experience_days",            "経験日数",         "state"),
    ("total_changes",              "総レビュー依頼数", "state"),
    ("total_reviews",              "総レビュー数",     "state"),
    ("recent_activity_frequency",  "最近の活動頻度",   "state"),
    ("avg_activity_gap",           "平均活動間隔",     "state"),
    ("activity_trend",             "月次活動変化率",   "state"),
    ("collaboration_score",        "協力スコア",       "state"),
    ("code_quality_score",         "総承諾率",         "state"),
    ("recent_acceptance_rate",     "最近の承諾率",     "state"),
    ("review_load",                "レビュー負荷",     "state"),
    ("days_since_last_activity",   "最終活動経過日数", "state"),
    ("acceptance_trend",           "承諾率トレンド",   "state"),
    ("reciprocity_score",          "相互レビュー率",   "state"),
    ("load_trend",                 "負荷トレンド",     "state"),
    ("core_reviewer_ratio",        "コアレビュアー率", "state"),
    ("recent_rejection_streak",    "直近拒否連続数",   "state"),
    ("acceptance_rate_last10",     "直近10件承諾率",   "state"),
    ("active_months_ratio",        "活動月割合",       "state"),
    ("response_time_trend",        "応答速度トレンド", "state"),
    ("complex_pr_bias",            "複雑PRバイアス",   "state"),
    ("path_review_count",          "パスレビュー件数", "path"),
    ("path_recency",               "パス新しさ",       "path"),
    ("path_acceptance_rate",       "パス承諾率",       "path"),
    ("avg_action_intensity",       "行動強度",         "action"),
    ("avg_change_lines",           "平均変更行数",     "action"),
    ("avg_response_time",          "応答速度",         "action"),
    ("avg_review_size",            "レビュー規模",     "action"),
    ("repeat_collaboration_rate",  "リピート協力率",   "action"),
]
FEATURE_NAME_JA: Dict[str, str] = {en: ja for en, ja, _ in FEATURE_META}
FEATURE_CATEGORY: Dict[str, str] = {en: cat for en, _, cat in FEATURE_META}
CATEGORY_COLOR = {"state": "#1f77b4", "path": "#2ca02c", "action": "#d62728"}
CATEGORY_LABEL = {"state": "状態 (20)", "path": "パス (3)", "action": "行動 (5)"}


def _load_window(base_dir: pathlib.Path, period: str) -> Optional[pd.DataFrame]:
    eval_dir = base_dir / f"train_{period}" / f"eval_{period}"
    irl_path = eval_dir / "irl_feature_importance.json"
    irl_signed_path = eval_dir / "irl_feature_importance_signed.json"
    rf_path = eval_dir / "rf_metrics.json"

    if not irl_path.exists() or not rf_path.exists():
        return None

    irl = json.loads(irl_path.read_text())
    rf_full = json.loads(rf_path.read_text())
    rf = rf_full.get("feature_importance", {})
    irl_signed = (
        json.loads(irl_signed_path.read_text()) if irl_signed_path.exists() else {}
    )

    rows = []
    for en, ja, cat in FEATURE_META:
        rows.append({
            "feature": en,
            "feature_ja": ja,
            "category": cat,
            "period": period,
            "irl_imp": float(irl.get(en, 0.0)),
            "rf_imp": float(rf.get(en, 0.0)),
            "irl_signed": float(irl_signed.get(en, 0.0)),
        })
    df = pd.DataFrame(rows)
    irl_sum = df["irl_imp"].sum()
    rf_sum = df["rf_imp"].sum()
    df["irl_share"] = df["irl_imp"] / irl_sum if irl_sum > 0 else 0.0
    df["rf_share"] = df["rf_imp"] / rf_sum if rf_sum > 0 else 0.0
    df["irl_rank"] = df["irl_imp"].rank(ascending=False, method="min").astype(int)
    df["rf_rank"] = df["rf_imp"].rank(ascending=False, method="min").astype(int)
    return df


def _load_all(base_dir: pathlib.Path) -> pd.DataFrame:
    frames = []
    for p in PERIODS:
        df = _load_window(base_dir, p)
        if df is not None:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No importance JSONs under {base_dir}/train_*/eval_*/")
    return pd.concat(frames, ignore_index=True)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["feature", "feature_ja", "category"], as_index=False)
        .agg(
            irl_share_avg=("irl_share", "mean"),
            rf_share_avg=("rf_share", "mean"),
            irl_signed_avg=("irl_signed", "mean"),
            irl_rank_avg=("irl_rank", "mean"),
            rf_rank_avg=("rf_rank", "mean"),
        )
    )
    agg["share_diff"] = agg["irl_share_avg"] - agg["rf_share_avg"]
    agg["rank_diff"] = agg["rf_rank_avg"] - agg["irl_rank_avg"]
    return agg.sort_values("irl_share_avg", ascending=False).reset_index(drop=True)


def _spearman_per_window(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in PERIODS:
        sub = df[df["period"] == p]
        if sub.empty:
            continue
        rho, _ = spearmanr(sub["irl_imp"], sub["rf_imp"])
        out[p] = float(rho)
    if out:
        out["mean"] = float(np.mean(list(out.values())))
    return out


def _wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return wide format: per-feature row with share/rank columns per window + averages."""
    pieces = []
    for p in PERIODS:
        sub = df[df["period"] == p][
            ["feature", "irl_share", "rf_share", "irl_rank", "rf_rank", "irl_signed"]
        ].copy()
        sub = sub.rename(
            columns={
                "irl_share": f"irl_share_{p}",
                "rf_share": f"rf_share_{p}",
                "irl_rank": f"irl_rank_{p}",
                "rf_rank": f"rf_rank_{p}",
                "irl_signed": f"irl_signed_{p}",
            }
        )
        pieces.append(sub.set_index("feature"))
    wide = pd.concat(pieces, axis=1).reset_index()
    return wide


def _plot_scatter(agg: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for cat in ["state", "path", "action"]:
        sub = agg[agg["category"] == cat]
        ax.scatter(
            sub["irl_share_avg"], sub["rf_share_avg"],
            s=80, c=CATEGORY_COLOR[cat], label=CATEGORY_LABEL[cat],
            edgecolors="black", linewidths=0.5, alpha=0.85,
        )
    max_v = max(agg["irl_share_avg"].max(), agg["rf_share_avg"].max()) * 1.1
    ax.plot([0, max_v], [0, max_v], "k--", linewidth=0.8, alpha=0.5, label="y = x")

    label_thresh = 0.03
    for _, row in agg.iterrows():
        if row["irl_share_avg"] >= label_thresh or row["rf_share_avg"] >= label_thresh:
            ax.annotate(
                row["feature_ja"],
                (row["irl_share_avg"], row["rf_share_avg"]),
                fontsize=8.5, xytext=(5, 4), textcoords="offset points",
            )

    ax.set_xlim(0, max_v)
    ax.set_ylim(0, max_v)
    ax.set_xlabel("IRL importance share (window avg)", fontsize=11)
    ax.set_ylabel("RF importance share (window avg)", fontsize=11)
    ax.set_title("特徴量重要度: IRL vs RF (lstm_baseline, 4窓平均)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_divergence(agg: pd.DataFrame, out_path: pathlib.Path, top_n: int = 15) -> None:
    """Horizontal bar chart of share_diff (IRL - RF), sorted by absolute value."""
    df = agg.assign(abs_diff=agg["share_diff"].abs())
    df = df.nlargest(top_n, "abs_diff").sort_values("share_diff")

    colors = ["#d62728" if d < 0 else "#1f77b4" for d in df["share_diff"]]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(df))))
    ax.barh(df["feature_ja"], df["share_diff"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("IRL share − RF share (4窓平均)", fontsize=11)
    ax.set_title(
        f"重要度の発散 top {top_n}: 青=IRL重視, 赤=RF重視 (lstm_baseline)",
        fontsize=12,
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    for i, (val, cat) in enumerate(zip(df["share_diff"], df["category"])):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            i,
            f"[{cat}]",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7,
            color="#555555",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare IRL gradient importance vs RF Gini importance quantitatively"
    )
    parser.add_argument(
        "--input-dir", type=pathlib.Path, required=True,
        help="Variant base dir, e.g. outputs/variant_comparison_server/lstm_baseline",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, required=True,
        help="Output dir for CSV/PDF/JSON",
    )
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_df = _load_all(args.input_dir)
    agg = _aggregate(long_df)
    wide = _wide_table(long_df).merge(
        agg[["feature", "feature_ja", "category", "irl_share_avg", "rf_share_avg",
             "irl_signed_avg", "irl_rank_avg", "rf_rank_avg", "share_diff", "rank_diff"]],
        on="feature",
    )
    cols_first = ["feature", "feature_ja", "category",
                  "irl_share_avg", "rf_share_avg", "share_diff",
                  "irl_rank_avg", "rf_rank_avg", "rank_diff",
                  "irl_signed_avg"]
    wide = wide[cols_first + [c for c in wide.columns if c not in cols_first]]
    wide = wide.sort_values("irl_share_avg", ascending=False)

    csv_path = args.output_dir / "irl_vs_rf_importance.csv"
    wide.to_csv(csv_path, index=False)

    spearman = _spearman_per_window(long_df)
    spearman_path = args.output_dir / "irl_vs_rf_summary.json"
    spearman_path.write_text(json.dumps(
        {
            "spearman_irl_rf_per_window": spearman,
            "n_features": int(len(agg)),
            "input_dir": str(args.input_dir),
        },
        indent=2, ensure_ascii=False,
    ))

    _plot_scatter(agg, args.output_dir / "irl_vs_rf_scatter")
    _plot_divergence(agg, args.output_dir / "irl_vs_rf_divergence", top_n=args.top_n)

    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] JSON: {spearman_path}")
    print(f"[OK] Scatter: {args.output_dir / 'irl_vs_rf_scatter.pdf'}")
    print(f"[OK] Divergence: {args.output_dir / 'irl_vs_rf_divergence.pdf'}")
    print(f"  Spearman(IRL rank, RF rank) per window: {spearman}")


if __name__ == "__main__":
    main()
