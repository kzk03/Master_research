# -*- coding: utf-8 -*-
"""
Co-change graph と (reviewer, directory) ペア予測を組み合わせた事後分析。

(2) Hub score 連続値による層別:
    pair_predictions.csv の各ペアに hub_score を付与し、4分位で
    継続率 / IRL AUC / RF AUC / 両モデル合致率を比較。窓 × tier でも集計。

(3) Co-change 近傍カバレッジによる層別:
    各 (reviewer, directory) ペアについて
        coverage = |{dir の co-change 近傍} ∩ {reviewer が担当する他の dir}| / |dir の co-change 近傍|
    を計算し、coverage bin で継続率と AUC を比較。

Inputs:
    experiments/dependency_analysis/results/hub_scores.csv
    experiments/dependency_analysis/results/top_cochange_pairs.csv
    outputs/variant_comparison_server/lstm_baseline/train_*/eval_*/pair_predictions.csv

Outputs (--output-dir 配下):
    hub_score_stratified.csv      hub 4分位の継続率・AUC
    hub_score_stratified.pdf      横棒グラフ
    cochange_coverage_stratified.csv
    cochange_coverage_stratified.pdf

Usage:
    uv run python scripts/analyze/analyze_cochange_continuation.py \
        --pair-base outputs/variant_comparison_server/lstm_baseline \
        --hub-csv experiments/dependency_analysis/results/hub_scores.csv \
        --cochange-csv experiments/dependency_analysis/results/top_cochange_pairs.csv \
        --output-dir outputs/variant_comparison_server/figures/cochange_analysis
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]


# ---------- Loading ----------

def _load_pair_predictions(pair_base: pathlib.Path) -> pd.DataFrame:
    frames = []
    for p in PERIODS:
        path = pair_base / f"train_{p}" / f"eval_{p}" / "pair_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["period"] = p
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No pair_predictions.csv under {pair_base}/train_*/eval_*/")
    return pd.concat(frames, ignore_index=True)


def _build_neighbor_map(cochange_csv: pathlib.Path) -> Dict[str, set]:
    """directory → set of co-change neighbor directories"""
    df = pd.read_csv(cochange_csv)
    neighbors: Dict[str, set] = {}
    for _, row in df.iterrows():
        d1, d2 = row["dir1"], row["dir2"]
        neighbors.setdefault(d1, set()).add(d2)
        neighbors.setdefault(d2, set()).add(d1)
    return neighbors


# ---------- Metrics ----------

def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or np.all(np.isnan(scores)):
        return np.nan
    mask = ~np.isnan(scores)
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(roc_auc_score(y[mask], scores[mask]))


def _agreement_rate(df: pd.DataFrame) -> float:
    """両モデルが中央値閾値で同じ判定 (positive/negative) を出した割合"""
    if df.empty:
        return np.nan
    irl_thresh = df["irl_dir_prob"].median()
    rf_thresh = df["rf_dir_prob"].median()
    irl_pos = df["irl_dir_prob"] >= irl_thresh
    rf_pos = df["rf_dir_prob"] >= rf_thresh
    return float((irl_pos == rf_pos).mean())


def _stratify(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """継続率 / IRL AUC / RF AUC / 合致率をグループ別に計算"""
    rows = []
    for grp, sub in df.groupby(group_col):
        if sub.empty:
            continue
        rows.append({
            group_col: grp,
            "n": len(sub),
            "continuation_rate": float(sub["label"].mean()),
            "irl_auc": _safe_auc(sub["label"].values, sub["irl_dir_prob"].values),
            "rf_auc": _safe_auc(sub["label"].values, sub["rf_dir_prob"].values),
            "agreement_rate": _agreement_rate(sub),
        })
    return pd.DataFrame(rows)


def _stratify_cross(df: pd.DataFrame, row_col: str, col_col: str = "period") -> pd.DataFrame:
    """row × col の AUC クロス集計"""
    pivot_rows = []
    for (rv, cv), sub in df.groupby([row_col, col_col]):
        if sub.empty:
            continue
        pivot_rows.append({
            row_col: rv, col_col: cv, "n": len(sub),
            "continuation_rate": float(sub["label"].mean()),
            "irl_auc": _safe_auc(sub["label"].values, sub["irl_dir_prob"].values),
            "rf_auc": _safe_auc(sub["label"].values, sub["rf_dir_prob"].values),
        })
    return pd.DataFrame(pivot_rows)


# ---------- Analysis 1: Hub score ----------

def analyze_hub(
    pair_df: pd.DataFrame,
    hub_csv: pathlib.Path,
    out_dir: pathlib.Path,
) -> pd.DataFrame:
    hub = pd.read_csv(hub_csv)
    df = pair_df.merge(
        hub[["directory", "hub_score"]], on="directory", how="left",
    )
    matched = df.dropna(subset=["hub_score"]).copy()
    print(f"[hub] {len(matched)}/{len(df)} pairs matched to hub_scores")

    matched["hub_q"] = pd.qcut(
        matched["hub_score"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )

    overall = _stratify(matched, "hub_q")
    overall["bucket"] = overall["hub_q"].astype(str)
    overall = overall[["bucket", "n", "continuation_rate",
                       "irl_auc", "rf_auc", "agreement_rate"]]

    by_window = _stratify_cross(matched, "hub_q", "period")
    by_window["hub_q"] = by_window["hub_q"].astype(str)

    csv_path = out_dir / "hub_score_stratified.csv"
    overall.to_csv(csv_path, index=False)
    by_window.to_csv(out_dir / "hub_score_stratified_by_window.csv", index=False)

    _plot_stratified(
        overall, x_col="bucket",
        title="Hub score 4分位 × 継続率・AUC (lstm_baseline, 全窓統合)",
        out_path=out_dir / "hub_score_stratified",
    )
    _plot_window_lines(
        by_window, group_col="hub_q",
        title="Hub × 窓: IRL/RF AUC の窓別推移",
        out_path=out_dir / "hub_score_window_lines",
    )

    return overall


# ---------- Analysis 2: Co-change neighbor coverage ----------

def analyze_coverage(
    pair_df: pd.DataFrame,
    cochange_csv: pathlib.Path,
    out_dir: pathlib.Path,
) -> pd.DataFrame:
    neighbors = _build_neighbor_map(cochange_csv)

    reviewer_dirs: Dict[str, set] = (
        pair_df.groupby("developer")["directory"].apply(set).to_dict()
    )

    def _coverage(row) -> Tuple[float, int, int]:
        nbrs = neighbors.get(row["directory"], set())
        if not nbrs:
            return (np.nan, 0, 0)
        own = reviewer_dirs.get(row["developer"], set()) - {row["directory"]}
        covered = len(nbrs & own)
        return (covered / len(nbrs), covered, len(nbrs))

    cov = pair_df.apply(_coverage, axis=1, result_type="expand")
    cov.columns = ["coverage", "covered_count", "neighbor_count"]
    df = pd.concat([pair_df, cov], axis=1)

    has_nbr = df["neighbor_count"] > 0
    print(f"[coverage] {has_nbr.sum()}/{len(df)} pairs have ≥1 co-change neighbor")
    print(
        f"[coverage] coverage stats among those: "
        f"mean={df.loc[has_nbr, 'coverage'].mean():.3f}, "
        f"median={df.loc[has_nbr, 'coverage'].median():.3f}, "
        f"p75={df.loc[has_nbr, 'coverage'].quantile(0.75):.3f}"
    )

    def _bin(v: float, has: bool) -> str:
        if not has:
            return "no_neighbors"
        if v == 0:
            return "0%"
        if v < 0.25:
            return "0-25%"
        if v < 0.50:
            return "25-50%"
        return "50%+"

    df["coverage_bin"] = [
        _bin(v, h) for v, h in zip(df["coverage"], has_nbr)
    ]
    bin_order = ["no_neighbors", "0%", "0-25%", "25-50%", "50%+"]
    df["coverage_bin"] = pd.Categorical(
        df["coverage_bin"], categories=bin_order, ordered=True,
    )

    overall = _stratify(df, "coverage_bin")
    overall["bucket"] = overall["coverage_bin"].astype(str)
    overall = overall[["bucket", "n", "continuation_rate",
                       "irl_auc", "rf_auc", "agreement_rate"]]
    overall["bucket"] = pd.Categorical(overall["bucket"], categories=bin_order, ordered=True)
    overall = overall.sort_values("bucket")

    by_window = _stratify_cross(df, "coverage_bin", "period")
    by_window["coverage_bin"] = by_window["coverage_bin"].astype(str)

    overall.to_csv(out_dir / "cochange_coverage_stratified.csv", index=False)
    by_window.to_csv(out_dir / "cochange_coverage_stratified_by_window.csv", index=False)

    _plot_stratified(
        overall, x_col="bucket",
        title="Co-change 近傍カバレッジ × 継続率・AUC (lstm_baseline, 全窓統合)",
        out_path=out_dir / "cochange_coverage_stratified",
    )
    _plot_window_lines(
        by_window, group_col="coverage_bin",
        title="Coverage × 窓: IRL/RF AUC の窓別推移",
        out_path=out_dir / "cochange_coverage_window_lines",
    )

    df_summary_path = out_dir / "cochange_coverage_summary.json"
    df_summary_path.write_text(json.dumps({
        "n_total": int(len(df)),
        "n_with_neighbors": int(has_nbr.sum()),
        "coverage_mean": float(df.loc[has_nbr, "coverage"].mean()),
        "coverage_median": float(df.loc[has_nbr, "coverage"].median()),
    }, indent=2))

    return overall


# ---------- Plot helpers ----------

def _plot_stratified(
    df: pd.DataFrame, x_col: str, title: str, out_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(df))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width / 2, df["irl_auc"], width, label="IRL_Dir AUC", color="#1f77b4")
    ax.bar(x + width / 2, df["rf_auc"], width, label="RF_Dir AUC", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str), rotation=15)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("AUC by stratum")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend()
    for i, n in enumerate(df["n"]):
        ax.text(i, 0.51, f"n={int(n)}", ha="center", fontsize=8, color="#444")

    ax = axes[1]
    ax.bar(x, df["continuation_rate"], color="#2ca02c", alpha=0.85)
    ax.plot(x, df["agreement_rate"], "o-", color="black", label="IRL/RF 合致率")
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str), rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("継続率 (緑) と IRL/RF 判定合致率 (黒線)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_window_lines(
    df: pd.DataFrame, group_col: str, title: str, out_path: pathlib.Path,
) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    groups = df[group_col].unique()
    cmap = plt.get_cmap("viridis")
    colors = {g: cmap(i / max(1, len(groups) - 1)) for i, g in enumerate(sorted(groups, key=str))}

    for metric, ax, ylabel in [("irl_auc", axes[0], "IRL_Dir AUC"),
                                ("rf_auc", axes[1], "RF_Dir AUC")]:
        for g in sorted(groups, key=str):
            sub = df[df[group_col] == g].sort_values("period")
            ax.plot(sub["period"], sub[metric], "o-", label=str(g),
                    color=colors[g], linewidth=1.8, markersize=6)
        ax.set_xlabel("予測窓")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-base", type=pathlib.Path, required=True)
    parser.add_argument("--hub-csv", type=pathlib.Path, required=True)
    parser.add_argument("--cochange-csv", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_df = _load_pair_predictions(args.pair_base)
    print(f"[load] {len(pair_df)} pair_prediction rows from {pair_df['period'].nunique()} windows")

    print("\n=== (2) Hub score 層別 ===")
    hub_overall = analyze_hub(pair_df, args.hub_csv, args.output_dir)
    print(hub_overall.to_string(index=False))

    print("\n=== (3) Co-change 近傍カバレッジ層別 ===")
    cov_overall = analyze_coverage(pair_df, args.cochange_csv, args.output_dir)
    print(cov_overall.to_string(index=False))


if __name__ == "__main__":
    main()
