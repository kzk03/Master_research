# -*- coding: utf-8 -*-
"""
静的依存 (06_static_dir_deps) と co-change グラフ (01_cochange_graph) の比較。

(a) Edge overlap: プロジェクト別に Jaccard / overlap 数。
    static は (src→dst) を無向化して co-change の (dir1, dir2) と比較。
(b) Hub score 相関: static_hub_score (in+out degree) と
    co-change hub_score の Spearman / Pearson (プロジェクト別 + 全体)。
(c) Coverage 分析: B-12 と同じ方式を static-dep neighbor で再実行し、
    継続率と IRL/RF AUC が co-change ベースの結果と整合するか検証。

Inputs:
    experiments/dependency_analysis/results/static_dir_edges.csv
    experiments/dependency_analysis/results/static_hub_scores.csv
    experiments/dependency_analysis/results/top_cochange_pairs.csv
    experiments/dependency_analysis/results/hub_scores.csv
    outputs/variant_comparison_server/lstm_baseline/train_*/eval_*/pair_predictions.csv

Outputs (--output-dir):
    edge_overlap.csv                プロジェクト別 Jaccard 等
    hub_correlation.csv             プロジェクト別 + 全体 Spearman/Pearson
    hub_correlation_scatter.pdf     全プロジェクト統合の散布図
    static_coverage_stratified.csv  B-12 と同じ層別表 (static-dep neighbor 版)
    static_coverage_stratified.pdf
    cochange_vs_static_coverage_compare.csv  B-12 結果との並列比較

Usage:
    uv run python scripts/analyze/compare/compare_static_vs_cochange.py \
        --static-edges experiments/dependency_analysis/results/static_dir_edges.csv \
        --static-hub experiments/dependency_analysis/results/static_hub_scores.csv \
        --cochange-pairs experiments/dependency_analysis/results/top_cochange_pairs.csv \
        --cochange-hub experiments/dependency_analysis/results/hub_scores.csv \
        --pair-base outputs/variant_comparison_server/lstm_baseline \
        --output-dir outputs/variant_comparison_server/figures/cochange_analysis
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]


# ---------- (a) Edge overlap ----------

def edge_overlap(
    static_edges: pd.DataFrame, cochange_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """プロジェクト別に edge set の Jaccard を計算"""
    # static は src/dst が同一プロジェクト内のものに絞る
    static_intra = static_edges.merge(
        cochange_pairs[["project"]].drop_duplicates(), on="project",
    )
    static_intra = static_intra.assign(
        in_proj=lambda d: d.apply(lambda r: _same_project_dirs(r, cochange_pairs), axis=1)
    )

    rows = []
    for project in sorted(set(cochange_pairs["project"])):
        # static: undirected edge set
        s = static_edges[static_edges["project"] == project]
        # filter to intra-project (dst が同プロジェクトの dir のみ)
        proj_dirs = set(
            cochange_pairs.loc[cochange_pairs["project"] == project, "dir1"]
        ) | set(
            cochange_pairs.loc[cochange_pairs["project"] == project, "dir2"]
        )
        s = s[s["src"].isin(proj_dirs) & s["dst"].isin(proj_dirs)]
        s_edges = set(frozenset((r["src"], r["dst"])) for _, r in s.iterrows())

        c = cochange_pairs[cochange_pairs["project"] == project]
        c_edges = set(frozenset((r["dir1"], r["dir2"])) for _, r in c.iterrows())

        inter = s_edges & c_edges
        union = s_edges | c_edges
        rows.append({
            "project": project,
            "static_edges": len(s_edges),
            "cochange_edges": len(c_edges),
            "intersection": len(inter),
            "static_only": len(s_edges - c_edges),
            "cochange_only": len(c_edges - s_edges),
            "jaccard": len(inter) / len(union) if union else np.nan,
            "static_recall_of_cochange": (
                len(inter) / len(c_edges) if c_edges else np.nan
            ),
            "cochange_recall_of_static": (
                len(inter) / len(s_edges) if s_edges else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _same_project_dirs(_row, _pairs):
    return True


# ---------- (b) Hub correlation ----------

def hub_correlation(
    static_hub: pd.DataFrame, cochange_hub: pd.DataFrame,
    out_path: pathlib.Path,
) -> pd.DataFrame:
    merged = cochange_hub.merge(
        static_hub[["project", "directory", "static_hub_score",
                    "in_degree", "out_degree"]],
        on=["project", "directory"], how="inner",
    )
    rows = []
    for project, sub in merged.groupby("project"):
        if len(sub) < 3:
            continue
        rho_s, _ = spearmanr(sub["hub_score"], sub["static_hub_score"])
        r_p, _ = pearsonr(sub["hub_score"], sub["static_hub_score"])
        rows.append({
            "project": project, "n": len(sub),
            "spearman": float(rho_s), "pearson": float(r_p),
        })
    overall_s, _ = spearmanr(merged["hub_score"], merged["static_hub_score"])
    overall_p, _ = pearsonr(merged["hub_score"], merged["static_hub_score"])
    rows.append({
        "project": "ALL", "n": len(merged),
        "spearman": float(overall_s), "pearson": float(overall_p),
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, (proj, sub) in enumerate(merged.groupby("project")):
        ax.scatter(sub["hub_score"], sub["static_hub_score"],
                   color=cmap(i % 10), label=proj.split("/")[-1], s=30, alpha=0.75)
    ax.set_xlabel("co-change hub_score")
    ax.set_ylabel("static dep hub_score (in_degree + out_degree)")
    ax.set_title(
        f"Hub score 相関 (Spearman={overall_s:.3f}, Pearson={overall_p:.3f}, n={len(merged)})"
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(rows)


# ---------- (c) Coverage analysis (static-dep neighbor 版) ----------

def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    mask = ~np.isnan(scores)
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(roc_auc_score(y[mask], scores[mask]))


def _agreement(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    irl_t = df["irl_dir_prob"].median()
    rf_t = df["rf_dir_prob"].median()
    return float(((df["irl_dir_prob"] >= irl_t) == (df["rf_dir_prob"] >= rf_t)).mean())


def _stratify(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for grp, sub in df.groupby(group_col):
        rows.append({
            group_col: grp, "n": len(sub),
            "continuation_rate": float(sub["label"].mean()),
            "irl_auc": _safe_auc(sub["label"].values, sub["irl_dir_prob"].values),
            "rf_auc": _safe_auc(sub["label"].values, sub["rf_dir_prob"].values),
            "agreement_rate": _agreement(sub),
        })
    return pd.DataFrame(rows)


def coverage_with_static(
    pair_df: pd.DataFrame, static_edges: pd.DataFrame, out_dir: pathlib.Path,
) -> pd.DataFrame:
    """B-12 と同じロジックを static-dep neighbor で実施"""
    neighbors: Dict[str, set] = {}
    for _, r in static_edges.iterrows():
        neighbors.setdefault(r["src"], set()).add(r["dst"])
        neighbors.setdefault(r["dst"], set()).add(r["src"])

    reviewer_dirs: Dict[str, set] = (
        pair_df.groupby("developer")["directory"].apply(set).to_dict()
    )

    cov, neighbor_count = [], []
    for _, row in pair_df.iterrows():
        nbrs = neighbors.get(row["directory"], set())
        if not nbrs:
            cov.append(np.nan)
            neighbor_count.append(0)
            continue
        own = reviewer_dirs.get(row["developer"], set()) - {row["directory"]}
        cov.append(len(nbrs & own) / len(nbrs))
        neighbor_count.append(len(nbrs))

    df = pair_df.copy()
    df["coverage"] = cov
    df["neighbor_count"] = neighbor_count

    has_nbr = df["neighbor_count"] > 0
    print(
        f"[static_coverage] {has_nbr.sum()}/{len(df)} pairs have ≥1 static-dep neighbor"
    )

    def _bin(v, has):
        if not has:
            return "no_neighbors"
        if v == 0:
            return "0%"
        if v < 0.25:
            return "0-25%"
        if v < 0.50:
            return "25-50%"
        return "50%+"

    df["coverage_bin"] = [_bin(v, h) for v, h in zip(df["coverage"], has_nbr)]
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

    overall.to_csv(out_dir / "static_coverage_stratified.csv", index=False)

    # 図
    x = np.arange(len(overall))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, overall["irl_auc"], width, label="IRL_Dir", color="#1f77b4")
    ax.bar(x + width/2, overall["rf_auc"], width, label="RF_Dir", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(overall["bucket"].astype(str), rotation=15)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC"); ax.set_title("AUC")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4); ax.legend()
    for i, n in enumerate(overall["n"]):
        ax.text(i, 0.51, f"n={int(n)}", ha="center", fontsize=8, color="#444")

    ax = axes[1]
    ax.bar(x, overall["continuation_rate"], color="#2ca02c", alpha=0.85)
    ax.plot(x, overall["agreement_rate"], "o-", color="black", label="IRL/RF 合致率")
    ax.set_xticks(x); ax.set_xticklabels(overall["bucket"].astype(str), rotation=15)
    ax.set_ylim(0, 1.0); ax.set_ylabel("rate")
    ax.set_title("継続率 (緑) と 合致率 (黒)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4); ax.legend()

    fig.suptitle("Static-dep coverage × 継続率・AUC (lstm_baseline)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "static_coverage_stratified.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    return overall


# ---------- Main ----------

def _load_pairs(pair_base: pathlib.Path) -> pd.DataFrame:
    frames = []
    for p in PERIODS:
        path = pair_base / f"train_{p}" / f"eval_{p}" / "pair_predictions.csv"
        if path.exists():
            d = pd.read_csv(path)
            d["period"] = p
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-edges", type=pathlib.Path, required=True)
    parser.add_argument("--static-hub", type=pathlib.Path, required=True)
    parser.add_argument("--cochange-pairs", type=pathlib.Path, required=True)
    parser.add_argument("--cochange-hub", type=pathlib.Path, required=True)
    parser.add_argument("--pair-base", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    static_edges = pd.read_csv(args.static_edges)
    static_hub = pd.read_csv(args.static_hub)
    cochange_pairs = pd.read_csv(args.cochange_pairs)
    cochange_hub = pd.read_csv(args.cochange_hub)
    pair_df = _load_pairs(args.pair_base)
    print(f"[load] static edges={len(static_edges)}, cochange pairs={len(cochange_pairs)}, "
          f"pair preds={len(pair_df)}")

    print("\n=== (a) Edge overlap ===")
    overlap = edge_overlap(static_edges, cochange_pairs)
    overlap.to_csv(args.output_dir / "edge_overlap.csv", index=False)
    print(overlap.to_string(index=False))

    print("\n=== (b) Hub correlation ===")
    corr = hub_correlation(
        static_hub, cochange_hub,
        args.output_dir / "hub_correlation_scatter",
    )
    corr.to_csv(args.output_dir / "hub_correlation.csv", index=False)
    print(corr.to_string(index=False))

    print("\n=== (c) Static-dep coverage 層別 ===")
    static_cov = coverage_with_static(pair_df, static_edges, args.output_dir)
    print(static_cov.to_string(index=False))

    # B-12 (co-change coverage) との並列比較
    cochange_cov_path = args.output_dir / "cochange_coverage_stratified.csv"
    if cochange_cov_path.exists():
        cc = pd.read_csv(cochange_cov_path)
        compare = cc[["bucket", "n", "continuation_rate", "irl_auc", "rf_auc"]].rename(
            columns={c: f"cochange_{c}" for c in ["n", "continuation_rate", "irl_auc", "rf_auc"]},
        ).merge(
            static_cov[["bucket", "n", "continuation_rate", "irl_auc", "rf_auc"]].rename(
                columns={c: f"static_{c}" for c in ["n", "continuation_rate", "irl_auc", "rf_auc"]},
            ),
            on="bucket", how="outer",
        )
        compare.to_csv(args.output_dir / "cochange_vs_static_coverage_compare.csv", index=False)
        print("\n=== B-12 (co-change) vs (c) (static-dep) 比較 ===")
        print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
