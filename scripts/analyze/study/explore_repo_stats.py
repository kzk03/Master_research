#!/usr/bin/env python3
"""
combined_raw_{N}.csv の探索的集計（A. 下見集計）

目的:
  231 repo に拡張した combined_raw_231.csv の構造を把握し、
  論文の A-8 表に使える per-tier 統計を出す。

出力:
  outputs/exploration/repo_stats.csv     : per-repo メトリクス
  outputs/exploration/tier_summary.csv   : tier 別集計
  outputs/exploration/team_summary.csv   : team 別集計
  outputs/figures/explore_*.pdf          : 4 種の可視化

使い方:
  uv run python scripts/analyze/study/explore_repo_stats.py
  uv run python scripts/analyze/study/explore_repo_stats.py \\
      --combined data/combined_raw_231.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIER_ORDER = ["大", "中", "小", "極小", "非分類"]
TIER_COLORS = {"大": "#1f77b4", "中": "#ff7f0e", "小": "#2ca02c",
               "極小": "#d62728", "非分類": "#999999"}
# 図表向けの英語ラベル（DejaVu Sans に CJK が無いため）
TIER_LABEL_EN = {"大": "Large", "中": "Medium", "小": "Small",
                 "極小": "Tiny", "非分類": "Unclassified"}


def load_repo_meta(repos_csv: Path) -> dict:
    """service_teams_repos.csv から repo → {team, tier} を構築。"""
    meta = {}
    with open(repos_csv) as f:
        for r in csv.DictReader(f):
            meta[r["repo"]] = {"team": r["team"], "tier": r["tier"]}
    return meta


def compute_repo_stats(df: pd.DataFrame, repos_meta: dict) -> pd.DataFrame:
    """per-repo の集計を行う。"""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    rows = []
    for repo, grp in df.groupby("project"):
        meta = repos_meta.get(repo, {"team": "?", "tier": "?"})

        # 期間計算
        ts_min = grp["timestamp"].min()
        ts_max = grp["timestamp"].max()
        if pd.notna(ts_min) and pd.notna(ts_max):
            active_months = max(1, (ts_max - ts_min).days / 30.44)
        else:
            active_months = np.nan

        n_rows = len(grp)
        rows.append({
            "repo": repo,
            "team": meta["team"],
            "tier": meta["tier"],
            "n_rows": n_rows,
            "n_changes": grp["change_id"].nunique(),
            "n_reviewers": grp["email"].nunique(),
            "n_owners": grp["owner_email"].nunique() if "owner_email" in grp.columns else np.nan,
            "acceptance_rate": grp["label"].mean(),
            "first_review": ts_min,
            "last_review": ts_max,
            "active_months": active_months,
            "rows_per_month": n_rows / active_months if active_months else np.nan,
        })

    return pd.DataFrame(rows)


def tier_aggregate(repo_stats: pd.DataFrame) -> pd.DataFrame:
    """tier 別の集計。"""
    agg = repo_stats.groupby("tier", observed=True).agg(
        n_repos=("repo", "count"),
        total_rows=("n_rows", "sum"),
        total_changes=("n_changes", "sum"),
        total_reviewers=("n_reviewers", "sum"),
        median_rows=("n_rows", "median"),
        median_reviewers=("n_reviewers", "median"),
        median_acceptance=("acceptance_rate", "median"),
        median_active_months=("active_months", "median"),
        median_rows_per_month=("rows_per_month", "median"),
    ).reset_index()

    # 順序固定
    agg["tier"] = pd.Categorical(agg["tier"], categories=TIER_ORDER, ordered=True)
    agg = agg.sort_values("tier").reset_index(drop=True)
    return agg


def team_aggregate(repo_stats: pd.DataFrame) -> pd.DataFrame:
    """team 別の集計。"""
    agg = repo_stats.groupby(["team", "tier"], observed=True).agg(
        n_repos=("repo", "count"),
        total_rows=("n_rows", "sum"),
        total_reviewers=("n_reviewers", "sum"),
        avg_acceptance=("acceptance_rate", "mean"),
    ).reset_index()
    return agg.sort_values("total_rows", ascending=False).reset_index(drop=True)


def cross_repo_reviewers(df: pd.DataFrame, repos_meta: dict) -> pd.DataFrame:
    """レビュアーが何 repo / 何 team / 何 tier にまたがるかを集計。"""
    df = df.copy()
    df["team"] = df["project"].map(lambda r: repos_meta.get(r, {}).get("team", "?"))
    df["tier"] = df["project"].map(lambda r: repos_meta.get(r, {}).get("tier", "?"))

    by_reviewer = df.groupby("email").agg(
        n_repos=("project", "nunique"),
        n_teams=("team", "nunique"),
        n_tiers=("tier", "nunique"),
        n_reviews=("change_id", "count"),
        acceptance_rate=("label", "mean"),
    ).reset_index()

    return by_reviewer


def plot_tier_boxplots(repo_stats: pd.DataFrame, out_dir: Path):
    """tier 別の各種 box plot を 4 枚生成。"""
    metrics = [
        ("n_rows", "Review rows per repo", True),
        ("n_reviewers", "Unique reviewers per repo", True),
        ("acceptance_rate", "Acceptance rate", False),
        ("rows_per_month", "Reviews per month", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, label, log) in zip(axes.flat, metrics):
        data = []
        labels = []
        tier_keys = []
        for t in TIER_ORDER:
            sub = repo_stats[repo_stats["tier"] == t][col].dropna().values
            if len(sub) > 0:
                data.append(sub)
                labels.append(f"{TIER_LABEL_EN[t]}\n(n={len(sub)})")
                tier_keys.append(t)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, t in zip(bp["boxes"], tier_keys):
            patch.set_facecolor(TIER_COLORS.get(t, "gray"))
            patch.set_alpha(0.6)

        if log:
            ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by tier")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-tier activity distribution (231 repos)", fontsize=14)
    plt.tight_layout()
    out_path = out_dir / "explore_tier_boxplots.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    return out_path


def plot_reviewer_spread(cross: pd.DataFrame, out_dir: Path):
    """レビュアーが何 repo にまたがるかのヒストグラム。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(cross["n_repos"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of repos a reviewer touches")
    axes[0].set_ylabel("Number of reviewers (log)")
    axes[0].set_title("Reviewer spread across repos")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(cross["n_teams"], bins=range(1, cross["n_teams"].max() + 2),
                 color="darkorange", edgecolor="black", alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of service teams a reviewer touches")
    axes[1].set_ylabel("Number of reviewers (log)")
    axes[1].set_title("Reviewer spread across teams")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "explore_reviewer_spread.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    return out_path


def plot_top_repos(repo_stats: pd.DataFrame, out_dir: Path, n: int = 30):
    """活動量上位 N repo の横棒グラフ。"""
    top = repo_stats.nlargest(n, "n_rows").iloc[::-1]
    colors = [TIER_COLORS.get(t, "gray") for t in top["tier"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(top["repo"].str.replace("openstack/", ""), top["n_rows"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Number of review rows (log)")
    ax.set_title(f"Top {n} repos by review activity")
    ax.grid(True, alpha=0.3, axis="x")

    # 凡例（CJK 回避のため英語）
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t]) for t in TIER_ORDER if t in top["tier"].values]
    labels = [TIER_LABEL_EN[t] for t in TIER_ORDER if t in top["tier"].values]
    ax.legend(handles, labels, title="Tier", loc="lower right")

    plt.tight_layout()
    out_path = out_dir / "explore_top_repos.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    return out_path


def plot_activity_timeline(df: pd.DataFrame, repos_meta: dict, out_dir: Path):
    """tier 別の月次活動推移。"""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["tier"] = df["project"].map(lambda r: repos_meta.get(r, {}).get("tier", "?"))
    df["year_month"] = df["timestamp"].dt.to_period("M")

    monthly = df.groupby(["year_month", "tier"]).size().unstack(fill_value=0)
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    for t in TIER_ORDER:
        if t in monthly.columns:
            ax.plot(monthly.index, monthly[t], label=TIER_LABEL_EN[t],
                    color=TIER_COLORS[t], linewidth=1.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Review rows per month")
    ax.set_title("Monthly review activity by tier (2020-2026)")
    ax.legend(title="Tier")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "explore_activity_timeline.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combined", type=Path,
                        default=Path("data/combined_raw_231.csv"),
                        help="統合済みレビューデータ CSV")
    parser.add_argument("--repos-csv", type=Path,
                        default=Path("data/service_teams_repos.csv"),
                        help="repo メタデータ CSV")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/exploration"),
                        help="CSV 出力ディレクトリ")
    parser.add_argument("--figure-dir", type=Path,
                        default=Path("outputs/figures"),
                        help="PDF 図表出力ディレクトリ")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 読み込み ---
    print(f"=== combined_raw 読み込み: {args.combined} ===")
    df = pd.read_csv(args.combined, low_memory=False)
    print(f"  total rows  : {len(df):,}")
    print(f"  unique repos: {df['project'].nunique()}")
    print(f"  unique reviewers: {df['email'].nunique():,}")

    repos_meta = load_repo_meta(args.repos_csv)

    # --- 2. per-repo 集計 ---
    print(f"\n=== per-repo 集計 ===")
    repo_stats = compute_repo_stats(df, repos_meta)
    repo_stats_path = args.output_dir / "repo_stats.csv"
    repo_stats.to_csv(repo_stats_path, index=False)
    print(f"  → {repo_stats_path} ({len(repo_stats)} rows)")

    # --- 3. tier 別集計 ---
    print(f"\n=== tier 別集計 ===")
    tier_summary = tier_aggregate(repo_stats)
    tier_summary_path = args.output_dir / "tier_summary.csv"
    tier_summary.to_csv(tier_summary_path, index=False)
    print(tier_summary.to_string(index=False))
    print(f"  → {tier_summary_path}")

    # --- 4. team 別集計 ---
    print(f"\n=== team 別集計 (Top 15) ===")
    team_summary = team_aggregate(repo_stats)
    team_summary_path = args.output_dir / "team_summary.csv"
    team_summary.to_csv(team_summary_path, index=False)
    print(team_summary.head(15).to_string(index=False))
    print(f"  → {team_summary_path} (全 {len(team_summary)} teams)")

    # --- 5. レビュアーの cross-repo 分布 ---
    print(f"\n=== レビュアー cross-repo 分布 ===")
    cross = cross_repo_reviewers(df, repos_meta)
    cross_path = args.output_dir / "reviewer_cross_repo.csv"
    cross.to_csv(cross_path, index=False)
    print(f"  total reviewers: {len(cross):,}")
    print(f"  1 repo のみ     : {(cross['n_repos'] == 1).sum():,} ({(cross['n_repos'] == 1).mean()*100:.1f}%)")
    print(f"  2 repos        : {(cross['n_repos'] == 2).sum():,}")
    print(f"  3+ repos       : {(cross['n_repos'] >= 3).sum():,}")
    print(f"  10+ repos      : {(cross['n_repos'] >= 10).sum():,}")
    print(f"  1 team のみ     : {(cross['n_teams'] == 1).sum():,} ({(cross['n_teams'] == 1).mean()*100:.1f}%)")
    print(f"  3+ teams       : {(cross['n_teams'] >= 3).sum():,}")
    print(f"  → {cross_path}")

    # --- 6. 可視化 ---
    print(f"\n=== 可視化 (PDF) ===")
    p1 = plot_tier_boxplots(repo_stats, args.figure_dir)
    print(f"  → {p1}")
    p2 = plot_reviewer_spread(cross, args.figure_dir)
    print(f"  → {p2}")
    p3 = plot_top_repos(repo_stats, args.figure_dir)
    print(f"  → {p3}")
    p4 = plot_activity_timeline(df, repos_meta, args.figure_dir)
    print(f"  → {p4}")

    print(f"\n=== 完了 ===")


if __name__ == "__main__":
    main()
