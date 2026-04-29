"""
dependency_analysis 結果の可視化

results/ 内の CSV / JSON を読み込み、以下の図を PDF で出力する。
1. ディレクトリ別 hub score 分布（上位 + 全体ヒストグラム）
2. Co-change ペア上位
3. クロスプロジェクトレビュアー統計
4. レビュアー coherence 分布
5. 期間別レビュアーステータス（stacked bar）
6. 状態遷移ヒートマップ
7. Hub tier 別 継続率・AUC
8. プロジェクト別の変動率
"""

import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType for editable text in PDF

# 日本語フォントは使わず英語表記で統一
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})


def plot_hub_scores(hub_df: pd.DataFrame, output_dir: Path):
    """プロジェクト別ディレクトリ hub score"""
    projects = sorted(hub_df["project"].unique())
    n_proj = len(projects)
    ncols = 5
    nrows = (n_proj + ncols - 1) // ncols

    # --- Per-project top directories ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, project in enumerate(projects):
        ax = axes[idx]
        proj = hub_df[hub_df["project"] == project].sort_values(
            "hub_score", ascending=False
        ).head(10)
        short = project.replace("openstack/", "")
        ax.barh(range(len(proj)), proj["hub_score"], color="steelblue")
        ax.set_yticks(range(len(proj)))
        # ディレクトリ名からプロジェクト名プレフィックスを短縮
        labels = [d.replace(f"{short}/", "") for d in proj["directory"]]
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(short, fontsize=11)
        if idx % ncols == 0:
            ax.set_xlabel("Hub Score")

    for idx in range(n_proj, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Top 10 Hub Directories per Project", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "hub_scores_per_project.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("  -> hub_scores_per_project.pdf")

    # --- Hub score histogram across all projects ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(hub_df["hub_score"], bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("Hub Score")
    ax.set_ylabel("Count")
    ax.set_title("Hub Score Distribution (All Projects, Boilerplate Excluded)")
    fig.tight_layout()
    fig.savefig(output_dir / "hub_scores_histogram.pdf", format="pdf")
    plt.close(fig)
    print("  -> hub_scores_histogram.pdf")


def plot_cochange_pairs(pairs_df: pd.DataFrame, output_dir: Path):
    """プロジェクト別 co-change ペア上位"""
    projects = sorted(pairs_df["project"].unique())
    n_proj = len(projects)
    ncols = 5
    nrows = (n_proj + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, project in enumerate(projects):
        ax = axes[idx]
        proj = pairs_df[pairs_df["project"] == project].head(8)
        short = project.replace("openstack/", "")
        # ラベル短縮: プロジェクト名プレフィックス除去
        labels = []
        for _, r in proj.iterrows():
            d1 = r["dir1"].replace(f"{short}/", "")
            d2 = r["dir2"].replace(f"{short}/", "")
            labels.append(f"{d1} <-> {d2}")
        ax.barh(range(len(proj)), proj["cochange_count"], color="darkorange")
        ax.set_yticks(range(len(proj)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(short, fontsize=11)

    for idx in range(n_proj, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Top Co-change Pairs per Project (Boilerplate Excluded)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "cochange_pairs_per_project.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("  -> cochange_pairs_per_project.pdf")


def plot_cross_project_stats(stats_df: pd.DataFrame, output_dir: Path):
    """クロスプロジェクトレビュアー統計"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    tiers = stats_df["project_tier"]

    ax = axes[0]
    ax.bar(tiers, stats_df["n_reviewers"], color="teal")
    ax.set_ylabel("# Reviewers")
    ax.set_title("Reviewers by Project Tier")

    ax = axes[1]
    ax.bar(tiers, stats_df["avg_reviews"], color="coral")
    ax.set_ylabel("Avg Reviews")
    ax.set_title("Avg Reviews per Reviewer")

    ax = axes[2]
    ax.bar(tiers, stats_df["avg_acceptance"], color="mediumpurple")
    ax.set_ylabel("Avg Acceptance Rate")
    ax.set_title("Acceptance Rate by Tier")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_dir / "cross_project_reviewer_stats.pdf", format="pdf")
    plt.close(fig)
    print("  -> cross_project_reviewer_stats.pdf")


def plot_coherence_distribution(coherence_df: pd.DataFrame, output_dir: Path):
    """Reviewer coherence 分布"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(coherence_df["coherence"], bins=30, color="seagreen", edgecolor="white")
    ax.set_xlabel("Coherence Score")
    ax.set_ylabel("Count")
    ax.set_title("Reviewer Coherence Distribution")

    ax = axes[1]
    ax.scatter(
        coherence_df["n_dirs"],
        coherence_df["coherence"],
        alpha=0.3, s=10, color="seagreen",
    )
    ax.set_xlabel("# Directories Reviewed")
    ax.set_ylabel("Coherence Score")
    ax.set_title("Coherence vs # Directories")

    fig.tight_layout()
    fig.savefig(output_dir / "reviewer_coherence_distribution.pdf", format="pdf")
    plt.close(fig)
    print("  -> reviewer_coherence_distribution.pdf")


def plot_fluctuation_stacked(fluct_df: pd.DataFrame, output_dir: Path):
    """期間別ステータス stacked bar（exclude 除外）"""
    windows = ["0-3m", "3-6m", "6-9m", "9-12m"]
    statuses = ["accept", "reject", "weak_neg"]
    colors = {"accept": "#4CAF50", "reject": "#F44336", "weak_neg": "#FFC107"}

    # exclude 以外のレビュアーのみ
    active = fluct_df[
        fluct_df[windows].apply(lambda r: any(v != "exclude" for v in r), axis=1)
    ]

    counts = {}
    for s in statuses:
        counts[s] = [int((active[w] == s).sum()) for w in windows]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(windows))
    bottom = np.zeros(len(windows))
    for s in statuses:
        vals = np.array(counts[s])
        ax.bar(x, vals, bottom=bottom, label=s, color=colors[s], width=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_xlabel("Time Window")
    ax.set_ylabel("# Reviewers")
    ax.set_title("Reviewer Status by Time Window (excl. excluded)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fluctuation_stacked_bar.pdf", format="pdf")
    plt.close(fig)
    print("  -> fluctuation_stacked_bar.pdf")


def plot_transition_heatmap(summary: dict, output_dir: Path):
    """状態遷移ヒートマップ"""
    statuses = ["accept", "reject", "weak_neg"]
    transitions = summary.get("transitions", {})
    matrix = np.zeros((len(statuses), len(statuses)))
    for i, from_s in enumerate(statuses):
        for j, to_s in enumerate(statuses):
            key = f"{from_s}->{to_s}"
            matrix[i, j] = transitions.get(key, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(statuses)))
    ax.set_xticklabels(statuses)
    ax.set_yticks(range(len(statuses)))
    ax.set_yticklabels(statuses)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Reviewer State Transitions")

    for i in range(len(statuses)):
        for j in range(len(statuses)):
            val = int(matrix[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > matrix.max() * 0.6 else "black",
                    fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_dir / "transition_heatmap.pdf", format="pdf")
    plt.close(fig)
    print("  -> transition_heatmap.pdf")


def plot_hub_vs_continuation(hub_cont_df: pd.DataFrame, output_dir: Path):
    """Hub tier 別 継続率・AUC"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tiers = hub_cont_df["hub_tier"]

    ax = axes[0]
    ax.bar(tiers, hub_cont_df["continuation_rate"], color="steelblue")
    ax.set_ylabel("Continuation Rate")
    ax.set_title("Continuation Rate by Hub Tier")
    ax.set_ylim(0, 1)

    ax = axes[1]
    x = np.arange(len(tiers))
    w = 0.35
    ax.bar(x - w / 2, hub_cont_df["irl_dir_auc"], w, label="IRL", color="steelblue")
    ax.bar(x + w / 2, hub_cont_df["rf_dir_auc"], w, label="RF", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Prediction AUC by Hub Tier")
    ax.set_ylim(0.5, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "hub_tier_continuation_auc.pdf", format="pdf")
    plt.close(fig)
    print("  -> hub_tier_continuation_auc.pdf")


def plot_project_fluctuation(data_path: Path, fluct_df: pd.DataFrame, output_dir: Path):
    """プロジェクト別の変動率"""
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    base_date = df["timestamp"].max() - pd.DateOffset(months=12)
    base_date = base_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    windows = ["0-3m", "3-6m", "6-9m", "9-12m"]
    active_statuses = {"accept", "reject", "weak_neg"}
    negative_statuses = {"reject", "weak_neg"}

    TIME_WINDOWS = [
        ("0-3m", 0, 3),
        ("3-6m", 3, 6),
        ("6-9m", 6, 9),
        ("9-12m", 9, 12),
    ]

    project_results = []
    for project in sorted(df["project"].unique()):
        proj_df = df[df["project"] == project]
        ext_start = base_date
        ext_end = base_date + pd.DateOffset(months=12)
        ext_df = proj_df[
            (proj_df["timestamp"] >= ext_start) & (proj_df["timestamp"] < ext_end)
        ]
        ext_reviewers = set(ext_df["email"].unique())

        classified_rows = []
        for email, group in proj_df.groupby("email"):
            row = {"email": email}
            for window_name, start_m, end_m in TIME_WINDOWS:
                ws = base_date + pd.DateOffset(months=start_m)
                we = base_date + pd.DateOffset(months=end_m)
                wdata = group[(group["timestamp"] >= ws) & (group["timestamp"] < we)]
                if len(wdata) > 0:
                    row[window_name] = "accept" if wdata["label"].sum() > 0 else "reject"
                elif email in ext_reviewers:
                    row[window_name] = "weak_neg"
                else:
                    row[window_name] = "exclude"
            classified_rows.append(row)

        classified = pd.DataFrame(classified_rows)
        # multi-period
        classified["n_active"] = classified[windows].apply(
            lambda r: sum(1 for v in r if v in active_statuses), axis=1
        )
        multi = classified[classified["n_active"] >= 2]
        if len(multi) == 0:
            continue

        def has_fluct(row):
            states = [row[w] for w in windows if row[w] in active_statuses]
            return "accept" in states and any(s in negative_statuses for s in states)

        n_fluct = multi.apply(has_fluct, axis=1).sum()
        short_name = project.replace("openstack/", "")

        # Per-window accept counts
        per_window = {}
        for w in windows:
            active_in_w = classified[classified[w] != "exclude"]
            per_window[w] = {
                "accept": int((active_in_w[w] == "accept").sum()),
                "reject": int((active_in_w[w] == "reject").sum()),
                "weak_neg": int((active_in_w[w] == "weak_neg").sum()),
            }

        project_results.append({
            "project": short_name,
            "n_multi": len(multi),
            "n_fluct": int(n_fluct),
            "fluct_rate": n_fluct / len(multi) * 100,
            "per_window": per_window,
        })

    proj_res_df = pd.DataFrame(project_results).sort_values("n_multi", ascending=True)

    # --- Figure 1: Project fluctuation rate ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.barh(proj_res_df["project"], proj_res_df["n_multi"], color="steelblue",
            label="Stable", alpha=0.7)
    ax.barh(proj_res_df["project"], proj_res_df["n_fluct"], color="tomato",
            label="Fluctuating")
    ax.set_xlabel("# Multi-period Reviewers")
    ax.set_title("Fluctuating vs Stable Reviewers by Project")
    ax.legend()

    ax = axes[1]
    ax.barh(proj_res_df["project"], proj_res_df["fluct_rate"], color="tomato")
    ax.set_xlabel("Fluctuation Rate (%)")
    ax.set_title("Fluctuation Rate by Project")
    ax.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(output_dir / "project_fluctuation_rate.pdf", format="pdf")
    plt.close(fig)
    print("  -> project_fluctuation_rate.pdf")

    # --- Figure 2: Per-project per-window stacked bar ---
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
    axes = axes.flatten()
    statuses = ["accept", "reject", "weak_neg"]
    colors = {"accept": "#4CAF50", "reject": "#F44336", "weak_neg": "#FFC107"}

    for idx, res in enumerate(sorted(project_results, key=lambda x: x["project"])):
        ax = axes[idx]
        x = np.arange(len(windows))
        bottom = np.zeros(len(windows))
        for s in statuses:
            vals = np.array([res["per_window"][w][s] for w in windows])
            ax.bar(x, vals, bottom=bottom, color=colors[s], width=0.6,
                   label=s if idx == 0 else None)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(windows, fontsize=7, rotation=45)
        ax.set_title(res["project"], fontsize=10)
        if idx % 5 == 0:
            ax.set_ylabel("# Reviewers")

    # Hide unused axes
    for idx in range(len(project_results), len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in statuses]
    fig.legend(handles, statuses, loc="lower center", ncol=3, fontsize=10)
    fig.suptitle("Reviewer Status by Project and Time Window", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "project_window_stacked.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("  -> project_window_stacked.pdf")


def main():
    parser = argparse.ArgumentParser(description="Visualize dependency analysis results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/combined_raw.csv"),
        help="combined_raw.csv (for per-project fluctuation)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/dependency_analysis/figures"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rd = args.results_dir

    print("=== Generating figures ===")

    # 1. Hub scores
    hub_path = rd / "hub_scores.csv"
    if hub_path.exists():
        hub_df = pd.read_csv(hub_path)
        plot_hub_scores(hub_df, args.output_dir)

    # 2. Co-change pairs
    pairs_path = rd / "top_cochange_pairs.csv"
    if pairs_path.exists():
        pairs_df = pd.read_csv(pairs_path)
        plot_cochange_pairs(pairs_df, args.output_dir)

    # 3. Cross-project stats
    cross_path = rd / "cross_project_stats.csv"
    if cross_path.exists():
        cross_df = pd.read_csv(cross_path)
        plot_cross_project_stats(cross_df, args.output_dir)

    # 4. Coherence
    coh_path = rd / "reviewer_coherence.csv"
    if coh_path.exists():
        coh_df = pd.read_csv(coh_path)
        plot_coherence_distribution(coh_df, args.output_dir)

    # 5. Fluctuation stacked bar
    fluct_path = rd / "reviewer_fluctuation.csv"
    if fluct_path.exists():
        fluct_df = pd.read_csv(fluct_path)
        plot_fluctuation_stacked(fluct_df, args.output_dir)

    # 6. Transition heatmap
    summary_path = rd / "fluctuation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        plot_transition_heatmap(summary, args.output_dir)

    # 7. Hub tier vs continuation
    hub_cont_path = rd / "hub_vs_continuation.csv"
    if hub_cont_path.exists():
        hub_cont_df = pd.read_csv(hub_cont_path)
        plot_hub_vs_continuation(hub_cont_df, args.output_dir)

    # 8. Per-project fluctuation
    if fluct_path.exists() and args.data.exists():
        fluct_df = pd.read_csv(fluct_path)
        plot_project_fluctuation(args.data, fluct_df, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
