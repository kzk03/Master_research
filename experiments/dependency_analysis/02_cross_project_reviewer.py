"""
クロスプロジェクトレビュアー分析

複数プロジェクトでレビューしている人 vs 単一プロジェクトの人で
継続率・IRL予測精度に差があるかを分析する。
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def analyze_cross_project(
    data_path: Path,
    pair_predictions_path: Path | None,
    output_dir: Path,
):
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # レビュアーごとのプロジェクト数
    reviewer_projects = df.groupby("email")["project"].nunique().reset_index()
    reviewer_projects.columns = ["email", "n_projects"]

    # レビュアーごとの基本統計
    reviewer_stats = df.groupby("email").agg(
        n_reviews=("change_id", "count"),
        n_accept=("label", "sum"),
        first_review=("timestamp", "min"),
        last_review=("timestamp", "max"),
    ).reset_index()
    reviewer_stats["acceptance_rate"] = reviewer_stats["n_accept"] / reviewer_stats["n_reviews"]
    reviewer_stats["active_days"] = (
        reviewer_stats["last_review"] - reviewer_stats["first_review"]
    ).dt.days

    reviewer_stats = reviewer_stats.merge(reviewer_projects, on="email")

    # 層別化
    reviewer_stats["project_tier"] = pd.cut(
        reviewer_stats["n_projects"],
        bins=[0, 1, 2, 10],
        labels=["single", "2_projects", "3+_projects"],
    )

    # 層ごとの統計
    print("=== Cross-project reviewer statistics ===")
    tier_stats = reviewer_stats.groupby("project_tier", observed=True).agg(
        n_reviewers=("email", "count"),
        avg_reviews=("n_reviews", "mean"),
        avg_acceptance=("acceptance_rate", "mean"),
        avg_active_days=("active_days", "mean"),
    )
    print(tier_stats.to_string())
    tier_stats.to_csv(output_dir / "cross_project_stats.csv")

    # 分布
    print(f"\nProject count distribution:")
    print(reviewer_stats["n_projects"].describe())
    print(f"\nValue counts:")
    print(reviewer_stats["n_projects"].value_counts().sort_index())

    # pair_predictions との突き合わせ
    if pair_predictions_path and pair_predictions_path.exists():
        print("\n=== Cross-project tier vs IRL prediction ===")
        pairs = pd.read_csv(pair_predictions_path)
        pairs = pairs.merge(
            reviewer_stats[["email", "n_projects", "project_tier"]],
            left_on="developer",
            right_on="email",
            how="left",
        )

        for tier, group in pairs.groupby("project_tier", observed=True):
            n = len(group)
            cont_rate = group["label"].mean()
            irl_auc = rf_auc = None
            if group["label"].nunique() == 2:
                irl_valid = group.dropna(subset=["irl_dir_prob"])
                rf_valid = group.dropna(subset=["rf_dir_prob"])
                if len(irl_valid) > 0 and irl_valid["label"].nunique() == 2:
                    irl_auc = roc_auc_score(
                        irl_valid["label"], irl_valid["irl_dir_prob"]
                    )
                if len(rf_valid) > 0 and rf_valid["label"].nunique() == 2:
                    rf_auc = roc_auc_score(rf_valid["label"], rf_valid["rf_dir_prob"])

            irl_str = f"{irl_auc:.3f}" if irl_auc is not None else "N/A"
            rf_str = f"{rf_auc:.3f}" if rf_auc is not None else "N/A"
            print(
                f"  {tier}: n={n}, cont_rate={cont_rate:.3f}, "
                f"IRL_AUC={irl_str}, RF_AUC={rf_str}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="combined_raw.csv")
    parser.add_argument("--pair-predictions", type=Path, help="pair_predictions.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyze_cross_project(args.data, args.pair_predictions, args.output_dir)


if __name__ == "__main__":
    main()
