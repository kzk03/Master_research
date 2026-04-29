"""
ディレクトリ間レビュアー重複分析

同じレビュアーが担当するディレクトリ群を分析し、
co-change で結ばれた関連ディレクトリ群を担当するレビュアーほど
継続しやすいかを検証する。
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def build_reviewer_dir_map(
    raw_json_paths: list[Path],
    data_path: Path,
) -> pd.DataFrame:
    """レビュアーごとの担当ディレクトリ一覧を構築"""
    # change_id (= JSON の "id" フィールド) -> directories
    change_dirs = {}
    for json_path in raw_json_paths:
        with open(json_path) as f:
            changes = json.load(f)
        for change in changes:
            cid = change.get("id", "")  # "openstack%2Fnova~969380" 形式
            rev_hash = change.get("current_revision")
            if not rev_hash or not cid:
                continue
            files = change.get("revisions", {}).get(rev_hash, {}).get("files", {})
            dirs = set()
            for fpath in files.keys():
                parts = fpath.split("/")
                if len(parts) >= 2:
                    dirs.add("/".join(parts[:2]))
                else:
                    dirs.add("(root)")
            change_dirs[cid] = dirs

    # combined_raw.csv からレビュアー × change の対応
    df = pd.read_csv(data_path)

    reviewer_dirs = defaultdict(Counter)
    for _, row in df.iterrows():
        email = row["email"]
        cid = str(row["change_id"])
        dirs = change_dirs.get(cid)
        if dirs:
            for d in dirs:
                reviewer_dirs[email][d] += 1

    return reviewer_dirs


def compute_dir_coherence(
    reviewer_dirs: dict,
    cochange_counts: Counter,
) -> dict:
    """レビュアーの担当ディレクトリ間の coherence score を計算

    coherence = (担当ディレクトリ対のうち co-change 関係にある割合)
    高い = 関連性の高いディレクトリ群を担当
    低い = バラバラなディレクトリを担当
    """
    coherence_scores = {}
    for email, dir_counts in reviewer_dirs.items():
        dirs = list(dir_counts.keys())
        if len(dirs) < 2:
            coherence_scores[email] = 1.0  # 1ディレクトリのみ
            continue

        pairs = list(combinations(sorted(dirs), 2))
        n_pairs = len(pairs)
        n_cochange = sum(1 for d1, d2 in pairs if cochange_counts.get((d1, d2), 0) > 0)
        coherence_scores[email] = n_cochange / n_pairs if n_pairs > 0 else 0.0

    return coherence_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json", nargs="+", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="combined_raw.csv")
    parser.add_argument("--pair-predictions", type=Path, help="pair_predictions.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # co-change グラフ構築（01 と同じロジック）
    print("=== Building co-change graph ===")
    cochange_counts = Counter()
    for json_path in args.raw_json:
        with open(json_path) as f:
            changes = json.load(f)
        for change in changes:
            rev_hash = change.get("current_revision")
            if not rev_hash:
                continue
            files = change.get("revisions", {}).get(rev_hash, {}).get("files", {})
            dirs = set()
            for fpath in files.keys():
                parts = fpath.split("/")
                dirs.add("/".join(parts[:2]) if len(parts) >= 2 else "(root)")
            for d1, d2 in combinations(sorted(dirs), 2):
                cochange_counts[(d1, d2)] += 1

    # レビュアー × ディレクトリ マップ構築
    print("=== Building reviewer-directory map ===")
    reviewer_dirs = build_reviewer_dir_map(args.raw_json, args.data)
    print(f"  {len(reviewer_dirs)} reviewers mapped")

    # coherence score 計算
    print("=== Computing coherence scores ===")
    coherence = compute_dir_coherence(reviewer_dirs, cochange_counts)

    coherence_df = pd.DataFrame(
        [
            {
                "email": email,
                "coherence": score,
                "n_dirs": len(reviewer_dirs[email]),
            }
            for email, score in coherence.items()
        ]
    )
    coherence_df.to_csv(args.output_dir / "reviewer_coherence.csv", index=False)

    # 層別化して分析
    try:
        coherence_df["coherence_tier"] = pd.qcut(
            coherence_df["coherence"],
            q=3,
            labels=False,
            duplicates="drop",
        )
        tier_labels = {0: "low", 1: "mid", 2: "high"}
        n_tiers = coherence_df["coherence_tier"].nunique()
        if n_tiers == 2:
            tier_labels = {0: "low", 1: "high"}
        elif n_tiers == 1:
            tier_labels = {0: "all"}
        coherence_df["coherence_tier"] = coherence_df["coherence_tier"].map(tier_labels)
    except ValueError:
        # All same value
        coherence_df["coherence_tier"] = "all"

    print("\nCoherence distribution:")
    print(coherence_df["coherence"].describe())
    print(f"\nTier counts:")
    print(coherence_df["coherence_tier"].value_counts())

    # pair_predictions との突き合わせ
    if args.pair_predictions and args.pair_predictions.exists():
        print("\n=== Coherence vs Continuation ===")
        pairs = pd.read_csv(args.pair_predictions)
        pairs = pairs.merge(
            coherence_df[["email", "coherence", "coherence_tier"]],
            left_on="developer",
            right_on="email",
            how="left",
        )

        for tier, group in pairs.groupby("coherence_tier", observed=True):
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
                f"  coherence={tier}: n={n}, cont_rate={cont_rate:.3f}, "
                f"IRL_AUC={irl_str}, RF_AUC={rf_str}"
            )


if __name__ == "__main__":
    main()
