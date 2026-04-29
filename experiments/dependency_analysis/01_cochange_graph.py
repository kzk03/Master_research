"""
co-change グラフの構築と分析

raw_json から各 change に含まれるファイルパスを取得し、
同一 change 内で同時変更されたディレクトリ対を集計する。
hub ディレクトリ（多くのディレクトリと co-change する）を特定し、
pair_predictions.csv と突き合わせてレビュアー継続率との関係を分析する。
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import pandas as pd
import numpy as np


def build_cochange_graph(raw_json_paths: list[Path]) -> dict:
    """raw_json から co-change ディレクトリ対を集計"""
    # directory pair -> co-change count
    cochange_counts = Counter()
    # directory -> set of co-changed directories
    dir_neighbors = defaultdict(set)
    # directory -> total changes count
    dir_change_count = Counter()
    # project -> directory -> change count
    project_dir_counts = defaultdict(Counter)

    for json_path in raw_json_paths:
        project = json_path.stem.replace("__", "/")
        print(f"Processing {project}...")
        with open(json_path) as f:
            changes = json.load(f)

        for change in changes:
            rev_hash = change.get("current_revision")
            if not rev_hash:
                continue
            revisions = change.get("revisions", {})
            rev = revisions.get(rev_hash, {})
            files = rev.get("files", {})
            if not files:
                continue

            # ファイルパスからディレクトリを抽出（第1階層）
            dirs = set()
            for fpath in files.keys():
                parts = fpath.split("/")
                if len(parts) >= 2:
                    dirs.add(parts[0])
                else:
                    dirs.add("(root)")

            for d in dirs:
                dir_change_count[d] += 1
                project_dir_counts[project][d] += 1

            # ディレクトリ対の co-change を記録
            for d1, d2 in combinations(sorted(dirs), 2):
                cochange_counts[(d1, d2)] += 1
                dir_neighbors[d1].add(d2)
                dir_neighbors[d2].add(d1)

    # hub 度 = co-change する異なるディレクトリの数
    hub_scores = {d: len(neighbors) for d, neighbors in dir_neighbors.items()}

    return {
        "cochange_counts": cochange_counts,
        "dir_neighbors": dir_neighbors,
        "dir_change_count": dir_change_count,
        "hub_scores": hub_scores,
        "project_dir_counts": project_dir_counts,
    }


def analyze_hub_vs_continuation(
    hub_scores: dict,
    pair_predictions_path: Path,
) -> pd.DataFrame:
    """hub ディレクトリのレビュアー継続率を分析"""
    df = pd.read_csv(pair_predictions_path)

    # ディレクトリの第1階層を取得
    df["dir_top"] = df["directory"].apply(
        lambda x: x.split("/")[0] if "/" in str(x) else str(x)
    )

    # hub score を付与
    df["hub_score"] = df["dir_top"].map(hub_scores).fillna(0).astype(int)

    # hub score で層別化
    df["hub_tier"] = pd.qcut(
        df["hub_score"].clip(lower=0),
        q=3,
        labels=["low_hub", "mid_hub", "high_hub"],
        duplicates="drop",
    )

    # 層ごとの継続率・予測精度
    results = []
    for tier, group in df.groupby("hub_tier", observed=True):
        n = len(group)
        cont_rate = group["label"].mean()
        irl_auc = None
        rf_auc = None
        if group["label"].nunique() == 2:
            from sklearn.metrics import roc_auc_score

            irl_valid = group.dropna(subset=["irl_dir_prob"])
            rf_valid = group.dropna(subset=["rf_dir_prob"])
            if len(irl_valid) > 0 and irl_valid["label"].nunique() == 2:
                irl_auc = roc_auc_score(irl_valid["label"], irl_valid["irl_dir_prob"])
            if len(rf_valid) > 0 and rf_valid["label"].nunique() == 2:
                rf_auc = roc_auc_score(rf_valid["label"], rf_valid["rf_dir_prob"])

        results.append(
            {
                "hub_tier": tier,
                "n_pairs": n,
                "continuation_rate": cont_rate,
                "irl_dir_auc": irl_auc,
                "rf_dir_auc": rf_auc,
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-json",
        nargs="+",
        type=Path,
        required=True,
        help="raw JSON files",
    )
    parser.add_argument(
        "--pair-predictions",
        type=Path,
        help="pair_predictions.csv path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. co-change グラフ構築
    print("=== Building co-change graph ===")
    graph = build_cochange_graph(args.raw_json)

    # hub scores
    hub_df = pd.DataFrame(
        [
            {"directory": d, "hub_score": s, "change_count": graph["dir_change_count"][d]}
            for d, s in sorted(graph["hub_scores"].items(), key=lambda x: -x[1])
        ]
    )
    hub_df.to_csv(args.output_dir / "hub_scores.csv", index=False)
    print(f"\nTop 20 hub directories:")
    print(hub_df.head(20).to_string(index=False))

    # top co-change pairs
    top_pairs = graph["cochange_counts"].most_common(30)
    print(f"\nTop 30 co-change pairs:")
    for (d1, d2), count in top_pairs:
        print(f"  {d1} <-> {d2}: {count}")

    pairs_df = pd.DataFrame(
        [{"dir1": d1, "dir2": d2, "cochange_count": c} for (d1, d2), c in top_pairs]
    )
    pairs_df.to_csv(args.output_dir / "top_cochange_pairs.csv", index=False)

    # 2. pair_predictions との突き合わせ
    if args.pair_predictions and args.pair_predictions.exists():
        print("\n=== Hub score vs Continuation ===")
        result_df = analyze_hub_vs_continuation(
            graph["hub_scores"], args.pair_predictions
        )
        print(result_df.to_string(index=False))
        result_df.to_csv(args.output_dir / "hub_vs_continuation.csv", index=False)


if __name__ == "__main__":
    main()
