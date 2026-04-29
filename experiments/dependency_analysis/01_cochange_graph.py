"""
co-change グラフの構築と分析（プロジェクト別）

raw_json から各 change に含まれるファイルパスを取得し、
同一 change 内で同時変更されたディレクトリ対を集計する。

D案: プロジェクト別 × boilerplate除外 × 本体コード2階層
- releasenotes, doc, (root), tools 等を除外
- プロジェクト本体コード（e.g. nova/）配下は2階層目で分析
- プロジェクトごとに独立した co-change グラフを構築
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import pandas as pd
import numpy as np


# boilerplate ディレクトリ（全プロジェクト共通で co-change を汚染するもの）
BOILERPLATE_DIRS = {
    "releasenotes",
    "doc",
    "(root)",
    "tools",
    "etc",
    "api-ref",
    "api-guide",
    "devstack",
    "playbooks",
    "roles",
    "zuul.d",
    ".zuul.d",
    "contrib",
    "rally-jobs",
    "specs",
    "bin",
}


def extract_directory(fpath: str, project_name: str) -> str | None:
    """ファイルパスからディレクトリを抽出

    - boilerplate は除外 (None を返す)
    - プロジェクト本体コード（project_name/ 配下）は2階層目
      e.g. nova/compute/api.py → "nova/compute"
    - それ以外の第1階層ディレクトリはそのまま
    """
    parts = fpath.split("/")

    if len(parts) < 2:
        return None  # (root) ファイルは除外

    top = parts[0]

    if top in BOILERPLATE_DIRS:
        return None

    # プロジェクト本体コード: 2階層目まで使う
    if top == project_name and len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"

    # 1階層のみ (e.g. "setup.cfg" 的なものは既に除外済み)
    return top


def build_cochange_graph_per_project(raw_json_paths: list[Path]) -> dict:
    """プロジェクト別に co-change グラフを構築"""
    results = {}

    for json_path in raw_json_paths:
        project_full = json_path.stem.replace("__", "/")  # "openstack/nova"
        project_name = project_full.split("/")[-1]  # "nova"
        print(f"Processing {project_full}...")

        with open(json_path) as f:
            changes = json.load(f)

        cochange_counts = Counter()
        dir_neighbors = defaultdict(set)
        dir_change_count = Counter()

        for change in changes:
            rev_hash = change.get("current_revision")
            if not rev_hash:
                continue
            revisions = change.get("revisions", {})
            rev = revisions.get(rev_hash, {})
            files = rev.get("files", {})
            if not files:
                continue

            dirs = set()
            for fpath in files.keys():
                d = extract_directory(fpath, project_name)
                if d is not None:
                    dirs.add(d)

            for d in dirs:
                dir_change_count[d] += 1

            for d1, d2 in combinations(sorted(dirs), 2):
                cochange_counts[(d1, d2)] += 1
                dir_neighbors[d1].add(d2)
                dir_neighbors[d2].add(d1)

        hub_scores = {d: len(neighbors) for d, neighbors in dir_neighbors.items()}

        results[project_full] = {
            "cochange_counts": cochange_counts,
            "dir_neighbors": dir_neighbors,
            "dir_change_count": dir_change_count,
            "hub_scores": hub_scores,
        }

    return results


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
        "--output-dir",
        type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building per-project co-change graphs (D plan) ===")
    print(f"Excluded dirs: {sorted(BOILERPLATE_DIRS)}")
    all_results = build_cochange_graph_per_project(args.raw_json)

    # プロジェクト別 hub scores を1つの CSV にまとめる
    hub_rows = []
    for project, graph in all_results.items():
        for d, score in sorted(graph["hub_scores"].items(), key=lambda x: -x[1]):
            hub_rows.append({
                "project": project,
                "directory": d,
                "hub_score": score,
                "change_count": graph["dir_change_count"][d],
            })
    hub_df = pd.DataFrame(hub_rows)
    hub_df.to_csv(args.output_dir / "hub_scores.csv", index=False)

    # プロジェクト別 top co-change pairs
    pair_rows = []
    for project, graph in all_results.items():
        for (d1, d2), count in graph["cochange_counts"].most_common(30):
            pair_rows.append({
                "project": project,
                "dir1": d1,
                "dir2": d2,
                "cochange_count": count,
            })
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(args.output_dir / "top_cochange_pairs.csv", index=False)

    # サマリ表示
    for project, graph in sorted(all_results.items()):
        n_dirs = len(graph["hub_scores"])
        n_pairs = len(graph["cochange_counts"])
        print(f"\n--- {project} ---")
        print(f"  {n_dirs} directories, {n_pairs} co-change pairs")

        top_hubs = sorted(graph["hub_scores"].items(), key=lambda x: -x[1])[:10]
        print(f"  Top 10 hub dirs:")
        for d, s in top_hubs:
            print(f"    {d}: hub_score={s}, changes={graph['dir_change_count'][d]}")

        top_pairs = graph["cochange_counts"].most_common(5)
        print(f"  Top 5 co-change pairs:")
        for (d1, d2), c in top_pairs:
            print(f"    {d1} <-> {d2}: {c}")


if __name__ == "__main__":
    main()
