"""
co-change グラフの構築 (main32 スコープ / Phase 2 用)

01_cochange_graph.py の main32 拡張版。違い:
  1. ディレクトリ抽出を path_features.py の file_to_dir(depth=2) と整合させる
     - 01_cochange は project_name 配下のみ 2 階層、他は 1 階層 → path_features と不整合
     - 本スクリプトは常に 2 階層、boilerplate dir は post-filter で除外
  2. coverage 特徴量用に **全 co-change ペア** を出力 (top 30 だけでなく)
  3. 旧 10-project 用と区別するため出力ファイル名を `_main32` サフィックスに

出力:
  results/hub_scores_main32.csv         (project, directory, hub_score, change_count)
  results/cochange_neighbors_main32.csv (project, directory, neighbor, weight)

使い方:
  uv run python experiments/dependency_analysis/01b_cochange_graph_main32.py \
      --raw-json data/raw_json/openstack__*.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd


# 01_cochange_graph.py から流用 (一階層目で除外する boilerplate dirs)
BOILERPLATE_TOP_DIRS = {
    "releasenotes", "doc", "(root)", "tools", "etc",
    "api-ref", "api-guide", "devstack", "playbooks", "roles",
    "zuul.d", ".zuul.d", "contrib", "rally-jobs", "specs", "bin",
}


def file_to_dir_2(fpath: str) -> str | None:
    """
    path_features.py の file_to_dir(depth=2) と整合させたディレクトリ抽出。
      - 1階層ファイル (README.md など): None
      - 2階層以上: 先頭 2 セグメント
      - 1階層目が boilerplate dir に該当: None
    """
    parts = [p for p in fpath.split("/") if p]
    if len(parts) <= 1:
        return None
    if parts[0] in BOILERPLATE_TOP_DIRS:
        return None
    return f"{parts[0]}/{parts[1]}"


def build_cochange_graphs(raw_json_paths: list[Path]) -> dict:
    results = {}
    for jp in raw_json_paths:
        project_full = jp.stem.replace("__", "/")  # "openstack__nova" -> "openstack/nova"
        print(f"[cochange] {project_full} ...", flush=True)
        with open(jp) as f:
            changes = json.load(f)

        pair_counts: Counter = Counter()
        neighbors: dict[str, set[str]] = defaultdict(set)
        dir_change_count: Counter = Counter()

        for change in changes:
            rev_hash = change.get("current_revision")
            if not rev_hash:
                continue
            files = (change.get("revisions") or {}).get(rev_hash, {}).get("files") or {}
            if not files:
                continue
            dirs = set()
            for fp in files.keys():
                if fp.startswith("/"):  # gerrit special entries "/COMMIT_MSG" など
                    continue
                d = file_to_dir_2(fp)
                if d is not None:
                    dirs.add(d)
            for d in dirs:
                dir_change_count[d] += 1
            for d1, d2 in combinations(sorted(dirs), 2):
                pair_counts[(d1, d2)] += 1
                neighbors[d1].add(d2)
                neighbors[d2].add(d1)

        hub_scores = {d: len(ns) for d, ns in neighbors.items()}
        results[project_full] = {
            "pair_counts": pair_counts,
            "neighbors": neighbors,
            "hub_scores": hub_scores,
            "dir_change_count": dir_change_count,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-json", nargs="+", type=Path, required=True,
        help="raw JSON files (32 main repos を想定)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    parser.add_argument(
        "--min-pair-weight", type=int, default=1,
        help="この weight 未満の co-change ペアは出力に含めない (default 1 = 全保存)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building co-change graphs for {len(args.raw_json)} repos ===")
    print(f"BOILERPLATE_TOP_DIRS excluded: {sorted(BOILERPLATE_TOP_DIRS)}")
    print(f"Directory granularity: 2-level (matches path_features.py file_to_dir(depth=2))")
    all_results = build_cochange_graphs(args.raw_json)

    # ── hub_scores_main32.csv ────────────────────────────────
    hub_rows = []
    for project, g in all_results.items():
        for d, score in sorted(g["hub_scores"].items(), key=lambda x: -x[1]):
            hub_rows.append({
                "project": project,
                "directory": d,
                "hub_score": score,
                "change_count": g["dir_change_count"][d],
            })
    hub_df = pd.DataFrame(hub_rows)
    hub_out = args.output_dir / "hub_scores_main32.csv"
    hub_df.to_csv(hub_out, index=False)
    print(f"[write] {hub_out}: {len(hub_df):,} rows")

    # ── cochange_neighbors_main32.csv (FULL pairs, weight >= min_pair_weight) ─
    # ペアの全方向版 (project, directory, neighbor, weight) で coverage 計算しやすく
    neigh_rows = []
    for project, g in all_results.items():
        for (d1, d2), w in g["pair_counts"].items():
            if w < args.min_pair_weight:
                continue
            neigh_rows.append({"project": project, "directory": d1, "neighbor": d2, "weight": w})
            neigh_rows.append({"project": project, "directory": d2, "neighbor": d1, "weight": w})
    neigh_df = pd.DataFrame(neigh_rows)
    neigh_out = args.output_dir / "cochange_neighbors_main32.csv"
    neigh_df.to_csv(neigh_out, index=False)
    print(f"[write] {neigh_out}: {len(neigh_df):,} rows ({len(neigh_df)//2:,} unique pairs)")

    # ── summary ──────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'project':35s} {'n_dirs':>8} {'n_pairs':>10} {'max_hub':>9}")
    for project, g in sorted(all_results.items()):
        n_dirs = len(g["hub_scores"])
        n_pairs = len(g["pair_counts"])
        max_hub = max(g["hub_scores"].values()) if g["hub_scores"] else 0
        print(f"{project:35s} {n_dirs:>8} {n_pairs:>10} {max_hub:>9}")


if __name__ == "__main__":
    main()
