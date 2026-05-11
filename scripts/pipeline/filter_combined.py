#!/usr/bin/env python3
"""
combined_raw_*.csv からプロジェクト集合をフィルタしてサブセットを切り出すスクリプト。

ユースケース:
    - 全 231 repos の combined_raw_231.csv から「32 main repos のみ」「tier=大 のみ」
      など任意のサブセットを抽出し、学習/評価対象を切り替えやすくする。

セレクタ (排他、いずれか 1 つを指定):
    --main                  data/service_teams_main_repos.csv の main_repo 列 (sunbeam 除く 32 件)
    --repos-csv FILE        任意の CSV を指定 (デフォルトで repo 列を読む。--repos-col で変更可)
    --tier 大,中            tier 列が一致する repo (service_teams_repos.csv を参照)
    --teams nova,neutron    team 列が一致する repo (service_teams_repos.csv を参照)
    --projects p1,p2        プロジェクト名を直接指定 (例: openstack/nova)

出力:
    --output PATH           フィルタ後の CSV (未指定なら data/combined_raw_<tag>.csv)
    .raw_json_list.txt      対応する raw_json パスのリスト (train スクリプトの --raw-json に流せる)

例:
    uv run python scripts/pipeline/filter_combined.py --main
    uv run python scripts/pipeline/filter_combined.py --tier 大
    uv run python scripts/pipeline/filter_combined.py --projects openstack/nova,openstack/neutron \\
        --output data/combined_raw_pair.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
RAW_JSON_DIR = DATA_DIR / "raw_json"


def load_projects(args) -> tuple[set[str], str]:
    """セレクタからプロジェクト集合と出力タグを返す。"""
    sources = [args.main, args.repos_csv, args.tier, args.teams, args.projects]
    if sum(bool(s) for s in sources) != 1:
        sys.exit("ERROR: --main / --repos-csv / --tier / --teams / --projects のいずれか 1 つを指定してください")

    if args.main:
        df = pd.read_csv(DATA_DIR / "service_teams_main_repos.csv").dropna(subset=["main_repo"])
        return set(df["main_repo"]), f"main{len(df)}"

    if args.repos_csv:
        df = pd.read_csv(args.repos_csv).dropna(subset=[args.repos_col])
        tag = Path(args.repos_csv).stem
        return set(df[args.repos_col]), tag

    if args.tier:
        tiers = [t.strip() for t in args.tier.split(",")]
        df = pd.read_csv(DATA_DIR / "service_teams_repos.csv")
        df = df[df["tier"].isin(tiers)].dropna(subset=["repo"])
        return set(df["repo"]), "tier_" + "_".join(tiers)

    if args.teams:
        teams = [t.strip() for t in args.teams.split(",")]
        df = pd.read_csv(DATA_DIR / "service_teams_repos.csv")
        df = df[df["team"].isin(teams)].dropna(subset=["repo"])
        return set(df["repo"]), "teams_" + "_".join(teams)

    projects = [p.strip() for p in args.projects.split(",")]
    return set(projects), f"custom{len(projects)}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DATA_DIR / "combined_raw_231.csv",
                   help="入力 combined CSV (default: data/combined_raw_231.csv)")
    p.add_argument("--output", type=Path, default=None,
                   help="出力 CSV (未指定なら data/combined_raw_<tag>.csv)")
    p.add_argument("--overwrite", action="store_true", help="既存ファイルを上書き")

    sel = p.add_argument_group("selector (排他)")
    sel.add_argument("--main", action="store_true", help="32 main repos を選択")
    sel.add_argument("--repos-csv", type=Path, help="任意の CSV から repo 列を読む")
    sel.add_argument("--repos-col", default="repo", help="--repos-csv の列名 (default: repo)")
    sel.add_argument("--tier", type=str, help="tier フィルタ (カンマ区切り、例: 大,中)")
    sel.add_argument("--teams", type=str, help="team フィルタ (カンマ区切り)")
    sel.add_argument("--projects", type=str, help="プロジェクト名を直接指定 (カンマ区切り)")

    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: {args.input} が存在しません")

    projects, tag = load_projects(args)
    print(f"=== 選択プロジェクト: {len(projects)} 件 (tag={tag}) ===")

    df = pd.read_csv(args.input)
    before = len(df)
    matched_in_csv = set(df["project"].unique()) & projects
    missing_in_csv = projects - set(df["project"].unique())
    df = df[df["project"].isin(projects)].reset_index(drop=True)
    print(f"  入力 {before:,} 行 → 出力 {len(df):,} 行 (マッチ {len(matched_in_csv)} repos)")
    if missing_in_csv:
        print(f"  WARN: CSV に存在しない repo: {len(missing_in_csv)} 件")
        for r in sorted(missing_in_csv)[:10]:
            print(f"    - {r}")
        if len(missing_in_csv) > 10:
            print(f"    ... 他 {len(missing_in_csv) - 10} 件")

    out_path = args.output or DATA_DIR / f"combined_raw_{tag}.csv"
    if out_path.exists() and not args.overwrite:
        sys.exit(f"ERROR: {out_path} は既に存在します (--overwrite で上書き)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    # 対応する raw_json リストを副産物として出力
    raw_json_paths = []
    missing_json = []
    for proj in sorted(matched_in_csv):
        jpath = RAW_JSON_DIR / (proj.replace("/", "__") + ".json")
        if jpath.exists():
            raw_json_paths.append(str(jpath))
        else:
            missing_json.append(proj)

    list_path = out_path.with_suffix(".raw_json_list.txt")
    list_path.write_text(" ".join(raw_json_paths) + "\n")
    print(f"  → {list_path} ({len(raw_json_paths)} JSON paths)")
    if missing_json:
        print(f"  WARN: raw_json が無い repo: {len(missing_json)} 件")
        for r in missing_json[:10]:
            print(f"    - {r}")

    # プロジェクト別の行数 (Top 10)
    print("\n--- 行数 Top 10 ---")
    for proj, n in df["project"].value_counts().head(10).items():
        print(f"  {proj:55s} {n:>8,}")


if __name__ == "__main__":
    main()
