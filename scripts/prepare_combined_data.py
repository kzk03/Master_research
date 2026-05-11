#!/usr/bin/env python3
"""
service_teams_repos.csv をもとに、取得済み per-repo CSV を統合して
combined_raw.csv を生成する。

入力:
  data/service_teams_repos.csv      : 245 行（excluded_reason 列で除外管理）
  data/raw_csv/openstack__*.csv     : collect_service_teams.sh の出力

挙動:
  1. service_teams_repos.csv を読み、excluded_reason が空の行のみ対象
  2. 各 repo について data/raw_csv/openstack__{name}.csv を読む
  3. 全部を concat して data/combined_raw_{N}.csv に出力（N は実際に読めた repo 数）
  4. 統計（repo 別行数、tier 別行数、欠損 repo）を stdout に表示

旧形式 data/{name}_raw.csv は merge_legacy_raw_csv.py で raw_csv 配下に統合済み。
新規 legacy ファイルが現れた場合は同スクリプトを再実行すること。

出力ファイル名の自動命名:
  --output 未指定なら data/combined_raw_{N}.csv（旧 combined_raw.csv を上書きしない）
  既存ファイルがある場合は --overwrite を付けない限りエラー終了

使い方:
  uv run python scripts/prepare_combined_data.py             # data/combined_raw_231.csv 等
  uv run python scripts/prepare_combined_data.py --overwrite # 同名既存があれば上書き
  uv run python scripts/prepare_combined_data.py --output data/my.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos-csv", type=Path,
                        default=Path("data/service_teams_repos.csv"),
                        help="対象 repo リスト CSV（デフォルト: data/service_teams_repos.csv）")
    parser.add_argument("--raw-csv-dir", type=Path,
                        default=Path("data/raw_csv"),
                        help="per-repo CSV ディレクトリ")
    parser.add_argument("--output", type=Path, default=None,
                        help="統合先 CSV（未指定なら data/combined_raw_{N}.csv に自動命名）")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存ファイルがあっても上書きする")
    parser.add_argument("--include-excluded", action="store_true",
                        help="excluded_reason が付いた行も含める（デバッグ用）")
    args = parser.parse_args()

    # === 1. 対象 repo の読み込み ===
    if not args.repos_csv.exists():
        print(f"ERROR: {args.repos_csv} が存在しません", file=sys.stderr)
        sys.exit(1)

    targets = []  # [(team, tier, repo, excluded_reason), ...]
    with open(args.repos_csv) as f:
        for r in csv.DictReader(f):
            excl = r.get("excluded_reason", "")
            if excl and not args.include_excluded:
                continue
            targets.append((r["team"], r["tier"], r["repo"], excl))

    print(f"=== 対象 repo: {len(targets)} 件 (excluded_reason 空のみ) ===\n")

    # === 2. 各 repo の CSV を読む ===
    dfs = []
    stats = []           # (team, tier, repo, source, rows)
    missing = []         # (team, tier, repo)

    for team, tier, repo, _ in targets:
        safe_name = repo.replace("/", "__")             # openstack/nova -> openstack__nova
        path = args.raw_csv_dir / f"{safe_name}.csv"

        if not path.exists():
            missing.append((team, tier, repo))
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [WARN] {repo}: 読み込み失敗 {e}")
            missing.append((team, tier, repo))
            continue

        if "project" not in df.columns:
            df["project"] = repo

        dfs.append(df)
        stats.append((team, tier, repo, len(df)))

    # === 3. 統合 ===
    if not dfs:
        print("ERROR: 読み込めた CSV が 1 件もありません", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)

    # 出力パス決定: --output 未指定なら実 repo 数を埋め込む
    n_repos = len(dfs)
    if args.output is None:
        out_path = Path(f"data/combined_raw_{n_repos}.csv")
    else:
        out_path = args.output

    # 上書き保護
    if out_path.exists() and not args.overwrite:
        print(f"\nERROR: 出力先 {out_path} は既に存在します。", file=sys.stderr)
        print(f"  上書きする場合は --overwrite を指定してください。", file=sys.stderr)
        print(f"  別名で保存する場合は --output で指定してください。", file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    # === 4. 統計 ===
    print(f"=== 統合完了: {len(combined):,} 行, {n_repos} repos → {out_path} ===\n")

    # tier 別
    tier_rows = defaultdict(int)
    tier_repos = defaultdict(int)
    for team, tier, repo, n in stats:
        tier_rows[tier] += n
        tier_repos[tier] += 1
    print("--- tier 別 ---")
    for t in ["大", "中", "小", "極小", "非分類"]:
        if tier_repos[t]:
            print(f"  {t:4s}: {tier_repos[t]:3d} repos, {tier_rows[t]:>10,} rows")
    print()

    # 行数 Top 15
    stats_sorted = sorted(stats, key=lambda x: -x[3])
    print("--- 行数 Top 15 ---")
    for team, tier, repo, n in stats_sorted[:15]:
        print(f"  [{tier:4s}] {repo:50s} {n:>8,} rows")
    print()

    # 欠損 repo
    if missing:
        print(f"--- 取得 CSV が見つからない repo: {len(missing)} ---")
        for team, tier, repo in missing[:20]:
            print(f"  [{tier:4s}] {team:20s} {repo}")
        if len(missing) > 20:
            print(f"  ... 他 {len(missing)-20} 件")
    else:
        print("--- 欠損 repo: なし ---")


if __name__ == "__main__":
    main()
