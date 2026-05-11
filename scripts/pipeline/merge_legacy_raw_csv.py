#!/usr/bin/env python3
"""
旧形式 data/{name}_raw.csv と新形式 data/raw_csv/openstack__{name}.csv をマージし、
新形式に一本化する。

ロジック:
    new (data/raw_csv/openstack__{name}.csv) を最新スナップショットとして基準にし、
    new に存在しない change_id を持つ legacy 行のみ追加する。
    (5/10 再収集時に abandon/削除された change を保持するため)

挙動:
    - 旧形式ファイルを data/legacy_backup/ に退避してから処理
    - 新形式ファイルを上書き
    - 旧形式ファイル data/{name}_raw.csv は backup へ移動

対象:
    data/{name}_raw.csv が存在する全 repo を自動検出
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

DATA = Path("data")
RAW_CSV = DATA / "raw_csv"
BACKUP = DATA / "legacy_backup"


def main():
    BACKUP.mkdir(exist_ok=True)
    legacy_files = sorted(DATA.glob("*_raw.csv"))
    # combined_raw*.csv は除外
    legacy_files = [f for f in legacy_files if not f.name.startswith("combined_")]

    if not legacy_files:
        print("legacy CSV (data/*_raw.csv) が見つかりません。終了。")
        return

    print(f"=== legacy CSV: {len(legacy_files)} 件 ===\n")
    total_added = 0
    for legacy in legacy_files:
        name = legacy.stem.replace("_raw", "")
        new = RAW_CSV / f"openstack__{name}.csv"

        if not new.exists():
            # new 側に対応が無い場合はリネームのみ
            target = RAW_CSV / f"openstack__{name}.csv"
            print(f"[{name}] new 側に無し → legacy をそのまま {target} へコピー")
            shutil.copy2(legacy, target)
            shutil.move(str(legacy), BACKUP / legacy.name)
            continue

        df_new = pd.read_csv(new)
        df_legacy = pd.read_csv(legacy)

        new_ids = set(df_new["change_id"])
        extra = df_legacy[~df_legacy["change_id"].isin(new_ids)]
        merged = pd.concat([df_new, extra], ignore_index=True)

        print(f"[{name}] new={len(df_new):>6d}  legacy_only_changes={extra['change_id'].nunique():>4d}  "
              f"added_rows={len(extra):>5d}  → merged={len(merged):>6d}")
        total_added += len(extra)

        merged.to_csv(new, index=False)
        shutil.move(str(legacy), BACKUP / legacy.name)

    # combined_raw.csv (旧 10 統合) も backup へ
    old_combined = DATA / "combined_raw.csv"
    if old_combined.exists():
        shutil.move(str(old_combined), BACKUP / old_combined.name)
        print(f"\ndata/combined_raw.csv も {BACKUP}/ に退避")

    print(f"\n=== 完了: 追加 {total_added:,} 行 ===")
    print(f"legacy ファイルは {BACKUP}/ に退避済み (不要なら手動で削除可)")


if __name__ == "__main__":
    main()
