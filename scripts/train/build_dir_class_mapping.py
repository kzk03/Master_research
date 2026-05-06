#!/usr/bin/env python3
"""
dir_class_mapping_K{N}.json を生成するスクリプト (Plan B-1 Phase 1.1)

イベント単位 IRL の action 空間をマルチクラス化するための、
"depth=1 親 dir → class_id" のマッピング JSON を生成する。

頻度上位 K dir に class_id 1..K を割り当て、それ以外は class_id K+1 ("other")。
reject (受諾しない) は推論時に class_id 0 として扱われる。

使い方:
    uv run python scripts/train/build_dir_class_mapping.py \
        --reviews data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --train-start 2019-01-01 --train-end 2022-01-01 \
        --top-k 15 \
        --output outputs/dir_class_mapping_K15.json

出力 JSON 形式:
    {
        "K": 15,
        "depth": 1,
        "classes": {"cinder": 1, "nova": 2, ..., "other": 16},
        "reject_class_id": 0,
        "num_actions": 17,
        "coverage": 0.93,
        "frequency": {"cinder": 8241, "nova": 4824, ...},
        "metadata": {...}
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from review_predictor.IRL.features.path_features import (  # noqa: E402
    attach_dirs_to_df,
    load_change_dir_map,
    load_change_dir_map_multi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_review_requests(csv_path: str) -> pd.DataFrame:
    """combined_raw.csv を email/timestamp 形式で読み込む。"""
    df = pd.read_csv(csv_path)
    if "email" in df.columns and "reviewer_email" not in df.columns:
        df = df.rename(columns={"email": "reviewer_email"})
    if "timestamp" in df.columns and "request_time" not in df.columns:
        df = df.rename(columns={"timestamp": "request_time"})
    df["request_time"] = pd.to_datetime(df["request_time"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="マルチクラス accept action 用 dir → class_id mapping 生成"
    )
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument(
        "--train-start",
        type=str,
        default="2019-01-01",
        help="頻度集計対象期間 開始 (この期間内のレビュー依頼のみ集計)",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2022-01-01",
        help="頻度集計対象期間 終了",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="頻度上位 K 個の dir に class_id を割り当てる",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="dir 集約 depth (Plan B-1 では depth=1 推奨)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力 JSON パス",
    )
    parser.add_argument(
        "--use-accepted-only",
        action="store_true",
        help="承諾されたイベントのみで頻度集計する (デフォルト: 全イベント)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_review_requests(args.reviews)
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    df = df[(df["request_time"] >= train_start) & (df["request_time"] < train_end)].copy()
    logger.info(
        "頻度集計対象: %s 〜 %s, %d 件 (accepted=%d)",
        train_start, train_end,
        len(df), int((df["label"] == 1).sum()),
    )

    if args.use_accepted_only:
        df = df[df["label"] == 1].copy()
        logger.info("--use-accepted-only により受諾件数 %d に絞り込み", len(df))

    # ディレクトリマッピング (depth=1 親で集計)
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=args.depth)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=args.depth)
    df = attach_dirs_to_df(df, cdm, column="dirs")

    # depth=1 親の頻度を集計（dirs[0] を代表 dir とする）
    counter: Counter = Counter()
    n_no_dir = 0
    for dirs in df["dirs"]:
        if not dirs:
            n_no_dir += 1
            continue
        # frozenset は順序が保証されないので、 sorted で安定化
        rep = sorted(dirs)[0]
        counter[rep] += 1

    total_with_dir = sum(counter.values())
    if total_with_dir == 0:
        logger.error("dir 情報を持つレビュー依頼が 0 件でした。raw_json を確認してください。")
        sys.exit(1)

    top = counter.most_common(args.top_k)
    cover_top = sum(v for _, v in top)
    coverage = cover_top / total_with_dir
    logger.info(
        "Top-%d coverage: %.2f%% (top %d / total_with_dir %d, no_dir=%d)",
        args.top_k, coverage * 100, cover_top, total_with_dir, n_no_dir,
    )

    # class_id 1..K を割り当て、最後に "other" = K+1
    classes: dict = {}
    for i, (d, _v) in enumerate(top):
        classes[d] = i + 1
    other_id = args.top_k + 1
    classes["other"] = other_id

    out = {
        "K": args.top_k,
        "depth": args.depth,
        "classes": classes,
        "reject_class_id": 0,
        "num_actions": other_id + 1,
        "coverage": coverage,
        "frequency": dict(counter.most_common()),
        "metadata": {
            "reviews_csv": args.reviews,
            "raw_json": args.raw_json,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "use_accepted_only": args.use_accepted_only,
            "n_total_with_dir": total_with_dir,
            "n_no_dir": n_no_dir,
            "n_unique_parents": len(counter),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("保存完了: %s (num_actions=%d)", output_path, out["num_actions"])

    # 簡易レポート
    print("\n=== dir → class_id mapping (top-{}) ===".format(args.top_k))
    print(f"  {'class_id':>8s}  {'dir':30s}  {'count':>8s}  {'%':>6s}")
    print(f"  {0:>8d}  {'(reject)':30s}  {'-':>8s}  {'-':>6s}")
    for d, cid in classes.items():
        if d == "other":
            continue
        c = counter[d]
        print(f"  {cid:>8d}  {d:30s}  {c:>8d}  {c/total_with_dir*100:>5.2f}%")
    other_count = total_with_dir - cover_top
    print(
        f"  {other_id:>8d}  {'(other)':30s}  {other_count:>8d}  "
        f"{other_count/total_with_dir*100:>5.2f}%"
    )
    print(f"\n  num_actions = {out['num_actions']}, coverage(top-{args.top_k}) = {coverage*100:.2f}%")


if __name__ == "__main__":
    main()
