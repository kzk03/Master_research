#!/usr/bin/env python3
"""
軌跡抽出のみを行い、pickle でキャッシュする。

使い方:
    uv run python scripts/train/extract_trajectories.py \
        --reviews data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --train-start 2019-01-01 --train-end 2022-01-01 \
        --future-window-start 0 --future-window-end 3 \
        --output trajectory_cache/traj_0-3.pkl

    # Multi-task
    uv run python scripts/train/extract_trajectories.py \
        --reviews data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --train-start 2019-01-01 --train-end 2022-01-01 \
        --future-window-start 0 --future-window-end 3 \
        --multitask \
        --output trajectory_cache/traj_mt_0-3.pkl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts" / "train"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="軌跡抽出 & キャッシュ保存")
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument("--train-start", type=str, default="2019-01-01")
    parser.add_argument("--train-end", type=str, default="2022-01-01")
    parser.add_argument("--future-window-start", type=int, required=True)
    parser.add_argument("--future-window-end", type=int, required=True)
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--output", type=str, required=True, help="出力 .pkl パス")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        logger.info(f"キャッシュ済み、スキップ: {output_path}")
        return

    from train_model import (
        extract_directory_level_trajectories,
        load_review_requests,
    )
    from review_predictor.IRL.features.path_features import (
        PathFeatureExtractor,
        attach_dirs_to_df,
        load_change_dir_map,
        load_change_dir_map_multi,
    )

    # データ読み込み
    df = load_review_requests(args.reviews)
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)

    # ディレクトリマッピング
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    df = attach_dirs_to_df(df, cdm, column="dirs")

    df_for_path = df.rename(columns={
        "reviewer_email": "email",
        "request_time": "timestamp",
    })
    path_extractor = PathFeatureExtractor(df_for_path, window_days=180)

    # 軌跡抽出
    logger.info(f"軌跡抽出開始 (multitask={args.multitask})")
    trajectories = extract_directory_level_trajectories(
        df,
        train_start=train_start,
        train_end=train_end,
        path_extractor=path_extractor,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        multitask=args.multitask,
    )

    if not trajectories:
        logger.error("軌跡が抽出できませんでした")
        return

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    logger.info(f"保存完了: {output_path} ({len(trajectories)} 軌跡)")


if __name__ == "__main__":
    main()
