#!/usr/bin/env python3
"""
MCE-IRL **イベント単位** 用の軌跡抽出スクリプト

extract_mce_trajectories.py の月次集約版に対し、こちらは 1 ステップ = 1 レビュー
依頼イベントとして軌跡を構築する。具体的な抽出ロジックは既存の
``train_model_event.extract_event_level_trajectories`` をそのまま流用する。
出力 pickle には MCE-IRL 用に二値 action 列 (step_actions) を付与する。

state は 20 次元の累積特徴量 + 3 次元の path 特徴量 + 4 次元の event 特徴量 =
**27 次元** になる。MCEIRLSystem は state_dim を引数で受けるため、軌跡側で
event_features フィールドを保持しているだけで自動的に 27 dim 化される。

使い方:
    uv run python scripts/train/extract_mce_event_trajectories.py \
        --reviews data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --train-start 2019-01-01 --train-end 2022-01-01 \
        --future-window-start 0 --future-window-end 3 \
        --max-events 256 --sliding-window-days 180 \
        --n-jobs -1 \
        --output outputs/mce_event_irl_trajectory_cache/mce_event_traj_0-3.pkl
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
    parser = argparse.ArgumentParser(
        description="MCE-IRL 用 イベント単位 軌跡抽出 & キャッシュ保存"
    )
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument("--train-start", type=str, default="2019-01-01")
    parser.add_argument("--train-end", type=str, default="2022-01-01")
    parser.add_argument("--future-window-start", type=int, required=True)
    parser.add_argument("--future-window-end", type=int, required=True)
    parser.add_argument(
        "--sliding-window-days",
        type=int,
        default=180,
        help="特徴量計算用のスライディングウィンドウ (日数)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=256,
        help="軌跡 (= 1 ペア) あたりの最大イベント数。超えると直近を保持して truncate",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="joblib 並列数 (-1=全コア)",
    )
    parser.add_argument(
        "--per-dev",
        action="store_true",
        help="軌跡を (dev) 単位で構築する（全 dir 横断、1 reviewer = 1 軌跡）。"
             "デフォルトは (dev, dir) ペア単位。",
    )
    parser.add_argument("--output", type=str, required=True, help="出力 .pkl パス")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        logger.info(f"キャッシュ済み、スキップ: {output_path}")
        return

    # 既存実装からイベント単位抽出関数とローダを借りる
    from train_model_event import (
        extract_event_level_trajectories,
        extract_event_level_trajectories_dev_only,
        load_review_requests,
    )
    from review_predictor.IRL.features.path_features import (
        PathFeatureExtractor,
        attach_dirs_to_df,
        load_change_dir_map,
        load_change_dir_map_multi,
    )

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

    mode_label = "per-dev (全 dir 横断)" if args.per_dev else "(dev, dir) ペア"
    logger.info(
        f"軌跡抽出開始 [event-level, mode={mode_label}, "
        f"sliding={args.sliding_window_days}d, "
        f"max_events={args.max_events}, n_jobs={args.n_jobs}]"
    )
    if args.per_dev:
        trajectories = extract_event_level_trajectories_dev_only(
            df,
            train_start=train_start,
            train_end=train_end,
            path_extractor=path_extractor,
            future_window_start_months=args.future_window_start,
            future_window_end_months=args.future_window_end,
            sliding_window_days=args.sliding_window_days,
            max_events=args.max_events,
            n_jobs=args.n_jobs,
        )
    else:
        trajectories = extract_event_level_trajectories(
            df,
            train_start=train_start,
            train_end=train_end,
            path_extractor=path_extractor,
            future_window_start_months=args.future_window_start,
            future_window_end_months=args.future_window_end,
            sliding_window_days=args.sliding_window_days,
            max_events=args.max_events,
            n_jobs=args.n_jobs,
        )

    if not trajectories:
        logger.error("軌跡が抽出できませんでした")
        return

    # MCE-IRL 用に step_actions (二値 action 列) を付与
    n_pos_steps = 0
    n_total_steps = 0
    seq_lens = []
    for traj in trajectories:
        labels = traj.get("step_labels", []) or []
        actions = [int(bool(l)) for l in labels]
        traj["step_actions"] = actions
        n_pos_steps += sum(actions)
        n_total_steps += len(actions)
        seq_lens.append(len(actions))
    if n_total_steps:
        logger.info(
            f"二値 action ステップ統計: 正例 {n_pos_steps}/{n_total_steps} "
            f"({n_pos_steps / n_total_steps * 100:.1f}%)"
        )
    if seq_lens:
        import statistics
        logger.info(
            "系列長: mean=%.1f, median=%.1f, max=%d",
            statistics.mean(seq_lens),
            statistics.median(seq_lens),
            max(seq_lens),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    logger.info(
        f"保存完了 [MCE-IRL event-level, mode={mode_label}]: "
        f"{output_path} ({len(trajectories)} 軌跡)"
    )


if __name__ == "__main__":
    main()
