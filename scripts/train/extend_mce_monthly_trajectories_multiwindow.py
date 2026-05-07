#!/usr/bin/env python3
"""
Plan A-3: 月次 (dev, dir) cache に 4 窓 (0-3, 3-6, 6-9, 9-12m) ラベルを後付け
=========================================================================
既存の月次 monthly_traj_0-3.pkl (extract_mce_trajectories.py の出力) は
step_labels = 「月 t 末から先 0-3 ヶ月以内に accept したか」 のみを持つ。

各 step_context_dates[i] (= 月 t 末) を基準に、3-6 / 6-9 / 9-12 m の
未来窓ラベルを再計算し、step_labels_per_window dict を軌跡に追加した
新 cache を生成する。これで 1 cache に 4 窓のラベルを並列保持できる。

ラベル計算ロジック (train_mce_irl.py extract_review_acceptance_trajectories と同一):
    future_start = step_context_date + future_window_start_months
    future_end   = step_context_date + future_window_end_months
    label = 1 if dev が target_dir で [future_start, future_end) に accept

使い方:
    uv run python scripts/train/extend_mce_monthly_trajectories_multiwindow.py \\
        --reviews data/combined_raw.csv \\
        --raw-json data/raw_json/openstack__*.json \\
        --reuse-binary-cache outputs/mce_pilot/cache/monthly_traj_0-3.pkl \\
        --output outputs/mce_pilot_multiwindow/cache/monthly_traj_multiwindow.pkl \\
        --windows 0-3 3-6 6-9 9-12
"""

import argparse
import bisect
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_review_requests(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "email" in df.columns and "reviewer_email" not in df.columns:
        df = df.rename(columns={"email": "reviewer_email"})
    if "timestamp" in df.columns and "request_time" not in df.columns:
        df = df.rename(columns={"timestamp": "request_time"})
    df["request_time"] = pd.to_datetime(df["request_time"])
    return df


def _parse_window(s: str) -> Tuple[int, int]:
    """'0-3' → (0, 3)"""
    a, b = s.split("-")
    return int(a), int(b)


def main():
    parser = argparse.ArgumentParser(
        description="Plan A-3: 月次 cache に 4 窓 step_labels を付与"
    )
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument(
        "--reuse-binary-cache",
        type=str,
        required=True,
        help="既存の月次 (dev, dir) cache (.pkl)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        nargs="+",
        default=["0-3", "3-6", "6-9", "9-12"],
        help="付与する未来窓のラベル (例: 0-3 3-6 6-9 9-12)",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        logger.info("出力済み、スキップ: %s", out_path)
        return

    # 既存 cache を読み込み
    logger.info("既存月次 cache を読み込み: %s", args.reuse_binary_cache)
    with open(args.reuse_binary_cache, "rb") as f:
        trajectories: List[Dict] = pickle.load(f)
    if not trajectories:
        logger.error("cache が空です")
        return
    sample = trajectories[0]
    if "directory" not in sample or sample["directory"] is None:
        raise ValueError(
            "cache の軌跡に 'directory' が必要です。(dev, dir) ペア cache のみ対応。"
        )
    if "step_context_dates" not in sample:
        raise ValueError("cache に step_context_dates が必要です。")
    logger.info("軌跡数 %d, seq_len 例 %d", len(trajectories), sample.get("seq_len", 0))

    # df + dirs マッピング (depth=2)
    from review_predictor.IRL.features.path_features import (  # noqa: E402
        attach_dirs_to_df,
        load_change_dir_map,
        load_change_dir_map_multi,
    )
    df = _load_review_requests(args.reviews)
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    df = attach_dirs_to_df(df, cdm, column="dirs")

    # cache に登場する (dev, dir) の集合
    pair_set = set()
    for traj in trajectories:
        dev = traj["developer_info"].get("email") or traj["developer_info"].get("developer_id")
        d = traj["directory"]
        pair_set.add((dev, d))
    logger.info("対象 (dev, dir) ペア数: %d", len(pair_set))

    # accept event のみ抽出
    accept_df = df[df["label"] == 1][["reviewer_email", "request_time", "dirs"]].copy()
    logger.info("accept 件数 (label=1): %d", len(accept_df))

    # (dev, dir) → sorted accept 時刻リスト を構築
    pair_accept_times: Dict[Tuple[str, str], list] = {}
    for dev, target_dir in pair_set:
        sub = accept_df[
            (accept_df["reviewer_email"] == dev)
            & accept_df["dirs"].map(lambda ds: target_dir in ds if ds else False)
        ]
        ts = sorted(sub["request_time"].tolist())
        pair_accept_times[(dev, target_dir)] = ts
    logger.info("(dev, dir) accept 時刻 index 構築完了")

    # 軌跡ごとに 4 窓ラベルを計算
    parsed_windows = [(w, _parse_window(w)) for w in args.windows]
    pos_per_win: Dict[str, int] = {w: 0 for w, _ in parsed_windows}
    total_steps = 0
    seq_lens: List[int] = []

    for traj in trajectories:
        dev = traj["developer_info"].get("email") or traj["developer_info"].get("developer_id")
        target_dir = traj["directory"]
        ev_times = traj.get("step_context_dates", []) or []
        accept_times = pair_accept_times.get((dev, target_dir), [])

        labels_per_window: Dict[str, List[int]] = {}
        for win_label, (s_m, e_m) in parsed_windows:
            offset_s = pd.DateOffset(months=s_m)
            offset_e = pd.DateOffset(months=e_m)
            labs: List[int] = []
            for t_i in ev_times:
                t_pd = pd.Timestamp(t_i)
                fs = t_pd + offset_s
                fe = t_pd + offset_e
                lo = bisect.bisect_left(accept_times, fs)
                hi = bisect.bisect_left(accept_times, fe)
                labs.append(1 if hi - lo > 0 else 0)
            labels_per_window[win_label] = labs
            pos_per_win[win_label] += sum(labs)

        traj["step_labels_per_window"] = labels_per_window
        # 既存 step_labels は維持 (0-3m 互換)
        total_steps += len(ev_times)
        seq_lens.append(len(ev_times))

    logger.info("=" * 60)
    logger.info("窓別 step ラベル統計 (total %d steps, %d 軌跡):", total_steps, len(trajectories))
    for w, _ in parsed_windows:
        p = pos_per_win[w]
        logger.info(
            "  窓 %5s: positive %d/%d (%.2f%%)",
            w, p, total_steps, p / max(total_steps, 1) * 100,
        )
    if seq_lens:
        import statistics
        logger.info(
            "系列長: mean=%.1f, median=%.1f, max=%d",
            statistics.mean(seq_lens),
            statistics.median(seq_lens),
            max(seq_lens),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    logger.info("保存完了: %s (%d 軌跡)", out_path, len(trajectories))


if __name__ == "__main__":
    main()
