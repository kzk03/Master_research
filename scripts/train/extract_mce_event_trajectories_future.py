#!/usr/bin/env python3
"""
MCE-IRL **イベント単位 / 未来窓 step_labels** 用の軌跡抽出スクリプト

既存の (dev, dir) ペア軌跡 cache (extract_mce_event_trajectories.py が出力した二値版)
を読み込み、step_labels を **「event 時刻 t_i から先 N ヶ月以内に dev が target_dir で
accept したか」** に書き換えて新しい cache として保存する。

これは月次 MCE-IRL (train_mce_irl.py) の step_labels 定義をイベント時刻に適用したもの:

    monthly model:
        for each month_start in history_months:
            month_end = month_start + 1m
            future_start = month_end + future_window_start_months  # 例: 0
            future_end   = month_end + future_window_end_months    # 例: 3
            month_label = 1 if dev が target_dir で [future_start, future_end) に accept

    event model (このスクリプト):
        for each event_time t_i in step_context_dates:
            future_start = t_i + future_window_start_months
            future_end   = t_i + future_window_end_months
            step_labels[i] = 1 if dev が target_dir で [future_start, future_end) に accept

学習・評価指標 (3ヶ月先 accept) と完全に一致する step ラベルを各 event に与えることで、
イベント粒度のデータ密度 (~256 step/pair) を活かした学習を可能にする。

使い方:
    uv run python scripts/train/extract_mce_event_trajectories_future.py \
        --reviews data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --reuse-binary-cache outputs/mce_pilot_event/cache/event_traj_0-3.pkl \
        --future-window-start 0 --future-window-end 3 \
        --output outputs/mce_pilot_event_pair_future/cache/event_traj_0-3.pkl
"""

import argparse
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

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


def _load_review_requests(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "email" in df.columns and "reviewer_email" not in df.columns:
        df = df.rename(columns={"email": "reviewer_email"})
    if "timestamp" in df.columns and "request_time" not in df.columns:
        df = df.rename(columns={"timestamp": "request_time"})
    df["request_time"] = pd.to_datetime(df["request_time"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="MCE-IRL イベント単位 / 未来窓 step_labels 軌跡生成"
    )
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument(
        "--reuse-binary-cache",
        type=str,
        required=True,
        help="既存の (dev, dir) ペア event cache (.pkl)",
    )
    parser.add_argument("--future-window-start", type=int, default=0)
    parser.add_argument("--future-window-end", type=int, default=3)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        logger.info("キャッシュ済み、スキップ: %s", out_path)
        return

    # 既存 cache を読み込み
    logger.info("既存軌跡 cache を読み込み: %s", args.reuse_binary_cache)
    with open(args.reuse_binary_cache, "rb") as f:
        trajectories: List[Dict] = pickle.load(f)

    if not trajectories:
        logger.error("cache が空です")
        return
    sample = trajectories[0]
    if sample.get("per_dev", False):
        raise ValueError(
            "このスクリプトは (dev, dir) ペア cache を想定しています。"
            " per_dev=True の cache を渡されました。"
        )
    if "directory" not in sample or sample["directory"] is None:
        raise ValueError(
            "cache の軌跡に 'directory' が見つかりません。(dev, dir) ペア cache が必要です。"
        )

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

    # 高速化: (reviewer_email, target_dir) ごとの accept event 時刻リストを事前計算
    # 全件分は重いので、cache に登場する (dev, dir) のみ事前計算
    pair_set: Set = set()
    for traj in trajectories:
        dev = traj["developer_info"].get("email") or traj["developer_info"].get("developer_id")
        d = traj["directory"]
        pair_set.add((dev, d))
    logger.info("対象 (dev, dir) ペア数: %d", len(pair_set))

    # df を accept のみに絞り込み
    accept_df = df[df["label"] == 1][["reviewer_email", "request_time", "dirs"]].copy()
    logger.info("accept 件数 (label=1): %d", len(accept_df))

    # dev × dir × 時刻 のインデックスを構築
    # 各 dev について「accept した event 時刻のリスト」を dir 別に保持
    dev_dir_accept_times: Dict = {}  # {(dev, dir): sorted list of timestamps}
    for dev, target_dir in pair_set:
        sub = accept_df[
            (accept_df["reviewer_email"] == dev)
            & accept_df["dirs"].map(lambda ds: target_dir in ds if ds else False)
        ]
        ts = sorted(sub["request_time"].tolist())
        dev_dir_accept_times[(dev, target_dir)] = ts
    logger.info("dev × dir accept 時刻 index 構築完了")

    # step_labels を未来窓ラベルに書き換え
    fw_start = pd.DateOffset(months=args.future_window_start)
    fw_end = pd.DateOffset(months=args.future_window_end)

    pos_count = 0
    total_count = 0
    seq_lens: List[int] = []
    n_changed_trajs = 0
    cls_cnt: Counter = Counter()
    for traj in trajectories:
        dev = traj["developer_info"].get("email") or traj["developer_info"].get("developer_id")
        target_dir = traj["directory"]
        ev_times = traj.get("step_context_dates", []) or []
        old_labels = traj.get("step_labels", []) or []
        accept_times = dev_dir_accept_times.get((dev, target_dir), [])
        # accept_times は sorted list、bisect で高速判定
        import bisect

        new_labels: List[int] = []
        for t_i in ev_times:
            t_pd = pd.Timestamp(t_i)
            fs = t_pd + fw_start
            fe = t_pd + fw_end
            # bisect で範囲内にあるか
            lo = bisect.bisect_left(accept_times, fs)
            hi = bisect.bisect_left(accept_times, fe)
            new_labels.append(1 if hi - lo > 0 else 0)

        # step_labels が短い場合は (旧 labels が短かった場合は短い側に揃える)
        L = min(len(new_labels), len(old_labels)) if old_labels else len(new_labels)
        new_labels = new_labels[:L]
        if new_labels != old_labels[:L]:
            n_changed_trajs += 1
        traj["step_labels"] = new_labels
        traj["step_actions"] = [int(bool(l)) for l in new_labels]
        traj["future_window_label"] = True
        traj["future_window_months"] = (
            int(args.future_window_start),
            int(args.future_window_end),
        )

        pos_count += sum(new_labels)
        total_count += len(new_labels)
        seq_lens.append(len(new_labels))
        cls_cnt.update(new_labels)

    logger.info(
        "step_labels 書き換え完了: %d/%d 軌跡で変更あり",
        n_changed_trajs, len(trajectories),
    )
    if total_count > 0:
        logger.info(
            "未来窓ラベル分布: 正例 %d/%d (%.2f%%)",
            pos_count, total_count, pos_count / total_count * 100,
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
