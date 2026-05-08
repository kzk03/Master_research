#!/usr/bin/env python3
"""
MCE-IRL **per-dev / 未来窓 step_labels + マルチクラス accept action** 軌跡抽出
=====================================================================
B-1 (multiclass) と pair-future (未来窓 step_labels) を組み合わせた版。

per-dev cache の各軌跡を後処理:
  - **step_labels**: 「event i 時刻から先 N ヶ月以内に dev が (どこかで)
                     1 個以上 accept したか」(dev レベル継続)
  - **step_actions**: 各 event の dir を depth=1 親で class 化
                     (0=reject, 1..K=top-K accept, K+1=other_accept)

学習・推論:
  - 学習: 親 MCE-IRL の step_labels (アクション履歴) と step_actions (CE 損失) を別建て
        になっており、本実装では actions=step_actions (multiclass) を直接 CE で学習する。
        step_labels はモデル forward に直接効かない (state 計算には event 履歴のみ)。
        ただし、 cache には保存しておき、メタデータとして「未来窓ラベル」だと記録する。
  - 推論: π(a = cid(target_dir) | s_last) を「target_dir で先 N ヶ月内 accept」確率として返す。

使い方:
    uv run python scripts/train/extract_mce_event_trajectories_future_multiclass.py \\
        --reviews data/combined_raw.csv \\
        --raw-json data/raw_json/openstack__*.json \\
        --reuse-binary-cache outputs/mce_pilot_event_dev/cache/event_traj_0-3.pkl \\
        --dir-class-mapping outputs/dir_class_mapping_K15.json \\
        --future-window-start 0 --future-window-end 3 \\
        --output outputs/mce_pilot_event_dev_future_multiclass/cache/event_traj_0-3.pkl
"""

import argparse
import bisect
import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

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


def _load_dir_class_mapping(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    classes: Dict[str, int] = {k: int(v) for k, v in m["classes"].items()}
    num_actions = int(m["num_actions"])
    K = int(m["K"])
    other_id = int(classes["other"])
    logger.info(
        "dir_class_mapping: K=%d, num_actions=%d, classes=%d (depth=%d)",
        K, num_actions, len(classes), int(m.get("depth", 1)),
    )
    return classes, num_actions, K, other_id


def _depth1(d: str) -> str:
    if not d:
        return ""
    return d.split("/", 1)[0]


def _step_event_dir_to_class(ev_dirs, class_map, other_id) -> int:
    if not ev_dirs:
        return other_id
    rep = _depth1(sorted(ev_dirs)[0])
    return int(class_map.get(rep, other_id))


def main():
    parser = argparse.ArgumentParser(
        description="per-dev × 未来窓 step_labels + multiclass action 軌跡抽出"
    )
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument(
        "--reuse-binary-cache",
        type=str,
        required=True,
        help="既存の per-dev event cache (.pkl, per_dev=True 必須)",
    )
    parser.add_argument(
        "--dir-class-mapping",
        type=str,
        required=True,
        help="build_dir_class_mapping.py で生成した JSON",
    )
    parser.add_argument("--future-window-start", type=int, default=0)
    parser.add_argument("--future-window-end", type=int, default=3)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        logger.info("キャッシュ済み、スキップ: %s", out_path)
        return

    # mapping 読み込み
    class_map, num_actions, K, other_id = _load_dir_class_mapping(args.dir_class_mapping)

    # cache 読み込み
    logger.info("既存 per-dev cache を読み込み: %s", args.reuse_binary_cache)
    with open(args.reuse_binary_cache, "rb") as f:
        trajectories: List[Dict] = pickle.load(f)
    if not trajectories:
        logger.error("cache が空です")
        return
    sample = trajectories[0]
    if not sample.get("per_dev", False):
        raise ValueError(
            "このスクリプトは per-dev cache を想定しています。per_dev=False の cache が渡されました。"
        )
    if "step_event_dirs" not in sample:
        raise ValueError(
            "cache に step_event_dirs が見つかりません。"
            " 軌跡が古い形式の可能性があります。"
        )

    # df + dirs (depth=2) を構築
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

    # dev レベル accept 時刻リスト (target_dir に依存しない)
    accept_df = df[df["label"] == 1][["reviewer_email", "request_time"]].copy()
    logger.info("accept 件数 (label=1, 全 dir): %d", len(accept_df))

    # cache に登場する dev のみ事前計算
    dev_set = set()
    for traj in trajectories:
        dev = traj["developer_info"].get("email") or traj["developer_info"].get("developer_id")
        dev_set.add(dev)
    logger.info("対象 dev 数: %d", len(dev_set))

    dev_accept_times: Dict[str, list] = {}
    for dev in dev_set:
        sub = accept_df[accept_df["reviewer_email"] == dev]
        ts = sorted(sub["request_time"].tolist())
        dev_accept_times[dev] = ts
    logger.info("dev × accept 時刻 index 構築完了")

    # 後処理
    fw_start = pd.DateOffset(months=args.future_window_start)
    fw_end = pd.DateOffset(months=args.future_window_end)

    pos_count = 0
    total_count = 0
    seq_lens: List[int] = []
    n_changed_trajs = 0
    cls_cnt: Counter = Counter()

    for traj in trajectories:
        dev = traj["developer_info"].get("email") or traj["developer_info"].get("developer_id")
        ev_times = traj.get("step_context_dates", []) or []
        sed = traj.get("step_event_dirs", []) or []
        old_labels = traj.get("step_labels", []) or []
        accept_times = dev_accept_times.get(dev, [])

        # 未来窓 step_labels (dev レベル: dir 関係なく)
        new_labels: List[int] = []
        for t_i in ev_times:
            t_pd = pd.Timestamp(t_i)
            fs = t_pd + fw_start
            fe = t_pd + fw_end
            lo = bisect.bisect_left(accept_times, fs)
            hi = bisect.bisect_left(accept_times, fe)
            new_labels.append(1 if hi - lo > 0 else 0)

        # multiclass step_actions (B-1 と同じロジック)
        L = min(len(new_labels), len(sed)) if sed else len(new_labels)
        new_actions: List[int] = []
        for i in range(L):
            if new_labels[i] == 0:
                new_actions.append(0)  # reject (dev レベル不継続)
            else:
                ev_dirs = sed[i] if i < len(sed) else []
                new_actions.append(_step_event_dir_to_class(ev_dirs, class_map, other_id))
        # step_event_dirs が短い場合の補完
        for i in range(L, len(new_labels)):
            new_actions.append(0 if new_labels[i] == 0 else other_id)

        # 既存 labels のうち長さを揃える
        if old_labels:
            ll = min(len(new_labels), len(old_labels))
            new_labels = new_labels[:ll]
            new_actions = new_actions[:ll]

        if new_labels != old_labels[: len(new_labels)]:
            n_changed_trajs += 1

        traj["step_labels"] = new_labels
        traj["step_actions"] = new_actions
        traj["multi_class_action"] = True
        traj["num_actions"] = num_actions
        traj["dir_class_mapping_path"] = str(args.dir_class_mapping)
        traj["future_window_label"] = True
        traj["future_window_months"] = (
            int(args.future_window_start),
            int(args.future_window_end),
        )

        pos_count += sum(int(bool(l)) for l in new_labels)
        total_count += len(new_labels)
        seq_lens.append(len(new_labels))
        cls_cnt.update(new_actions)

    logger.info(
        "step_labels 書き換え完了: %d/%d 軌跡で変更あり",
        n_changed_trajs, len(trajectories),
    )
    if total_count > 0:
        logger.info(
            "未来窓ラベル分布 (dev レベル): 正例 %d/%d (%.2f%%)",
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
    logger.info("step_actions multiclass 分布:")
    n_total_steps = sum(cls_cnt.values())
    for cid in sorted(cls_cnt):
        v = cls_cnt[cid]
        logger.info("  class %2d : %7d (%5.2f%%)", cid, v, v / n_total_steps * 100)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    logger.info(
        "保存完了: %s (%d 軌跡, num_actions=%d)",
        out_path, len(trajectories), num_actions,
    )


if __name__ == "__main__":
    main()
