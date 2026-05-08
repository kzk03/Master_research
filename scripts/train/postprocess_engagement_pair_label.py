#!/usr/bin/env python3
"""エンゲージメント軌跡 cache を **(dev, dir) engagement label** で書き換える。

入力 (`engagement_traj_0-3.pkl`) は per-dev 全活動 step を持つ既存 cache。
本スクリプトは step_labels と step_actions のみを書き換える。

書き換えロジック:
  step_label[i] = 1 if 任意 event が `[t_i + fw_start, t_i + fw_end)` 内に存在し、
                  かつその event の dirs が step_event_dirs[i] と overlap
                else 0
  step_action[i] = 0 if step_label[i] == 0
                   else cluster_id(step_event_dirs[i])  (1..K, K+1=other)

これにより step_action は B-19 と同じ「ラベル整列」設計になり、
未来 window 内に target dir で活動しない場合のみ action=0 (reject) になる。

使い方:
    uv run python scripts/train/postprocess_engagement_pair_label.py \\
        --input-cache outputs/mce_pilot_engagement/cache/engagement_traj_0-3.pkl \\
        --events data/combined_events.csv \\
        --dir-class-mapping outputs/dir_class_mapping_K15.json \\
        --future-window-start 0 --future-window-end 3 \\
        --output outputs/mce_pilot_engagement_v2/cache/engagement_pair_traj_0-3.pkl
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
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


def _normalize_dir(d: str, depth: int) -> str:
    """dir を class mapping の depth に合わせて切り詰める。

    - depth=1: 先頭階層のみ (例: "nova/compute" → "nova")
    - depth>=2: そのまま (path_features は元々 depth=2 で抽出されるので OK)
    """
    if not d:
        return ""
    if depth <= 1:
        return d.split("/", 1)[0]
    return d


def _to_class_id(
    ev_dirs: List[str],
    class_map: Dict[str, int],
    other_id: int,
    depth: int,
) -> int:
    if not ev_dirs:
        return other_id
    rep = _normalize_dir(sorted(ev_dirs)[0], depth)
    return int(class_map.get(rep, other_id))


def _parse_dirs_json(s):
    if not isinstance(s, str) or s == "[]":
        return []
    try:
        return [d for d in json.loads(s) if d and d != "."]
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=str, required=True)
    parser.add_argument("--events", type=str, required=True, help="combined_events.csv")
    parser.add_argument("--dir-class-mapping", type=str, required=True)
    parser.add_argument("--future-window-start", type=int, default=0)
    parser.add_argument("--future-window-end", type=int, default=3)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        logger.info("出力先が既存、スキップ: %s", out_path)
        return

    # dir class mapping
    with open(args.dir_class_mapping, "r", encoding="utf-8") as f:
        m = json.load(f)
    class_map: Dict[str, int] = {k: int(v) for k, v in m["classes"].items()}
    num_actions = int(m["num_actions"])
    K = int(m["K"])
    other_id = int(class_map["other"])
    mapping_depth = int(m.get("depth", 1))
    logger.info(
        "dir_class_mapping: K=%d, num_actions=%d, classes=%d, depth=%d",
        K, num_actions, len(class_map), mapping_depth,
    )

    # cache 読み込み
    logger.info("既存 cache を読み込み: %s", args.input_cache)
    with open(args.input_cache, "rb") as f:
        trajectories: List[Dict] = pickle.load(f)
    if not trajectories:
        logger.error("cache が空です")
        return
    sample = trajectories[0]
    if "step_event_dirs" not in sample:
        raise ValueError("cache に step_event_dirs が見つかりません。エンゲージメント軌跡を再生成してください。")

    # combined_events.csv 読み込み
    logger.info("loading events: %s", args.events)
    events_df = pd.read_csv(args.events)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], errors="coerce", format="mixed")
    events_df = events_df.dropna(subset=["timestamp"]).reset_index(drop=True)
    events_df["dirs_list"] = events_df["dirs"].map(_parse_dirs_json)
    logger.info("  %d events, %d unique reviewers", len(events_df), events_df["email"].nunique())

    # 各 dev の (time, dirs_set) リスト (時系列ソート)
    logger.info("building per-dev (time, dirs) index ...")
    dev_event_index: Dict[str, Tuple[List[pd.Timestamp], List[FrozenSet[str]]]] = {}
    for dev, sub in events_df.groupby("email"):
        sub_sorted = sub.sort_values("timestamp")
        ts = sub_sorted["timestamp"].tolist()
        dirs = [frozenset(d) for d in sub_sorted["dirs_list"].tolist()]
        dev_event_index[dev] = (ts, dirs)
    logger.info("  %d devs indexed", len(dev_event_index))

    # 後処理
    fw_start = pd.DateOffset(months=args.future_window_start)
    fw_end = pd.DateOffset(months=args.future_window_end)

    pos_count = 0
    total_count = 0
    n_changed_trajs = 0
    cls_cnt: Counter = Counter()

    for traj in trajectories:
        dev = traj["developer_info"].get("email") or traj["developer_info"].get("developer_id")
        ev_times = traj.get("step_context_dates", []) or []
        sed = traj.get("step_event_dirs", []) or []
        old_labels = traj.get("step_labels", []) or []

        ts_list, dirs_list = dev_event_index.get(dev, ([], []))

        new_labels: List[int] = []
        new_actions: List[int] = []
        for i, t_i in enumerate(ev_times):
            t_pd = pd.Timestamp(t_i)
            fs = t_pd + fw_start
            fe = t_pd + fw_end
            target_dirs: FrozenSet[str] = frozenset(sed[i] if i < len(sed) else [])

            if not target_dirs or not ts_list:
                new_labels.append(0)
                new_actions.append(0)
                continue

            lo = bisect.bisect_left(ts_list, fs)
            hi = bisect.bisect_left(ts_list, fe)
            # event 自身を除外: fw_start = 0 のとき lo は t_i 自身を指すので進める
            if args.future_window_start == 0:
                while lo < len(ts_list) and ts_list[lo] <= t_pd:
                    lo += 1

            label = 0
            for j in range(lo, hi):
                if dirs_list[j] and target_dirs.intersection(dirs_list[j]):
                    label = 1
                    break
            new_labels.append(label)

            if label == 0:
                new_actions.append(0)
            else:
                new_actions.append(
                    _to_class_id(list(target_dirs), class_map, other_id, mapping_depth)
                )

        if new_labels != old_labels[: len(new_labels)]:
            n_changed_trajs += 1

        traj["step_labels"] = new_labels
        traj["step_actions"] = new_actions
        traj["multi_class_action"] = True
        traj["num_actions"] = num_actions
        traj["dir_class_mapping_path"] = args.dir_class_mapping
        traj["future_window_label"] = True
        traj["future_window_months"] = (
            int(args.future_window_start),
            int(args.future_window_end),
        )
        traj["engagement_label"] = True
        traj["engagement_label_dir_specific"] = True

        pos_count += sum(int(bool(l)) for l in new_labels)
        total_count += len(new_labels)
        cls_cnt.update(new_actions)

    logger.info("step_labels 書き換え: %d/%d 軌跡で変更あり", n_changed_trajs, len(trajectories))
    if total_count > 0:
        logger.info(
            "(dev, dir) engagement label: 正例 %d/%d (%.2f%%)",
            pos_count, total_count, pos_count / total_count * 100,
        )
    seq_lens = [len(t["step_labels"]) for t in trajectories]
    if seq_lens:
        import statistics
        logger.info(
            "系列長: mean=%.1f median=%d p90=%d p99=%d max=%d",
            statistics.mean(seq_lens), int(statistics.median(seq_lens)),
            int(np.percentile(seq_lens, 90)), int(np.percentile(seq_lens, 99)),
            max(seq_lens),
        )
    logger.info("step_actions multiclass 分布:")
    n_total_act = sum(cls_cnt.values())
    for cid in sorted(cls_cnt):
        v = cls_cnt[cid]
        logger.info("  class %2d : %7d (%5.2f%%)", cid, v, v / n_total_act * 100)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)
    logger.info("保存完了: %s (%d 軌跡, num_actions=%d)", out_path, len(trajectories), num_actions)


if __name__ == "__main__":
    main()
