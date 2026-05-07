#!/usr/bin/env python3
"""
エンゲージメント軌跡抽出スクリプト
==================================

per-dev × 全活動イベント × 未来窓エンゲージメントラベル × multiclass action 軌跡
を data/combined_events.csv から構築する。

設計:
  step (時間軸の点) = 全活動イベント (request, comment, vote, patchset, authored)
  step state       = レビュー依頼履歴 (combined_raw.csv) のスライディング 180d 窓
                     で extract_common_features (20 dim) + path_features (3 dim)
                     + event_features (4 dim) = 27 dim
  step label       = 「event 時刻 t から先 N ヶ月以内に dev が任意のイベントを発生」
                     (engagement, dev レベル)
  step action      = 0 (explicit reject, request かつ label=0)
                     または cluster_id(ev.dirs) (1..K)
                     または other_id (K+1) — 任意の positive engagement
  推論             = π(a = cid(target_dir) | s_last) を target_dir に対する
                     engagement 確率として返す

使い方:
    uv run python scripts/train/extract_engagement_event_trajectories.py \\
        --events data/combined_events.csv \\
        --reviews data/combined_raw.csv \\
        --raw-json data/raw_json/openstack__*.json \\
        --dir-class-mapping outputs/dir_class_mapping_K15.json \\
        --train-start 2019-01-01 --train-end 2022-01-01 \\
        --future-window-start 0 --future-window-end 3 \\
        --sliding-window-days 180 --max-events 256 \\
        --n-jobs -1 \\
        --output outputs/mce_pilot_engagement/cache/engagement_traj_0-3.pkl
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from review_predictor.IRL.features.path_features import (  # noqa: E402
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map,
    load_change_dir_map_multi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────


def _depth1(d: str) -> str:
    if not d:
        return ""
    return d.split("/", 1)[0]


def _step_event_dir_to_class(ev_dirs: List[str], class_map: Dict[str, int], other_id: int) -> int:
    if not ev_dirs:
        return other_id
    rep = _depth1(sorted(ev_dirs)[0])
    return int(class_map.get(rep, other_id))


def _load_dir_class_mapping(json_path: str) -> Tuple[Dict[str, int], int, int, int]:
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


def _parse_dirs(s: Optional[str]) -> List[str]:
    if not s or s == "[]":
        return []
    try:
        ds = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []
    return [d for d in ds if d and d != "."]


# ─────────────────────────────────────────────────────────────────────
# 1 dev 分のエンゲージメント軌跡を構築
# ─────────────────────────────────────────────────────────────────────


def _process_one_developer(
    dev: str,
    dev_events: pd.DataFrame,
    reviews_df: pd.DataFrame,
    path_extractor: PathFeatureExtractor,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    sliding_window_days: int,
    max_events: int,
    class_map: Dict[str, int],
    num_actions: int,
    other_id: int,
    all_event_times_sorted: List[pd.Timestamp],
) -> Optional[Dict[str, Any]]:
    """1 reviewer の engagement 軌跡を返す（step が 0 個なら None）。"""
    # train 期間内の event のみを step として採用
    in_train = dev_events[
        (dev_events["timestamp"] >= train_start) & (dev_events["timestamp"] < train_end)
    ].copy()
    if in_train.empty:
        return None

    # dirs を抽出した行のみ (path_features 計算可能)
    in_train["dirs_list"] = in_train["dirs"].map(_parse_dirs)
    in_train = in_train[in_train["dirs_list"].map(len) > 0].reset_index(drop=True)
    if in_train.empty:
        return None

    if len(in_train) > max_events:
        in_train = in_train.tail(max_events).reset_index(drop=True)

    sliding_delta = pd.Timedelta(days=sliding_window_days)
    fw_start_off = pd.DateOffset(months=future_window_start_months)
    fw_end_off = pd.DateOffset(months=future_window_end_months)

    # state 用の reviewer_history (combined_raw.csv の reviewer or owner として登場するレコード)
    rev_mask = (
        (reviews_df["reviewer_email"] == dev)
        | (reviews_df["owner_email"] == dev)
    )
    reviewer_history = reviews_df.loc[rev_mask, [
        "reviewer_email", "owner_email", "request_time", "label", "project",
        "first_response_time", "response_latency_days",
        "change_insertions", "change_deletions", "change_files_count",
        "is_cross_project", "dirs",
    ]].copy()
    reviewer_history = reviewer_history.sort_values("request_time")

    # 出力用バッファ
    step_labels: List[int] = []
    step_actions: List[int] = []
    step_event_dirs: List[List[str]] = []
    step_context_dates: List[pd.Timestamp] = []
    step_total_project_reviews: List[int] = []
    path_features_per_step: List[np.ndarray] = []
    event_features_list: List[Dict[str, float]] = []
    monthly_activity_histories: List[List[Dict[str, Any]]] = []

    prev_event_time: Optional[pd.Timestamp] = None

    for _, ev in in_train.iterrows():
        event_time: pd.Timestamp = ev["timestamp"]
        ev_dirs: List[str] = ev["dirs_list"]
        if not ev_dirs:
            continue
        ev_dirs_frozen = frozenset(ev_dirs)

        # ── 未来窓 engagement label (dev レベル) ──
        fs = event_time + fw_start_off
        fe = event_time + fw_end_off
        lo = bisect.bisect_left(all_event_times_sorted, fs)
        hi = bisect.bisect_left(all_event_times_sorted, fe)
        # event 自身が all_event_times_sorted に含まれるので、lo を進めるかどうかは
        # future_window_start_months の意味次第。fs=event_time+0m=event_time のとき、
        # bisect_left は event_time のインデックスを返すので「自分自身」が含まれる。
        # 自分は除外したいので、最低でも > event_time を要求する。
        # ただし同じ瞬間に複数 event がある場合の境界処理は複雑なので、
        # シンプルに「event 自身を 1 つ skip」だけ実装。
        if future_window_start_months == 0:
            # event 自身を skip するには event_time の位置を進める必要がある
            while lo < len(all_event_times_sorted) and all_event_times_sorted[lo] <= event_time:
                lo += 1
        step_label = 1 if hi - lo > 0 else 0
        step_labels.append(step_label)

        # ── multiclass action (B-1 に整合) ──
        is_request = ev["event_type"] == "request"
        if is_request and int(ev.get("label") or 0) == 0:
            action = 0  # explicit reject
        else:
            action = _step_event_dir_to_class(ev_dirs, class_map, other_id)
        step_actions.append(action)

        step_event_dirs.append(ev_dirs)
        step_context_dates.append(event_time)

        # ── activity_history (state 計算用)：レビュー依頼履歴の sliding window ──
        win_start = event_time - sliding_delta
        win_history = reviewer_history[
            (reviewer_history["request_time"] >= win_start)
            & (reviewer_history["request_time"] < event_time)
        ]
        activities: List[Dict[str, Any]] = []
        for _, row in win_history.iterrows():
            if row["reviewer_email"] == dev:
                activities.append(
                    {
                        "timestamp": row["request_time"],
                        "action_type": "review",
                        "project": row.get("project", "unknown"),
                        "project_id": row.get("project", "unknown"),
                        "request_time": row["request_time"],
                        "response_time": row.get("first_response_time"),
                        "accepted": int(row.get("label") or 0) == 1,
                        "owner_email": row.get("owner_email", ""),
                        "is_cross_project": bool(row.get("is_cross_project", False)),
                        "files_changed": int(row.get("change_files_count") or 0),
                        "lines_added": int(row.get("change_insertions") or 0),
                        "lines_deleted": int(row.get("change_deletions") or 0),
                    }
                )
            if row["owner_email"] == dev:
                activities.append(
                    {
                        "timestamp": row["request_time"],
                        "action_type": "authored",
                        "owner_email": dev,
                        "reviewer_email": row.get("reviewer_email", ""),
                        "files_changed": int(row.get("change_files_count") or 0),
                        "lines_added": int(row.get("change_insertions") or 0),
                        "lines_deleted": int(row.get("change_deletions") or 0),
                    }
                )
        monthly_activity_histories.append(activities)

        # total_project_reviews (期間頭からこの event 時刻までの全 project reviews)
        total_proj = int(((reviews_df["request_time"] >= train_start) & (reviews_df["request_time"] < event_time)).sum())
        step_total_project_reviews.append(total_proj)

        # ── path features (event の dirs で) ──
        pf = path_extractor.compute(dev, ev_dirs_frozen, event_time.to_pydatetime())
        path_features_per_step.append(pf)

        # ── event features (4 dim) ──
        if is_request:
            ins = ev.get("change_insertions") or 0
            dels = ev.get("change_deletions") or 0
            lines = (ins if pd.notna(ins) else 0) + (dels if pd.notna(dels) else 0)
            rt = ev.get("response_latency_days") or 0.0
            response_time = float(rt) if pd.notna(rt) else 0.0
            accepted = 1 if int(ev.get("label") or 0) == 1 else 0
        else:
            lines = 0
            response_time = 0.0
            accepted = 0  # 明示的な accept ではない
        time_since_prev = (
            (event_time - prev_event_time).total_seconds() / 86400.0
            if prev_event_time is not None
            else 30.0
        )
        event_features_list.append(
            {
                "event_lines_changed": max(0.0, min(lines / 2000.0, 1.0)),
                "event_response_time": max(0.0, min(response_time / 14.0, 1.0)),
                "event_accepted": float(accepted),
                "time_since_prev_event": max(0.0, min(time_since_prev / 180.0, 1.0)),
            }
        )
        prev_event_time = event_time

    if not step_labels:
        return None

    # developer_info
    dev_review_rows = reviewer_history[reviewer_history["reviewer_email"] == dev]
    n_changes_reviewed = int((dev_review_rows["label"] == 1).sum())
    n_requests = int(len(dev_review_rows))
    developer_info = {
        "developer_id": dev,
        "email": dev,
        "first_seen": (
            reviewer_history["request_time"].min()
            if not reviewer_history.empty
            else train_start
        ),
        "changes_reviewed": n_changes_reviewed,
        "requests_received": n_requests,
        "acceptance_rate": (
            n_changes_reviewed / n_requests if n_requests > 0 else 0.0
        ),
        "projects": (
            reviewer_history["project"].dropna().unique().tolist()
            if not reviewer_history.empty
            else []
        ),
    }

    trajectory: Dict[str, Any] = {
        "developer_info": developer_info,
        "directory": None,
        "activity_history": [],
        "monthly_activity_histories": monthly_activity_histories,
        "step_context_dates": step_context_dates,
        "step_total_project_reviews": step_total_project_reviews,
        "path_features_per_step": path_features_per_step,
        "event_features": event_features_list,
        "step_event_dirs": step_event_dirs,
        "context_date": train_end,
        "step_labels": step_labels,
        "step_actions": step_actions,
        "seq_len": len(step_labels),
        "reviewer": dev,
        "future_acceptance": bool(any(step_labels)),
        "sample_weight": 1.0,
        "per_dev": True,
        "multi_class_action": True,
        "num_actions": num_actions,
        "future_window_label": True,
        "future_window_months": (
            int(future_window_start_months),
            int(future_window_end_months),
        ),
        "engagement_label": True,
    }
    return trajectory


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=str, required=True, help="combined_events.csv")
    parser.add_argument("--reviews", type=str, required=True, help="combined_raw.csv")
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument("--dir-class-mapping", type=str, required=True)
    parser.add_argument("--train-start", type=str, default="2019-01-01")
    parser.add_argument("--train-end", type=str, default="2022-01-01")
    parser.add_argument("--future-window-start", type=int, default=0)
    parser.add_argument("--future-window-end", type=int, default=3)
    parser.add_argument("--sliding-window-days", type=int, default=180)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        logger.info("キャッシュ済み、スキップ: %s", out_path)
        return

    class_map, num_actions, K, other_id = _load_dir_class_mapping(args.dir_class_mapping)

    logger.info("loading events: %s", args.events)
    events_df = pd.read_csv(args.events)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], errors="coerce", format="mixed")
    events_df = events_df.dropna(subset=["timestamp"]).reset_index(drop=True)
    logger.info("  %d events, %d unique reviewers", len(events_df), events_df["email"].nunique())

    logger.info("loading reviews: %s", args.reviews)
    reviews_df = pd.read_csv(args.reviews)
    if "email" in reviews_df.columns and "reviewer_email" not in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={"email": "reviewer_email"})
    if "timestamp" in reviews_df.columns and "request_time" not in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={"timestamp": "request_time"})
    reviews_df["request_time"] = pd.to_datetime(reviews_df["request_time"], errors="coerce")
    reviews_df = reviews_df.dropna(subset=["request_time"]).reset_index(drop=True)

    # raw_json から dirs を埋める
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    reviews_df = attach_dirs_to_df(reviews_df, cdm, column="dirs")

    df_for_path = reviews_df.rename(columns={
        "reviewer_email": "email",
        "request_time": "timestamp",
    })
    path_extractor = PathFeatureExtractor(df_for_path, window_days=args.sliding_window_days)

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)

    # 各 dev の全 event 時刻リスト (engagement label 用 bisect index)
    logger.info("building per-dev event time index ...")
    dev_event_times: Dict[str, List[pd.Timestamp]] = {}
    for dev, sub in events_df.groupby("email"):
        dev_event_times[dev] = sorted(sub["timestamp"].tolist())
    logger.info("  %d devs indexed", len(dev_event_times))

    # 軌跡対象 dev = train 期間に何らか event を持つ dev
    in_train_devs = (
        events_df[(events_df["timestamp"] >= train_start) & (events_df["timestamp"] < train_end)]
        ["email"].unique()
    )
    logger.info("eligible devs: %d (train period activity あり)", len(in_train_devs))

    # 並列処理
    from joblib import Parallel, delayed  # noqa: E402

    events_by_dev: Dict[str, pd.DataFrame] = {
        dev: g for dev, g in events_df.groupby("email")
    }

    logger.info("extracting trajectories (n_jobs=%d) ...", args.n_jobs)
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_one_developer)(
            dev,
            events_by_dev[dev],
            reviews_df,
            path_extractor,
            train_start,
            train_end,
            args.future_window_start,
            args.future_window_end,
            args.sliding_window_days,
            args.max_events,
            class_map,
            num_actions,
            other_id,
            dev_event_times.get(dev, []),
        )
        for dev in in_train_devs
    )
    trajectories = [t for t in results if t is not None]
    logger.info("extracted %d trajectories (skipped %d)", len(trajectories), len(in_train_devs) - len(trajectories))

    if not trajectories:
        logger.error("軌跡が抽出できませんでした")
        return

    # 統計
    seq_lens = [len(t["step_labels"]) for t in trajectories]
    pos_steps = sum(int(bool(l)) for t in trajectories for l in t["step_labels"])
    total_steps = sum(len(t["step_labels"]) for t in trajectories)
    cls_cnt: Counter = Counter()
    for t in trajectories:
        cls_cnt.update(t["step_actions"])
    import statistics
    logger.info(
        "系列長: mean=%.1f median=%d p90=%d p99=%d max=%d (n_trajs=%d)",
        statistics.mean(seq_lens), int(statistics.median(seq_lens)),
        int(np.percentile(seq_lens, 90)), int(np.percentile(seq_lens, 99)),
        max(seq_lens), len(trajectories),
    )
    logger.info(
        "engagement label: 正例 %d/%d (%.2f%%)",
        pos_steps, total_steps, pos_steps / total_steps * 100 if total_steps else 0.0,
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
