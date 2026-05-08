#!/usr/bin/env python3
"""
MCE-IRL **イベント単位 / マルチクラス accept action** 用の軌跡抽出スクリプト
(Plan B-1 Phase 1.2)

extract_mce_event_trajectories.py のロジック (`extract_event_level_trajectories(_dev_only)`)
をそのまま流用しつつ、step_actions をマルチクラス化する:

    新 step_actions の値域:
        0       = reject (step_label == 0)
        1..K    = accept で代表 dir が dir_class_mapping に含まれる場合の class_id
        K+1     = accept だが代表 dir が "other"

軌跡 dict には次のフィールドを追加する:
    - multi_class_action : True
    - num_actions        : K+2 (= mapping["num_actions"])
    - dir_class_mapping_path : 入力 JSON のパス (記録用)

既存軌跡 cache とは別ファイルとして保存し、二値版コードへの影響はない。

使い方:
    uv run python scripts/train/extract_mce_event_trajectories_multiclass.py \
        --reviews data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --train-start 2019-01-01 --train-end 2022-01-01 \
        --future-window-start 0 --future-window-end 3 \
        --max-events 256 --sliding-window-days 180 \
        --per-dev \
        --dir-class-mapping outputs/dir_class_mapping_K15.json \
        --n-jobs -1 \
        --output outputs/mce_pilot_event_dev_multiclass/cache/event_traj_0-3.pkl
"""

import argparse
import json
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
SCRIPTS = ROOT / "scripts" / "train"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_dir_class_mapping(json_path: str) -> Tuple[Dict[str, int], int, int, int]:
    """dir_class_mapping JSON を読み込み (class_map, num_actions, K, other_id) を返す。"""
    with open(json_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    classes: Dict[str, int] = m["classes"]
    num_actions: int = int(m["num_actions"])
    K: int = int(m["K"])
    other_id: int = int(classes["other"])
    logger.info(
        "dir_class_mapping を読み込み: K=%d, num_actions=%d, classes=%d (depth=%d)",
        K, num_actions, len(classes), int(m.get("depth", 1)),
    )
    return classes, num_actions, K, other_id


def _depth1(d: str) -> str:
    """depth 不問の dir 文字列を depth=1 親に正規化する。"""
    if not d:
        return ""
    return d.split("/", 1)[0]


def _step_event_dir_to_class(
    ev_dirs: List[str],
    class_map: Dict[str, int],
    other_id: int,
) -> int:
    """1 イベント (dir 群) → class_id へ変換 (受諾を前提とした選択)。"""
    if not ev_dirs:
        return other_id
    # 軌跡には depth=2 の dir が入っているので depth=1 に丸めてから検索。
    # 安定性のため sorted の先頭を代表に取る。
    rep = _depth1(sorted(ev_dirs)[0])
    return int(class_map.get(rep, other_id))


def main():
    parser = argparse.ArgumentParser(
        description="MCE-IRL マルチクラス accept action 用 イベント単位 軌跡抽出"
    )
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument("--train-start", type=str, default="2019-01-01")
    parser.add_argument("--train-end", type=str, default="2022-01-01")
    parser.add_argument("--future-window-start", type=int, required=True)
    parser.add_argument("--future-window-end", type=int, required=True)
    parser.add_argument("--sliding-window-days", type=int, default=180)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--per-dev",
        action="store_true",
        help="(dev) 単位の軌跡を構築 (デフォルトは (dev, dir) ペア単位)",
    )
    parser.add_argument(
        "--dir-class-mapping",
        type=str,
        required=True,
        help="build_dir_class_mapping.py で生成した JSON ファイル",
    )
    parser.add_argument(
        "--reuse-binary-cache",
        type=str,
        default=None,
        help="既存の二値 step_actions 軌跡 cache (.pkl) を再利用してマルチクラス化のみ行う",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        logger.info("キャッシュ済み、スキップ: %s", output_path)
        return

    class_map, num_actions, K, other_id = _load_dir_class_mapping(args.dir_class_mapping)

    # 1) 既存の二値 cache を再利用するパス
    if args.reuse_binary_cache and Path(args.reuse_binary_cache).exists():
        logger.info("既存軌跡 cache を再利用: %s", args.reuse_binary_cache)
        with open(args.reuse_binary_cache, "rb") as f:
            trajectories = pickle.load(f)
    else:
        # 2) 軌跡を新規抽出 (extract_mce_event_trajectories.py と同じ実装を呼ぶ)
        from train_model_event import (  # type: ignore
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
            "軌跡抽出開始 [event-level, mode=%s, sliding=%dd, max_events=%d, n_jobs=%d]",
            mode_label, args.sliding_window_days, args.max_events, args.n_jobs,
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

    # マルチクラス step_actions を計算
    n_total = 0
    cls_counter: Counter = Counter()
    seq_lens: List[int] = []
    for traj in trajectories:
        labels = traj.get("step_labels", []) or []
        sed = traj.get("step_event_dirs", []) or []
        L = min(len(labels), len(sed)) if sed else len(labels)
        new_actions: List[int] = []
        for i in range(L):
            if labels[i] == 0:
                new_actions.append(0)
            else:
                ev_dirs = sed[i] if i < len(sed) else []
                new_actions.append(_step_event_dir_to_class(ev_dirs, class_map, other_id))
        # step_event_dirs が無い軌跡には全 other を割り当て
        for i in range(L, len(labels)):
            new_actions.append(0 if labels[i] == 0 else other_id)
        traj["step_actions"] = new_actions
        traj["multi_class_action"] = True
        traj["num_actions"] = num_actions
        traj["dir_class_mapping_path"] = str(args.dir_class_mapping)
        cls_counter.update(new_actions)
        n_total += len(new_actions)
        seq_lens.append(len(new_actions))

    if n_total > 0:
        logger.info(
            "step_actions マルチクラス化完了: 軌跡 %d, total steps %d, num_actions=%d",
            len(trajectories), n_total, num_actions,
        )
        for cid in sorted(cls_counter):
            v = cls_counter[cid]
            logger.info("  class %2d : %7d (%5.2f%%)", cid, v, v / n_total * 100)
        logger.info("最頻クラス: %s", cls_counter.most_common(1)[0])
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
        "保存完了 [MCE-IRL event-level multiclass, num_actions=%d]: %s (%d 軌跡)",
        num_actions, output_path, len(trajectories),
    )


if __name__ == "__main__":
    main()
