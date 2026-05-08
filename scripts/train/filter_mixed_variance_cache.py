#!/usr/bin/env python3
"""per-dev event 軌跡から「内部バリアンスのある」dev のみを抽出する。

step_labels が全部 1 or 全部 0 の dev は dir 条件付け学習に寄与しないため、
混在 dev (step_labels に 0 と 1 が両方含まれる) のみを残した cache を作る。
学習時はこの cache を使い、推論時は学習対象外の dev も含めて評価することで、
「dir 条件付け信号を濃くした学習が (dev, dir) AUC を改善するか」を検証する。

使い方:
    uv run python scripts/train/filter_mixed_variance_cache.py \\
        --input  outputs/mce_pilot_event_dev/cache/event_traj_0-3.pkl \\
        --output outputs/mce_pilot_event_dev_mixed/cache/event_traj_0-3.pkl
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument(
        "--min-seq-len", type=int, default=2,
        help="最低系列長 (混在判定には 2 以上必要)",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"読み込み: {in_path}")
    with open(in_path, "rb") as f:
        trajs = pickle.load(f)
    logger.info(f"全軌跡数: {len(trajs)}")

    kept = []
    n_all_pos = 0
    n_all_neg = 0
    n_short = 0
    for t in trajs:
        labels = t.get("step_labels", []) or []
        if len(labels) < args.min_seq_len:
            n_short += 1
            continue
        if all(l == 1 for l in labels):
            n_all_pos += 1
            continue
        if all(l == 0 for l in labels):
            n_all_neg += 1
            continue
        kept.append(t)

    logger.info(f"フィルタ結果:")
    logger.info(f"  全 1 (除外): {n_all_pos}")
    logger.info(f"  全 0 (除外): {n_all_neg}")
    logger.info(f"  系列長 <{args.min_seq_len} (除外): {n_short}")
    logger.info(f"  混在 (保持): {len(kept)}")
    if kept:
        seq_lens = [t["seq_len"] for t in kept]
        logger.info(
            f"  保持軌跡 系列長: mean={np.mean(seq_lens):.1f}, "
            f"median={np.median(seq_lens):.0f}, max={max(seq_lens)}"
        )
        # ステップレベル偏り
        all_labels = [l for t in kept for l in t["step_labels"]]
        pos_rate = sum(all_labels) / len(all_labels)
        logger.info(
            f"  保持軌跡 ステップレベル正例率: {pos_rate:.1%} "
            f"({sum(all_labels)}/{len(all_labels)})"
        )

    with open(out_path, "wb") as f:
        pickle.dump(kept, f)
    logger.info(f"保存: {out_path} ({len(kept)} 軌跡)")


if __name__ == "__main__":
    main()
