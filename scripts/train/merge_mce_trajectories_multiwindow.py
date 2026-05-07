#!/usr/bin/env python3
"""
Plan A-3: 月次 MCE-IRL の 4 訓練窓統合 (multi-window) cache 生成
=================================================================
各 (developer, directory) ペアの軌跡は、特徴量計算期間（state, path, event）は
窓に依存せず同じだが、step_labels（「月 t 末から先 N ヶ月以内に accept したか」）が
窓ごとに異なる。

A-3 のアプローチ: 4 つの窓 (0-3, 3-6, 6-9, 9-12m) のキャッシュを単純連結し、
各軌跡に `future_window_id` を付与した上で 1 つの統合 cache を作る。

  - 同じ (dev, dir) ペアが 4 軌跡として登場
  - 軌跡内の state は同じだが step_labels が違う
  - モデルは「同じ state でも違う未来窓を区別して予測する」ことを学ぶ

学習側は既存の二値 MCE-IRL (step_actions = step_labels の 0/1) をそのまま使い、
データ量を 4 倍化することで、未来窓に対するロバスト性 / 汎化能力を獲得することを狙う。

使い方:
    uv run python scripts/train/merge_mce_trajectories_multiwindow.py \\
        --inputs outputs/trajectory_cache/traj_0-3.pkl \\
                 outputs/trajectory_cache/traj_3-6.pkl \\
                 outputs/trajectory_cache/traj_6-9.pkl \\
                 outputs/trajectory_cache/traj_9-12.pkl \\
        --window-labels 0-3 3-6 6-9 9-12 \\
        --output outputs/mce_pilot_multiwindow/cache/monthly_traj_multiwindow.pkl
"""

import argparse
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Plan A-3: 月次 IRL の 4 訓練窓 cache を統合"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="月次軌跡 cache (.pkl) のリスト。窓ごとに 1 個",
    )
    parser.add_argument(
        "--window-labels",
        type=str,
        nargs="+",
        required=True,
        help="各 input に対応する窓ラベル (例: 0-3 3-6 6-9 9-12)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="統合 cache の出力パス (.pkl)",
    )
    parser.add_argument(
        "--no-add-step-actions",
        action="store_true",
        help="step_actions を step_labels から再生成しない (既に設定済みの場合)",
    )
    args = parser.parse_args()

    if len(args.inputs) != len(args.window_labels):
        raise ValueError(
            f"inputs ({len(args.inputs)}) と window-labels ({len(args.window_labels)}) の数が一致しません"
        )

    out_path = Path(args.output)
    if out_path.exists():
        logger.info("出力が既に存在、スキップ: %s", out_path)
        return

    merged: List[Dict[str, Any]] = []
    label_dist_per_window: Dict[str, Counter] = {}
    n_per_window: Dict[str, int] = {}

    for input_path, window_label in zip(args.inputs, args.window_labels):
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"cache が見つかりません: {path}")
        logger.info("窓 %s の cache を読み込み: %s", window_label, path)
        with open(path, "rb") as f:
            trajectories: List[Dict[str, Any]] = pickle.load(f)

        n_per_window[window_label] = len(trajectories)
        cnt = Counter()

        for traj in trajectories:
            traj["future_window_id"] = window_label
            # step_actions を step_labels から再生成 (未設定 / 二値の保証)
            if not args.no_add_step_actions:
                step_labels = traj.get("step_labels", []) or []
                traj["step_actions"] = [int(bool(l)) for l in step_labels]
                cnt.update(traj["step_actions"])
            else:
                if "step_actions" in traj:
                    cnt.update(traj["step_actions"])

        label_dist_per_window[window_label] = cnt
        merged.extend(trajectories)
        logger.info(
            "  軌跡 %d, step 合計 %d (positive %d / total %d, %.2f%%)",
            len(trajectories),
            sum(cnt.values()),
            cnt[1],
            sum(cnt.values()),
            cnt[1] / max(sum(cnt.values()), 1) * 100,
        )

    logger.info("=" * 60)
    logger.info("統合完了: 総軌跡数 %d", len(merged))
    for window_label, cnt in label_dist_per_window.items():
        n = n_per_window[window_label]
        total = sum(cnt.values())
        logger.info(
            "  窓 %5s : 軌跡=%d, total step=%d, positive=%d (%.2f%%)",
            window_label, n, total, cnt[1], cnt[1] / max(total, 1) * 100,
        )

    # 系列長サマリ
    seq_lens = [t.get("seq_len") or len(t.get("step_labels", [])) for t in merged]
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
        pickle.dump(merged, f)
    logger.info("保存完了: %s (%d 軌跡)", out_path, len(merged))


if __name__ == "__main__":
    main()
