#!/usr/bin/env python3
"""
レビュー承諾予測 - MCE-IRL 学習 (イベント単位 / マルチクラス accept action 版)
============================================================================
Plan B-1 Phase 1.5 / 学習ドライバ。

train_mce_event_irl.py を参考にしつつ、マルチクラス accept action 専用に
書き直した薄いランチャ:

    - 入力は extract_mce_event_trajectories_multiclass.py で生成した
      `step_actions` を持つ軌跡 cache (.pkl) のみ。
    - num_actions は cache の `num_actions` フィールド、または
      step_actions の max+1 から自動判定する。
    - MCEIRLSystemMulticlass を初期化し、親 (MCEIRLSystem) の
      train_mce_irl_temporal_trajectories で学習する。
    - 学習後は model_metadata.json に
      { multi_class_action: True, num_actions, dir_class_mapping_path }
      を保存する。

使い方:
    uv run python scripts/train/train_mce_event_irl_multiclass.py \
        --trajectories-cache outputs/mce_pilot_event_dev_multiclass/cache/event_traj_0-3.pkl \
        --dir-class-mapping outputs/dir_class_mapping_K15.json \
        --epochs 50 --patience 5 --model-type 0 \
        --output outputs/mce_pilot_event_dev_multiclass/train_0-3m
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

RANDOM_SEED = 777
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from review_predictor.IRL.model.mce_irl_predictor_multiclass import (  # noqa: E402
    MCEIRLSystemMulticlass,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _infer_event_state_dim(trajectories: List[Dict[str, Any]]) -> int:
    """軌跡から state_dim を判定 (20 base + path 3 + event 4)。"""
    base = 20
    if not trajectories:
        return base
    sample = trajectories[0]
    pf = sample.get("path_features_per_step")
    if pf is not None and len(pf) > 0:
        base += 3
    ef = sample.get("event_features")
    if ef is not None and len(ef) > 0:
        base += 4
    return base


def _infer_num_actions(trajectories: List[Dict[str, Any]]) -> int:
    """cache から num_actions を判定。優先順位: 軌跡 metadata → step_actions max+1。"""
    if not trajectories:
        return 2
    sample = trajectories[0]
    if isinstance(sample.get("num_actions"), int) and sample["num_actions"] >= 2:
        return int(sample["num_actions"])
    max_action = 0
    for t in trajectories:
        sa = t.get("step_actions") or []
        for v in sa:
            if int(v) > max_action:
                max_action = int(v)
    return max_action + 1


def _validate_cache(trajectories: List[Dict[str, Any]], cache_path: str) -> None:
    if not trajectories:
        raise ValueError(f"軌跡キャッシュが空です: {cache_path}")
    sample = trajectories[0]
    needed = (
        "step_actions",
        "event_features",
        "path_features_per_step",
        "monthly_activity_histories",
    )
    missing = [k for k in needed if k not in sample]
    if missing:
        raise ValueError(
            f"軌跡キャッシュ {cache_path} はイベント単位 MCE-IRL 用ではありません。"
            f" 欠損キー: {missing}"
        )
    if not (sample.get("event_features") or []):
        raise ValueError(
            f"軌跡キャッシュ {cache_path} に event_features が空です。"
        )


def main():
    parser = argparse.ArgumentParser(
        description="MCE-IRL 学習 (イベント単位 / マルチクラス accept action 版)"
    )
    parser.add_argument(
        "--trajectories-cache",
        type=str,
        required=True,
        help="extract_mce_event_trajectories_multiclass.py で生成した軌跡 cache (.pkl)",
    )
    parser.add_argument(
        "--dir-class-mapping",
        type=str,
        default=None,
        help="参照のため metadata に記録する dir_class_mapping JSON のパス (省略時は cache から読み取り)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--model-type",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="ネットワークバリアント (0:LSTM, 1:LSTM+Attn, 2:Transformer)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力ディレクトリ (mce_event_irl_model.pt と model_metadata.json を保存)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.trajectories_cache
    logger.info("軌跡キャッシュを読み込み: %s", cache_path)
    with open(cache_path, "rb") as f:
        train_trajectories = pickle.load(f)
    _validate_cache(train_trajectories, cache_path)

    state_dim = _infer_event_state_dim(train_trajectories)
    num_actions = _infer_num_actions(train_trajectories)
    cache_per_dev = bool(train_trajectories[0].get("per_dev", False))
    multi_class_action = bool(train_trajectories[0].get("multi_class_action", False))

    if num_actions < 3:
        logger.warning(
            "num_actions=%d (< 3) は二値学習相当です。"
            " step_actions が二値のままになっていないか確認してください。",
            num_actions,
        )

    dir_class_mapping_path = (
        args.dir_class_mapping
        or train_trajectories[0].get("dir_class_mapping_path")
    )

    logger.info(
        "学習設定: state_dim=%d, action_dim=5, num_actions=%d, model_type=%d, "
        "per_dev=%s, multi_class_action=%s",
        state_dim, num_actions, args.model_type,
        cache_per_dev, multi_class_action,
    )
    logger.info("dir_class_mapping_path = %s", dir_class_mapping_path)

    config: Dict[str, Any] = {
        "state_dim": state_dim,
        "action_dim": 5,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "model_type": args.model_type,
        "learning_rate": args.learning_rate,
        "num_actions": num_actions,
        "dir_class_mapping_path": dir_class_mapping_path,
    }
    irl_system = MCEIRLSystemMulticlass(config)

    # 親クラスの学習ループを使用 (CE 損失は num_actions 次元で動作)
    logger.info("MCE-IRL multiclass モデルを訓練...")
    train_stats = irl_system.train_mce_irl_temporal_trajectories(
        train_trajectories,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    # モデル保存
    model_path = output_dir / "mce_event_irl_model.pt"
    torch.save(irl_system.network.state_dict(), model_path)
    logger.info("モデルを保存: %s", model_path)

    metadata = {
        "model_class": "mce_event_irl_multiclass",
        "model_type": int(args.model_type),
        "state_dim": int(state_dim),
        "action_dim": 5,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "num_actions": int(num_actions),
        "multi_class_action": True,
        "dir_class_mapping_path": dir_class_mapping_path,
        "loss": "softmax_cross_entropy_trajectory_nll",
        "step_unit": "event",
        "per_dev": bool(cache_per_dev),
        "trajectories_cache": str(cache_path),
        "epochs_trained": int(train_stats.get("epochs_trained", 0)),
        "best_epoch": int(train_stats.get("best_epoch", 0)),
        "best_val_loss": float(train_stats.get("best_val_loss", 0.0)),
    }
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("メタデータを保存: %s", metadata_path)

    # 学習統計も保存 (train/val NLL の履歴)
    train_history_path = output_dir / "train_history.json"
    with open(train_history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_losses": train_stats.get("training_losses", []),
                "val_losses": train_stats.get("val_losses", []),
                "best_val_loss": train_stats.get("best_val_loss", 0.0),
                "best_epoch": train_stats.get("best_epoch", 0),
                "epochs_trained": train_stats.get("epochs_trained", 0),
            },
            f,
            indent=2,
        )
    logger.info("学習履歴を保存: %s", train_history_path)
    logger.info("=" * 80)
    logger.info(
        "完了: best_epoch=%d best_val_NLL=%.4f epochs=%d num_actions=%d",
        metadata["best_epoch"], metadata["best_val_loss"],
        metadata["epochs_trained"], metadata["num_actions"],
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
