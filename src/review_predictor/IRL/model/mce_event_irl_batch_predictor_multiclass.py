"""MCE-IRL イベント単位 / マルチクラス accept action モデル用のバッチ推論クラス
(Plan B-1 Phase 2.1〜2.3)

mce_event_irl_batch_predictor.py を継承し、

  - dir → class_id 変換 (dir_class_mapping JSON 経由)
  - (dev, dir) 予測: π(a = target_class_id | s_last) を返す (β 戦略は使わない)
  - dev レベル予測: π(a ≠ 0 | s_last) を返す

を提供する。学習側で MCEIRLSystemMulticlass を使ったモデルを想定。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from review_predictor.IRL.model.mce_event_irl_batch_predictor import (
    MCEEventBatchContinuationPredictor,
)

logger = logging.getLogger(__name__)


class MCEEventMulticlassBatchPredictor(MCEEventBatchContinuationPredictor):
    """マルチクラス accept action 用の Batch predictor。"""

    def __init__(
        self,
        *args,
        dir_class_mapping_path: Optional[str | Path] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # dir_class_mapping は明示指定がなければ model_metadata.json から読み込む
        self._dir_class_mapping_override = (
            str(dir_class_mapping_path) if dir_class_mapping_path else None
        )
        self._class_map: Dict[str, int] = {}
        self._other_class_id: int = 0
        self._num_actions: int = 2
        self._dir_mapping_depth: int = 1

    # ------------------------------------------------------------------
    # モデルロード (multiclass モデル用にオーバーライド)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._irl_system is not None:
            return

        from review_predictor.IRL.features.common_features import ACTION_FEATURES
        from review_predictor.IRL.model.mce_irl_predictor_multiclass import (
            MCEIRLSystemMulticlass,
        )

        metadata_path = self.model_path.parent / "model_metadata.json"
        model_type = 0
        meta_per_dev = False
        num_actions = 2
        dir_class_mapping_path = self._dir_class_mapping_override
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            model_type = int(metadata.get("model_type", 0))
            meta_per_dev = bool(metadata.get("per_dev", False))
            num_actions = int(metadata.get("num_actions", 2))
            mcls = metadata.get("model_class", "")
            if mcls and mcls != "mce_event_irl_multiclass":
                raise ValueError(
                    f"{self.model_path} の model_class={mcls!r} はマルチクラス用ではありません。"
                    f" mce_event_irl_multiclass のチェックポイントを指定してください。"
                )
            if not metadata.get("multi_class_action", False):
                logger.warning(
                    "model_metadata.json に multi_class_action フラグがありません。"
                    " 二値学習されたモデルかもしれません。"
                )
            if dir_class_mapping_path is None:
                dir_class_mapping_path = metadata.get("dir_class_mapping_path")
        else:
            logger.warning(
                "model_metadata.json が見つかりません: %s。multi-class 設定を直接 推測します。",
                metadata_path,
            )

        # per_dev フラグ確定
        if self._per_dev_override is None:
            self.per_dev = meta_per_dev
        else:
            self.per_dev = bool(self._per_dev_override)

        # state_dict から state_dim と num_actions (reward head 出力次元) を確認
        state_dict = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        state_encoder_w = state_dict.get("state_encoder.0.weight")
        state_dim = state_encoder_w.shape[1] if state_encoder_w is not None else 27
        if state_dim != 27:
            raise ValueError(
                f"{self.model_path} の state_dim={state_dim} は 27 ではありません。"
                f" イベント単位 MCE-IRL モデルでない可能性があります。"
            )
        # 最終 Linear 層 (= reward head 出力) の out_features を num_actions として採用。
        # reward_predictor は Linear → ReLU → Dropout → Linear の構成で、weight 名のうち
        # サフィックス番号が最大のものが最終 Linear。
        reward_weight_keys = sorted(
            (k for k in state_dict.keys()
             if k.startswith("reward_predictor.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        if reward_weight_keys:
            last_w = state_dict[reward_weight_keys[-1]]
            num_actions_from_weights = int(last_w.shape[0])
            if num_actions_from_weights != num_actions:
                logger.info(
                    "metadata num_actions=%d と state_dict reward head 出力次元 %d が"
                    " 異なるため、後者を採用します。",
                    num_actions, num_actions_from_weights,
                )
                num_actions = num_actions_from_weights
        if num_actions < 2:
            raise ValueError(f"num_actions={num_actions} は不正です (>= 2 が必要)。")

        config = {
            "state_dim": state_dim,
            "action_dim": len(ACTION_FEATURES),
            "hidden_dim": 128,
            "dropout": 0.1,
            "model_type": model_type,
            "num_actions": num_actions,
            "dir_class_mapping_path": dir_class_mapping_path,
        }
        irl_system = MCEIRLSystemMulticlass(config)
        irl_system.device = self.device
        irl_system.network = irl_system.network.to(self.device)
        irl_system.network.load_state_dict(state_dict)
        irl_system.network.eval()

        self._irl_system = irl_system
        self._num_actions = num_actions

        # dir_class_mapping を読み込み
        if dir_class_mapping_path:
            self._load_dir_class_mapping(dir_class_mapping_path)
        else:
            logger.warning(
                "dir_class_mapping JSON が指定されていません。"
                " predict_developer_directory は (dev, dir) 予測に other クラスを使い続けます。"
            )

        logger.info(
            "MCE-IRL [event/multiclass] モデルをロード: %s "
            "(model_type=%d, state_dim=%d, num_actions=%d, per_dev=%s, depth=%d)",
            self.model_path, model_type, state_dim, num_actions,
            self.per_dev, self._dir_mapping_depth,
        )

    def _load_dir_class_mapping(self, json_path: str | Path) -> None:
        """dir_class_mapping JSON を読み込む。"""
        path = Path(json_path)
        if not path.exists():
            logger.warning("dir_class_mapping JSON が見つかりません: %s", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        classes: Dict[str, int] = {k: int(v) for k, v in m["classes"].items()}
        self._class_map = classes
        self._other_class_id = int(classes.get("other", max(classes.values())))
        self._num_actions = int(m.get("num_actions", self._num_actions))
        self._dir_mapping_depth = int(m.get("depth", 1))

    # ------------------------------------------------------------------
    # dir → class_id ヘルパ
    # ------------------------------------------------------------------

    def _dir_to_class_id(self, directory: str) -> int:
        """target_dir を depth=1 親に正規化し class_id を返す (未知は other)。"""
        if not directory:
            return self._other_class_id
        d = directory.split("/", 1)[0] if self._dir_mapping_depth == 1 else directory
        return int(self._class_map.get(d, self._other_class_id))

    # ------------------------------------------------------------------
    # 推論 API (multiclass 版)
    # ------------------------------------------------------------------

    def predict_developer_directory(
        self,
        email: str,
        directory: str,
        prediction_time: datetime,
        path_extractor=None,
        head_index: int = 0,
    ) -> float:
        """π(a = class_id(directory) | s_last) を返す。

        β 戦略 (path_features 上書き) は使わない。dir 条件付けは action 出力分岐に委ねる。
        """
        self._load_model()

        # 軌跡は per-dev モデルなら全 dir 横断、(dev, dir) ペアモデルなら従来どおり
        if self.per_dev:
            seq = self._build_dev_event_sequence(
                email, None, prediction_time, path_extractor
            )
        else:
            seq = self._build_event_sequence(
                email, directory, prediction_time, path_extractor
            )

        (
            event_histories,
            step_context_dates,
            step_total_proj,
            path_feats,
            event_feats,
            developer_info,
        ) = seq

        if developer_info is None or not event_histories:
            return 0.5

        target_cid = self._dir_to_class_id(directory)
        result = self._irl_system.predict_continuation_probability_monthly(
            developer=developer_info,
            monthly_activity_histories=event_histories,
            step_context_dates=step_context_dates,
            context_date=prediction_time,
            step_total_project_reviews=step_total_proj,
            step_path_features=path_feats,
            step_event_features=event_feats,
            head_index=head_index,
            target_class_id=target_cid,
        )
        return float(result.get("continuation_probability", 0.5))

    def predict_developer(
        self,
        email: str,
        prediction_time: datetime,
        path_extractor=None,
        head_index: int = 0,
    ) -> float:
        """dev レベル継続確率 = π(a ≠ 0 | s_last) を返す。

        per_dev=True で学習されたモデル前提。
        """
        self._load_model()
        if not self.per_dev:
            raise ValueError(
                "predict_developer は per_dev=True で学習されたモデルでのみ使えます。"
            )

        seq = self._build_dev_event_sequence(
            email, None, prediction_time, path_extractor
        )
        (
            event_histories,
            step_context_dates,
            step_total_proj,
            path_feats,
            event_feats,
            developer_info,
        ) = seq

        if developer_info is None or not event_histories:
            return 0.5

        result = self._irl_system.predict_continuation_probability_monthly(
            developer=developer_info,
            monthly_activity_histories=event_histories,
            step_context_dates=step_context_dates,
            context_date=prediction_time,
            step_total_project_reviews=step_total_proj,
            step_path_features=path_feats,
            step_event_features=event_feats,
            head_index=head_index,
            target_class_id=None,  # dev レベル: π(a ≠ 0 | s)
        )
        return float(result.get("continuation_probability", 0.5))

    def predict_developer_directory_distribution(
        self,
        email: str,
        prediction_time: datetime,
        path_extractor=None,
    ) -> List[float]:
        """1 reviewer に対し 1 回の推論で全 K+2 クラスの確率を返す。

        (dev, dir) 予測を多 dir 一括で行うときに dev レベル軌跡構築コストを削減する用途。
        """
        self._load_model()
        if not self.per_dev:
            raise ValueError(
                "predict_developer_directory_distribution は per_dev=True 専用です。"
            )

        seq = self._build_dev_event_sequence(
            email, None, prediction_time, path_extractor
        )
        (
            event_histories,
            step_context_dates,
            step_total_proj,
            path_feats,
            event_feats,
            developer_info,
        ) = seq

        if developer_info is None or not event_histories:
            return [1.0 / self._num_actions] * self._num_actions

        info = self._irl_system.predict_action_distribution_monthly(
            developer=developer_info,
            monthly_activity_histories=event_histories,
            step_context_dates=step_context_dates,
            context_date=prediction_time,
            step_total_project_reviews=step_total_proj,
            step_path_features=path_feats,
            step_event_features=event_feats,
        )
        probs = info.get("action_probabilities") or [
            1.0 / self._num_actions
        ] * self._num_actions
        return [float(p) for p in probs]


__all__ = ["MCEEventMulticlassBatchPredictor"]
