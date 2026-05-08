"""Maximum Causal Entropy IRL — マルチクラス accept action 版 (Plan B-1)

action 空間を {0=reject, 1..K=accept(depth=1 dir cluster), K+1=other_accept} に
拡張した IRL システム。MCEIRLSystem を継承し、必要箇所のみオーバーライドする:

  - num_actions を config 経由で受け取り、Network の reward head 出力次元を K+2 にする
  - 学習時は trajectory["step_actions"] (multi-class) を long tensor にして CE で学習
  - 推論時は accept クラスごとの確率 π(a=c|s) と π(a≠0|s) を返す API を追加

二値版 MCEIRLSystem の既存挙動には触らない。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from review_predictor.IRL.features.common_features import (
    ACTION_FEATURES,
    STATE_FEATURES,
)
from review_predictor.IRL.model.mce_irl_predictor import (
    MCEIRLSystem,
    create_mce_network,
)

logger = logging.getLogger(__name__)


class MCEIRLSystemMulticlass(MCEIRLSystem):
    """マルチクラス accept action 用の MCE-IRL システム。

    config 追加項目:
        num_actions (int)            : action 空間サイズ (= K+2 など)
        dir_class_mapping_path (str) : 推論時に dir → class_id を引くための JSON

    軌跡側の追加項目:
        step_actions (List[int])         : 0..num_actions-1 のクラス ID 列
        multi_class_action (bool, 任意) : True 想定 (チェック用)
        num_actions (int, 任意)          : cache とモデルの整合性確認用
    """

    def __init__(self, config: Dict[str, Any]):
        # 親 (MCEIRLSystem) は NUM_ACTIONS=2 をクラス変数で固定しているため、
        # ここでは super().__init__ を呼ばず、必要な部分だけ自前で初期化する
        # (親のさらに親 RetentionIRLSystem の __init__ も呼ばない: ネットワーク構築だけ自前で)。
        self.config = config
        self.state_dim = config.get("state_dim", len(STATE_FEATURES))
        self.action_dim = config.get("action_dim", len(ACTION_FEATURES))
        self.hidden_dim = config.get("hidden_dim", 128)
        self.dropout = config.get("dropout", 0.1)
        self.output_temperature = float(config.get("output_temperature", 1.0))

        num_actions = int(config.get("num_actions", 2))
        if num_actions < 2:
            raise ValueError(f"num_actions must be >= 2, got {num_actions}")
        self.num_actions = num_actions
        # 親クラスのコードが NUM_ACTIONS を参照するケースに備えて instance 属性で上書き
        self.NUM_ACTIONS = num_actions

        self.dir_class_mapping_path = config.get("dir_class_mapping_path")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_type = config.get("model_type", 0)
        self.network = create_mce_network(
            self.model_type,
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.dropout,
            num_actions=self.num_actions,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self._scheduler: Optional[Any] = None

        # 親の互換用フィールド (使用しない)
        self.focal_alpha = 0.5
        self.focal_gamma = 0.0
        import torch.nn as nn
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        logger.info(
            "MCE-IRL (multiclass) システムを初期化 "
            "(state_dim=%d, action_dim=%d, model_type=%d, num_actions=%d)",
            self.state_dim, self.action_dim, self.model_type, self.num_actions,
        )

    # -----------------------------------------------------------------
    # _precompute_trajectories: actions を step_actions 由来のマルチクラスに
    # -----------------------------------------------------------------

    def _precompute_trajectories(
        self,
        trajectories_list: List[Dict[str, Any]],
        split_name: str = "",
    ) -> List[Dict[str, torch.Tensor]]:
        """親実装をそのまま呼び、最後に actions を step_actions ベースで上書きする。

        親実装は actions を step_labels から二値化していたが、本クラスでは
        step_actions (マルチクラス) を真とする。トラジェクトリの状態量計算は
        重い (joblib 並列) ため、二度回さないように親結果を再利用する。
        """
        precomputed = super()._precompute_trajectories(trajectories_list, split_name)

        if len(precomputed) != sum(1 for t in trajectories_list if self._traj_is_valid(t)):
            # 親が None で落とした軌跡があるため、index 一致は保証されない。
            # → 軌跡内の step_actions と min_len から再構築する戦略に切り替える。
            pass

        # 親が落とさなかった軌跡を順序保持で取り出して step_actions を貼る。
        valid_trajs = [t for t in trajectories_list if self._traj_is_valid(t)]
        # 親実装も同じ順序で valid だけ残すはずだが、長さが合わないケースに保険。
        n = min(len(precomputed), len(valid_trajs))

        for i in range(n):
            entry = precomputed[i]
            traj = valid_trajs[i]
            min_len = entry["min_len"]
            step_actions = traj.get("step_actions") or []
            if not step_actions:
                # multi-class でない軌跡 → 二値 (step_labels 由来) のままにしておく
                continue
            actions_long = torch.tensor(
                [int(a) for a in step_actions[:min_len]],
                dtype=torch.long,
                device=self.device,
            )
            if actions_long.numel() < min_len:
                pad = torch.zeros(min_len - actions_long.numel(), dtype=torch.long, device=self.device)
                actions_long = torch.cat([actions_long, pad], dim=0)
            entry["actions"] = actions_long

        return precomputed

    @staticmethod
    def _traj_is_valid(trajectory: Dict[str, Any]) -> bool:
        """親 _precompute_trajectories と同じ「採用条件」を再現する補助。"""
        activity_history = trajectory.get("activity_history", [])
        step_labels = trajectory.get("step_labels", [])
        monthly_histories = trajectory.get("monthly_activity_histories", [])
        if (not activity_history and not monthly_histories) or not step_labels or not monthly_histories:
            return False
        return True

    # -----------------------------------------------------------------
    # 推論: マルチクラス用 API
    # -----------------------------------------------------------------

    def predict_action_distribution_monthly(
        self,
        developer: Dict[str, Any],
        monthly_activity_histories: List[List[Dict[str, Any]]],
        step_context_dates: List,
        context_date: Optional[datetime] = None,
        step_total_project_reviews: Optional[List[int]] = None,
        step_path_features: Optional[List[np.ndarray]] = None,
        step_event_features: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """月次/イベント系列から最終ステップでの π(a=k|s) [num_actions 次元] を返す。"""
        if context_date is None:
            context_date = datetime.now()

        self.network.eval()

        with torch.no_grad():
            if not monthly_activity_histories:
                return {
                    "action_probabilities": [1.0 / self.num_actions] * self.num_actions,
                    "accept_probability": (self.num_actions - 1) / self.num_actions,
                    "confidence": 0.0,
                    "reasoning": "履歴が不足しているため、一様分布を返します",
                }

            email = developer.get(
                "email",
                developer.get("developer_id", developer.get("reviewer", "")),
            )

            state_tensors: List[torch.Tensor] = []
            action_tensors: List[torch.Tensor] = []
            for i, month_history in enumerate(monthly_activity_histories):
                if step_context_dates and i < len(step_context_dates):
                    month_ctx = step_context_dates[i]
                else:
                    month_ctx = context_date

                if not month_history:
                    state_tensors.append(torch.zeros(self.state_dim, device=self.device))
                    action_tensors.append(torch.zeros(self.action_dim, device=self.device))
                    continue

                total_proj = (
                    step_total_project_reviews[i]
                    if step_total_project_reviews and i < len(step_total_project_reviews)
                    else 0
                )
                pf = (
                    step_path_features[i]
                    if step_path_features and i < len(step_path_features)
                    else None
                )
                ef = (
                    step_event_features[i]
                    if step_event_features and i < len(step_event_features)
                    else None
                )
                s_t, a_t = self.extract_features_tensor(
                    email,
                    month_history,
                    month_ctx,
                    total_project_reviews=total_proj,
                    path_features_vec=pf,
                    event_features_vec=ef,
                )
                state_tensors.append(s_t)
                action_tensors.append(a_t)

            if not state_tensors:
                return {
                    "action_probabilities": [1.0 / self.num_actions] * self.num_actions,
                    "accept_probability": (self.num_actions - 1) / self.num_actions,
                    "confidence": 0.0,
                    "reasoning": "有効ステップなし",
                }

            state_seq = torch.stack(state_tensors).unsqueeze(0)
            action_seq = torch.stack(action_tensors).unsqueeze(0)
            lengths = torch.tensor(
                [len(state_tensors)], dtype=torch.long, device=self.device
            )

            logits = self.network(state_seq, action_seq, lengths)  # [1, num_actions]
            probs = F.softmax(logits, dim=-1)[0]  # [num_actions]
            probs_list = [float(p.item()) for p in probs]

            accept_prob = float(1.0 - probs_list[0])
            if (
                self.output_temperature
                and abs(self.output_temperature - 1.0) > 1e-6
            ):
                # accept_prob のみ温度補正 (per-class 補正は行わない)
                p = min(max(accept_prob, 1e-6), 1.0 - 1e-6)
                logit = math.log(p / (1.0 - p))
                accept_prob = 1.0 / (
                    1.0 + math.exp(-logit / self.output_temperature)
                )

            return {
                "action_probabilities": probs_list,
                "accept_probability": accept_prob,
                "confidence": min(abs(accept_prob - 0.5) * 2, 1.0),
                "reasoning": (
                    f"MCE-IRL multiclass π(a≠0|s) = {accept_prob:.1%} "
                    f"(num_actions={self.num_actions})"
                ),
                "method": "mce_irl_multiclass",
            }

    def predict_continuation_probability_monthly(
        self,
        developer: Dict[str, Any],
        monthly_activity_histories: List[List[Dict[str, Any]]],
        step_context_dates: List,
        context_date: Optional[datetime] = None,
        step_total_project_reviews: Optional[List[int]] = None,
        step_path_features: Optional[List[np.ndarray]] = None,
        step_event_features: Optional[List[Dict[str, float]]] = None,
        head_index: int = 0,
        target_class_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """既存 API 互換: continuation_probability を返すが意味を拡張する。

        target_class_id を指定: π(a=target|s) を継続確率として返す ((dev, dir) 用)。
        target_class_id=None      : π(a≠0|s) = 1 - π(a=0|s) を返す (dev レベル用)。
        """
        info = self.predict_action_distribution_monthly(
            developer=developer,
            monthly_activity_histories=monthly_activity_histories,
            step_context_dates=step_context_dates,
            context_date=context_date,
            step_total_project_reviews=step_total_project_reviews,
            step_path_features=step_path_features,
            step_event_features=step_event_features,
        )
        probs = info.get("action_probabilities", [])
        if target_class_id is None:
            cont_prob = info.get("accept_probability", 0.5)
            reasoning = (
                f"MCE-IRL multiclass π(a≠0|s) = {cont_prob:.1%}"
            )
        else:
            cid = int(target_class_id)
            if 0 <= cid < len(probs):
                cont_prob = float(probs[cid])
            else:
                cont_prob = 0.0
            reasoning = (
                f"MCE-IRL multiclass π(a={cid}|s) = {cont_prob:.1%}"
            )
        return {
            "continuation_probability": cont_prob,
            "action_probabilities": probs,
            "confidence": min(abs(cont_prob - 0.5) * 2, 1.0),
            "reasoning": reasoning,
            "method": "mce_irl_multiclass_monthly",
        }


__all__ = ["MCEIRLSystemMulticlass"]
