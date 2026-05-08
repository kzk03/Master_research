"""Maximum Causal Entropy IRL (MCE-IRL) システム

レビュアー継続予測を、二値 action a_t ∈ {0,1} (= 月 t に当該ディレクトリで
レビュー応答するか否か) を持つ MDP の Inverse RL 問題として定式化する。

定式化:
    - 状態 s_t : 月次集約特徴量 (state 20 + path 3 = 23 次元) と
                 観測行動特徴量 (action features 5 次元) を併せた拡張状態
    - 行動 a_t ∈ {0,1} : 月 t にレビュー応答するか
    - 報酬 R_θ(s, a) : Bi-LSTM で推定するパラメトリック報酬関数
    - 方策 π_θ(a|s) = exp R_θ(s,a) / Σ_{a'} exp R_θ(s,a')   (Boltzmann)
    - 学習目的 : expert demonstrations の trajectory log-likelihood
                L(θ) = -Σ_τ w_τ Σ_t log π_θ(a_t | s_t)

参考:
    Ziebart (2010) "Modeling Purposeful Adaptive Behavior with the
    Principle of Maximum Causal Entropy" PhD thesis, CMU.

設計上の注意:
    - 既存 RetentionIRLSystem の特徴量抽出ヘルパー (_history_to_df,
      extract_features_tensor 等) はそのまま継承して再利用する。
    - ネットワークの reward head は 2-unit (R(s,0), R(s,1)) に拡張。
    - 損失関数は softmax cross entropy = trajectory NLL。
    - 推論時は π(a=1|s) を継続確率として返すため、既存 evaluator との
      互換性が保たれる。
"""

from __future__ import annotations

import copy
import logging
import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from review_predictor.IRL.features.common_features import (
    ACTION_FEATURES,
    STATE_FEATURES,
    extract_common_features,
)
from review_predictor.IRL.model.irl_predictor_v2 import RetentionIRLSystem
from review_predictor.IRL.model.network_variants import (
    BaseIRLNetwork,
    LSTMBackbone,
    TemporalAttention,
    TransformerBackbone,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  MCE-IRL ネットワーク (2-unit reward head)
# ═══════════════════════════════════════════════════════════════


class _MCEIRLBaseNetwork(BaseIRLNetwork):
    """MCE-IRL 共通基底: 2-unit reward head を持つ。

    出力 logits[..., 0] = R_θ(s, a=0), logits[..., 1] = R_θ(s, a=1).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_actions: int = 2,
    ):
        super().__init__(state_dim, action_dim, hidden_dim, dropout)
        self.num_actions = num_actions
        # 2-unit reward head に置き換え
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )


class MCEIRLNetworkLSTM(_MCEIRLBaseNetwork):
    """LSTM ベースの MCE-IRL ネットワーク (model_type=0 互換)。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_actions: int = 2,
    ):
        super().__init__(state_dim, action_dim, hidden_dim, dropout, num_actions)
        self.backbone = LSTMBackbone(hidden_dim, dropout)

    def forward(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """最終ステップの reward logits [B, num_actions] を返す。"""
        combined = self._encode(state, action_features)
        _, last_hidden = self.backbone(combined, lengths)
        return self.reward_predictor(last_hidden)

    def forward_all_steps(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """全ステップの reward logits [B, L, num_actions] を返す。"""
        combined = self._encode(state, action_features)
        lstm_out, _ = self.backbone(combined, lengths)
        B, L, D = lstm_out.shape
        flat = lstm_out.reshape(-1, D)
        return self.reward_predictor(flat).view(B, L, self.num_actions)


class MCEIRLNetworkAttention(_MCEIRLBaseNetwork):
    """LSTM + Temporal Attention の MCE-IRL ネットワーク (model_type=1 互換)。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_actions: int = 2,
    ):
        super().__init__(state_dim, action_dim, hidden_dim, dropout, num_actions)
        self.backbone = LSTMBackbone(hidden_dim, dropout)
        self.attention = TemporalAttention(hidden_dim)

    def forward(self, state, action_features, lengths):
        combined = self._encode(state, action_features)
        lstm_out, last_hidden = self.backbone(combined, lengths)
        context = self.attention(lstm_out, last_hidden, lengths)
        return self.reward_predictor(context)

    def forward_all_steps(self, state, action_features, lengths):
        combined = self._encode(state, action_features)
        lstm_out, _ = self.backbone(combined, lengths)
        B, L, D = lstm_out.shape
        flat = lstm_out.reshape(-1, D)
        return self.reward_predictor(flat).view(B, L, self.num_actions)


class MCEIRLNetworkTransformer(_MCEIRLBaseNetwork):
    """Transformer ベースの MCE-IRL ネットワーク (model_type=2 互換)。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_actions: int = 2,
    ):
        super().__init__(state_dim, action_dim, hidden_dim, dropout, num_actions)
        self.backbone = TransformerBackbone(hidden_dim, dropout=dropout)

    def forward(self, state, action_features, lengths):
        combined = self._encode(state, action_features)
        _, cls_out = self.backbone(combined, lengths)
        return self.reward_predictor(cls_out)

    def forward_all_steps(self, state, action_features, lengths):
        combined = self._encode(state, action_features)
        all_out, _ = self.backbone(combined, lengths)
        B, L, D = all_out.shape
        flat = all_out.reshape(-1, D)
        return self.reward_predictor(flat).view(B, L, self.num_actions)


_MCE_REGISTRY = {
    0: MCEIRLNetworkLSTM,
    1: MCEIRLNetworkAttention,
    2: MCEIRLNetworkTransformer,
}


def create_mce_network(
    variant: int,
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    num_actions: int = 2,
) -> _MCEIRLBaseNetwork:
    if variant not in _MCE_REGISTRY:
        raise ValueError(
            f"MCE-IRL ではバリアント {variant} は未対応 (0/1/2 のみ)"
        )
    return _MCE_REGISTRY[variant](
        state_dim, action_dim, hidden_dim, dropout, num_actions
    )


# ═══════════════════════════════════════════════════════════════
#  MCE-IRL システム
# ═══════════════════════════════════════════════════════════════


class MCEIRLSystem(RetentionIRLSystem):
    """Maximum Causal Entropy IRL システム。

    RetentionIRLSystem の特徴量抽出ヘルパーを継承し、ネットワーク・学習・推論を
    MCE-IRL 形式に置き換える。
    """

    NUM_ACTIONS = 2  # {0: 応答しない, 1: 応答する}

    def __init__(self, config: Dict[str, Any]):
        # 親の __init__ は network 構築まで実行するが、その後に MCE-IRL 用の
        # ネットワークで置き換える。これにより親側の特徴量ヘルパーや
        # focal_loss 関連のフィールドは初期化されたまま残るが、本クラスでは
        # focal_loss は使用しない。
        self.config = config
        self.state_dim = config.get("state_dim", len(STATE_FEATURES))
        self.action_dim = config.get("action_dim", len(ACTION_FEATURES))
        self.hidden_dim = config.get("hidden_dim", 128)
        self.dropout = config.get("dropout", 0.1)
        self.output_temperature = float(config.get("output_temperature", 1.0))

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
            num_actions=self.NUM_ACTIONS,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self._scheduler: Optional[Any] = None

        # 親側の Focal Loss パラメータは未使用だがフィールドとして保持
        # (継承メソッドが参照しても落ちないように)
        self.focal_alpha = 0.5
        self.focal_gamma = 0.0
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        logger.info(
            "MCE-IRL システムを初期化 (state_dim=%d, action_dim=%d, model_type=%d)",
            self.state_dim,
            self.action_dim,
            self.model_type,
        )

    # -----------------------------------------------------------------
    # 学習ループ (Maximum Causal Entropy NLL)
    # -----------------------------------------------------------------

    def train_mce_irl_temporal_trajectories(
        self,
        expert_trajectories: List[Dict[str, Any]],
        epochs: int = 50,
        patience: int = 5,
        val_ratio: float = 0.2,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """MCE-IRL: trajectory log-likelihood (= softmax cross entropy) 最大化。

        Args:
            expert_trajectories: 軌跡リスト。各要素に 'monthly_activity_histories',
                                 'step_labels', 'sample_weight' を含むこと。
                                 step_labels は二値 action 列として再解釈される。

        Returns:
            訓練統計 (training_losses, val_losses, best_epoch, ...)
        """
        # ── train / val 分割 ──
        indices = list(range(len(expert_trajectories)))
        random.seed(42)
        random.shuffle(indices)
        val_size = max(1, int(len(indices) * val_ratio))
        val_indices = set(indices[:val_size])
        train_trajs = [
            t for i, t in enumerate(expert_trajectories) if i not in val_indices
        ]
        val_trajs = [
            t for i, t in enumerate(expert_trajectories) if i in val_indices
        ]

        logger.info("=" * 60)
        logger.info("MCE-IRL 訓練開始")
        logger.info(
            "軌跡数: %d (train=%d, val=%d)",
            len(expert_trajectories),
            len(train_trajs),
            len(val_trajs),
        )
        logger.info("エポック数: %d (patience=%d)", epochs, patience)
        logger.info("=" * 60)

        t0 = max(epochs // 4, 10)
        self._scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t0, T_mult=1, eta_min=1e-5
        )

        train_data = self._precompute_trajectories(train_trajs, "train")
        val_data = self._precompute_trajectories(val_trajs, "val")
        logger.info(
            "特徴量事前計算完了: train=%d, val=%d", len(train_data), len(val_data)
        )

        rng = random.Random(42)

        training_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0

        val_batches = self._collate_batches(val_data, batch_size, shuffle=False, rng=rng)

        epoch = 0
        for epoch in range(epochs):
            self.network.train()
            epoch_loss = 0.0
            batch_count = 0

            train_batches = self._collate_batches(
                train_data, batch_size, shuffle=True, rng=rng
            )
            for batch in train_batches:
                try:
                    loss = self._mce_loss_on_batch(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                except Exception as e:
                    logger.warning("バッチ処理エラー: %s", e)
                    continue

            avg_loss = epoch_loss / max(batch_count, 1)
            training_losses.append(avg_loss)
            if self._scheduler is not None:
                self._scheduler.step(epoch)

            # validation
            self.network.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_batches:
                    try:
                        v = self._mce_loss_on_batch(batch).item()
                        val_loss_sum += v
                        val_count += 1
                    except Exception:
                        continue
            avg_val_loss = val_loss_sum / max(val_count, 1)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.network.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "エポック %d: train_NLL=%.4f, val_NLL=%.4f, LR=%.6f, patience=%d/%d",
                epoch,
                avg_loss,
                avg_val_loss,
                current_lr,
                patience_counter,
                patience,
            )

            if patience_counter >= patience:
                logger.info(
                    "Early stopping: epoch %d, best_epoch=%d, best_val_NLL=%.4f",
                    epoch,
                    best_epoch,
                    best_val_loss,
                )
                break

        if best_state_dict is not None:
            self.network.load_state_dict(best_state_dict)
            logger.info(
                "ベストモデル復元: epoch %d, val_NLL=%.4f", best_epoch, best_val_loss
            )

        logger.info("MCE-IRL 訓練完了")

        return {
            "training_losses": training_losses,
            "val_losses": val_losses,
            "final_loss": training_losses[-1] if training_losses else 0.0,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_trained": epoch + 1 if training_losses else 0,
        }

    # -----------------------------------------------------------------
    # 内部: バッチ単位の MCE-IRL NLL
    # -----------------------------------------------------------------

    def _mce_loss_on_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """1 バッチ分の MCE-IRL NLL を返す。

        NLL = -Σ_t w_t * mask_t * log π_θ(a_t | s_t)
            = Σ_t w_t * mask_t * CE(logits_t, a_t)
        """
        logits = self.network.forward_all_steps(
            batch["state_seq"], batch["action_seq"], batch["lengths"]
        )  # [B, L, num_actions]
        actions = batch["actions"].long()  # [B, L]
        sample_w = batch["sample_weights"]  # [B, L]
        mask = batch["mask"]  # [B, L]

        B, L, A = logits.shape
        flat_logits = logits.reshape(B * L, A)
        flat_actions = actions.reshape(B * L)
        ce = F.cross_entropy(flat_logits, flat_actions, reduction="none").view(B, L)

        weighted = ce * sample_w * mask
        denom = mask.sum().clamp(min=1.0)
        return weighted.sum() / denom

    # -----------------------------------------------------------------
    # 内部: 軌跡 → テンソルの事前計算
    # -----------------------------------------------------------------

    def _precompute_trajectories(
        self,
        trajectories_list: List[Dict[str, Any]],
        split_name: str = "",
    ) -> List[Dict[str, torch.Tensor]]:
        """軌跡を [L, dim] のテンソル群へ展開する (joblib 並列)。

        既存 RetentionIRLSystem.train_irl_temporal_trajectories と同じ前処理を、
        action ラベルを Long テンソルとして保持する形に変えたもの。
        """
        from joblib import Parallel, delayed

        state_dim = self.state_dim
        action_dim = self.action_dim
        device = self.device

        def _extract_one(trajectory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            developer = trajectory.get("developer", trajectory.get("developer_info", {}))
            activity_history = trajectory.get("activity_history", [])
            step_labels = trajectory.get("step_labels", [])
            monthly_histories = trajectory.get("monthly_activity_histories", [])

            if (
                (not activity_history and not monthly_histories)
                or not step_labels
                or not monthly_histories
            ):
                return None

            email = developer.get(
                "email",
                developer.get("developer_id", developer.get("reviewer", "")),
            )
            step_context_dates = trajectory.get("step_context_dates", [])
            step_total_project_reviews = trajectory.get(
                "step_total_project_reviews", []
            )
            path_features_per_step = trajectory.get("path_features_per_step", [])
            event_features_per_step = trajectory.get("event_features", [])

            min_len = min(len(monthly_histories), len(step_labels))
            state_vecs: List[np.ndarray] = []
            action_feat_vecs: List[np.ndarray] = []

            for i in range(min_len):
                month_history = monthly_histories[i]
                if not month_history:
                    state_vecs.append(np.zeros(state_dim, dtype=np.float32))
                    action_feat_vecs.append(np.zeros(action_dim, dtype=np.float32))
                    continue

                if step_context_dates and i < len(step_context_dates):
                    month_context_date = step_context_dates[i]
                else:
                    month_context_date = month_history[-1]["timestamp"]

                total_proj = (
                    step_total_project_reviews[i]
                    if i < len(step_total_project_reviews)
                    else 0
                )
                pf = (
                    path_features_per_step[i]
                    if i < len(path_features_per_step)
                    else None
                )
                ef = (
                    event_features_per_step[i]
                    if i < len(event_features_per_step)
                    else None
                )

                rows: List[Dict[str, Any]] = []
                for act in month_history:
                    ts = act.get("timestamp")
                    if ts is None:
                        continue
                    if act.get("action_type") == "authored":
                        rows.append(
                            {
                                "email": act.get("reviewer_email", ""),
                                "timestamp": pd.Timestamp(ts)
                                if not isinstance(ts, pd.Timestamp)
                                else ts,
                                "label": 0,
                                "owner_email": email,
                                "change_insertions": act.get(
                                    "lines_added", act.get("change_insertions", 0)
                                )
                                or 0,
                                "change_deletions": act.get(
                                    "lines_deleted", act.get("change_deletions", 0)
                                )
                                or 0,
                                "change_files_count": act.get(
                                    "files_changed", act.get("change_files_count", 0)
                                )
                                or 0,
                                "first_response_time": None,
                            }
                        )
                    else:
                        rows.append(
                            {
                                "email": email,
                                "timestamp": pd.Timestamp(ts)
                                if not isinstance(ts, pd.Timestamp)
                                else ts,
                                "label": 1 if act.get("accepted", False) else 0,
                                "owner_email": act.get("owner_email", ""),
                                "change_insertions": act.get(
                                    "lines_added", act.get("change_insertions", 0)
                                )
                                or 0,
                                "change_deletions": act.get(
                                    "lines_deleted", act.get("change_deletions", 0)
                                )
                                or 0,
                                "change_files_count": act.get(
                                    "files_changed", act.get("change_files_count", 0)
                                )
                                or 0,
                                "first_response_time": act.get(
                                    "response_time", act.get("first_response_time")
                                ),
                            }
                        )
                df = pd.DataFrame(rows)

                if len(df) == 0:
                    state_vecs.append(np.zeros(state_dim, dtype=np.float32))
                    action_feat_vecs.append(np.zeros(action_dim, dtype=np.float32))
                    continue

                feature_start = df["timestamp"].min()
                feature_end = pd.Timestamp(month_context_date)
                try:
                    features = extract_common_features(
                        df,
                        email,
                        feature_start,
                        feature_end,
                        normalize=True,
                        total_project_reviews=total_proj,
                    )
                except Exception:
                    state_vecs.append(np.zeros(state_dim, dtype=np.float32))
                    action_feat_vecs.append(np.zeros(action_dim, dtype=np.float32))
                    continue

                sv = [float(features.get(f, 0.0)) for f in STATE_FEATURES]
                if pf is not None:
                    sv.extend(float(v) for v in pf)
                if ef is not None:
                    sv.extend(
                        float(ef.get(k, 0.0))
                        for k in [
                            "event_lines_changed",
                            "event_response_time",
                            "event_accepted",
                            "time_since_prev_event",
                        ]
                    )
                av = [float(features.get(f, 0.0)) for f in ACTION_FEATURES]
                state_vecs.append(np.array(sv, dtype=np.float32))
                action_feat_vecs.append(np.array(av, dtype=np.float32))

            if not state_vecs:
                return None

            return {
                "state_vecs": np.stack(state_vecs),
                "action_feat_vecs": np.stack(action_feat_vecs),
                "min_len": min_len,
                "step_labels": step_labels,
                "sample_weight": trajectory.get("sample_weight", 1.0),
            }

        logger.info(
            "  [%s] %d 軌跡を並列処理中...", split_name, len(trajectories_list)
        )
        raw_results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_extract_one)(t) for t in trajectories_list
        )

        precomputed: List[Dict[str, torch.Tensor]] = []
        for raw in raw_results:
            if raw is None:
                continue
            min_len = raw["min_len"]
            state_seq = torch.tensor(
                raw["state_vecs"], dtype=torch.float32, device=device
            )
            action_seq = torch.tensor(
                raw["action_feat_vecs"], dtype=torch.float32, device=device
            )
            actions = torch.tensor(
                [int(bool(l)) for l in raw["step_labels"][:min_len]],
                dtype=torch.long,
                device=device,
            )
            sample_w = torch.full(
                [min_len], float(raw["sample_weight"]), device=device
            )

            precomputed.append(
                {
                    "state_seq": state_seq,
                    "action_seq": action_seq,
                    "actions": actions,
                    "sample_weights": sample_w,
                    "min_len": min_len,
                }
            )

        return precomputed

    def _collate_batches(
        self,
        precomputed_list: List[Dict[str, torch.Tensor]],
        bs: int,
        shuffle: bool,
        rng: random.Random,
    ) -> List[Dict[str, torch.Tensor]]:
        """[L, dim] エントリ群を [B, L_max, dim] にパディングしてバッチ化。"""
        n = len(precomputed_list)
        order = list(range(n))
        if shuffle:
            rng.shuffle(order)
        batches: List[Dict[str, torch.Tensor]] = []
        if n == 0:
            return batches
        state_dim = precomputed_list[0]["state_seq"].shape[-1]
        action_dim = precomputed_list[0]["action_seq"].shape[-1]

        for start in range(0, n, bs):
            idx_chunk = order[start : start + bs]
            if not idx_chunk:
                continue
            chunk = [precomputed_list[i] for i in idx_chunk]
            B = len(chunk)
            max_len = max(e["min_len"] for e in chunk)

            state_b = torch.zeros(B, max_len, state_dim, device=self.device)
            action_b = torch.zeros(B, max_len, action_dim, device=self.device)
            actions_b = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
            sample_w_b = torch.zeros(B, max_len, device=self.device)
            mask_b = torch.zeros(B, max_len, device=self.device)
            lengths = torch.zeros(B, dtype=torch.long, device=self.device)

            for bi, entry in enumerate(chunk):
                L = entry["min_len"]
                state_b[bi, :L] = entry["state_seq"]
                action_b[bi, :L] = entry["action_seq"]
                actions_b[bi, :L] = entry["actions"]
                sample_w_b[bi, :L] = entry["sample_weights"]
                mask_b[bi, :L] = 1.0
                lengths[bi] = L

            batches.append(
                {
                    "state_seq": state_b,
                    "action_seq": action_b,
                    "lengths": lengths,
                    "actions": actions_b,
                    "sample_weights": sample_w_b,
                    "mask": mask_b,
                }
            )
        return batches

    # -----------------------------------------------------------------
    # 推論: 月次系列から π(a=1|s) = continuation_prob
    # -----------------------------------------------------------------

    def predict_continuation_probability_monthly(
        self,
        developer: Dict[str, Any],
        monthly_activity_histories: List[List[Dict[str, Any]]],
        step_context_dates: List,
        context_date: Optional[datetime] = None,
        step_total_project_reviews: Optional[List[int]] = None,
        step_path_features: Optional[List[np.ndarray]] = None,
        step_event_features: Optional[List[Dict[str, float]]] = None,
        head_index: int = 0,  # API 互換維持 (未使用)
    ) -> Dict[str, Any]:
        """月次シーケンスから最終ステップでの π(a=1|s) を返す。"""
        if context_date is None:
            context_date = datetime.now()

        self.network.eval()

        with torch.no_grad():
            if not monthly_activity_histories:
                return {
                    "continuation_probability": 0.5,
                    "confidence": 0.0,
                    "reasoning": "月次履歴が不足しているため、デフォルト確率を返します",
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
                    "continuation_probability": 0.5,
                    "confidence": 0.0,
                    "reasoning": "有効な月次ステップがありません",
                }

            state_seq = torch.stack(state_tensors).unsqueeze(0)
            action_seq = torch.stack(action_tensors).unsqueeze(0)
            lengths = torch.tensor(
                [len(state_tensors)], dtype=torch.long, device=self.device
            )

            logits = self.network(state_seq, action_seq, lengths)  # [1, num_actions]
            probs = F.softmax(logits, dim=-1)
            cont_prob = float(probs[0, 1].item())
            reward_diff = float((logits[0, 1] - logits[0, 0]).item())

            if (
                self.output_temperature
                and abs(self.output_temperature - 1.0) > 1e-6
            ):
                p = min(max(cont_prob, 1e-6), 1.0 - 1e-6)
                logit = math.log(p / (1.0 - p))
                cont_prob = 1.0 / (
                    1.0 + math.exp(-logit / self.output_temperature)
                )

            confidence = min(abs(cont_prob - 0.5) * 2, 1.0)

            return {
                "continuation_probability": cont_prob,
                "reward_score": reward_diff,
                "confidence": confidence,
                "reasoning": (
                    f"MCE-IRL π(a=1|s) = {cont_prob:.1%} "
                    f"(R(s,1)-R(s,0)={reward_diff:+.3f})"
                ),
                "method": "mce_irl_monthly_sequence",
            }

    # -----------------------------------------------------------------
    # 推論: 単一スナップショット (互換用)
    # -----------------------------------------------------------------

    def predict_continuation_probability_snapshot(
        self,
        developer: Dict[str, Any],
        activity_history: List[Dict[str, Any]],
        context_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """スナップショット 1 ステップで π(a=1|s) を返す。"""
        if context_date is None:
            context_date = datetime.now()

        self.network.eval()

        with torch.no_grad():
            if not activity_history:
                return {
                    "continuation_probability": 0.5,
                    "confidence": 0.0,
                    "reasoning": "活動履歴が不足しているため、デフォルト確率を返します",
                }

            email = developer.get(
                "email",
                developer.get("developer_id", developer.get("reviewer", "")),
            )
            state_tensor, action_tensor = self.extract_features_tensor(
                email, activity_history, context_date
            )
            state_seq = state_tensor.unsqueeze(0).unsqueeze(0)
            action_seq = action_tensor.unsqueeze(0).unsqueeze(0)
            lengths = torch.tensor([1], dtype=torch.long, device=self.device)

            logits = self.network(state_seq, action_seq, lengths)
            probs = F.softmax(logits, dim=-1)
            cont_prob = float(probs[0, 1].item())
            reward_diff = float((logits[0, 1] - logits[0, 0]).item())

            if (
                self.output_temperature
                and abs(self.output_temperature - 1.0) > 1e-6
            ):
                p = min(max(cont_prob, 1e-6), 1.0 - 1e-6)
                logit = math.log(p / (1.0 - p))
                cont_prob = 1.0 / (
                    1.0 + math.exp(-logit / self.output_temperature)
                )
            confidence = abs(cont_prob - 0.5) * 2

            return {
                "continuation_probability": cont_prob,
                "reward_score": reward_diff,
                "confidence": confidence,
                "reasoning": f"MCE-IRL snapshot π(a=1|s) = {cont_prob:.1%}",
                "method": "mce_irl_snapshot",
            }

    # -----------------------------------------------------------------
    # 特徴量重要度 (gradient-based) — 親と同形だが reward logit に対する勾配
    # -----------------------------------------------------------------

    def compute_gradient_importance(
        self,
        trajectories: List[Dict[str, Any]],
        max_samples: int = 200,
    ) -> Dict[str, float]:
        """π(a=1|s) に対する入力特徴量の感度を計算。"""
        self.network.eval()

        all_grads: List[np.ndarray] = []
        used = 0

        for traj in trajectories:
            if used >= max_samples:
                break

            developer = traj.get("developer", traj.get("developer_info", {}))
            activity_history = traj["activity_history"]
            context_date = traj.get("context_date", datetime.now())

            if not activity_history:
                continue

            email = developer.get(
                "email",
                developer.get("developer_id", developer.get("reviewer", "")),
            )

            state_tensors: List[torch.Tensor] = []
            action_tensors: List[torch.Tensor] = []
            for i in range(len(activity_history)):
                step_history = activity_history[: i + 1]
                s_t, a_t = self.extract_features_tensor(email, step_history, context_date)
                state_tensors.append(s_t)
                action_tensors.append(a_t)

            state_seq = (
                torch.stack(state_tensors).unsqueeze(0).requires_grad_(True)
            )
            action_seq = (
                torch.stack(action_tensors).unsqueeze(0).requires_grad_(True)
            )
            lengths = torch.tensor(
                [len(activity_history)], dtype=torch.long, device=self.device
            )

            logits = self.network(state_seq, action_seq, lengths)  # [1, 2]
            log_pi_a1 = F.log_softmax(logits, dim=-1)[0, 1]
            log_pi_a1.backward()

            s_grad = (
                state_seq.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
            )
            a_grad = (
                action_seq.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
            )
            all_grads.append(np.concatenate([s_grad, a_grad]))
            used += 1

        if not all_grads:
            return {}

        mean_grads = np.mean(all_grads, axis=0)
        total = mean_grads.sum()
        if total > 0:
            mean_grads = mean_grads / total

        names = list(STATE_FEATURES) + list(ACTION_FEATURES)
        return {name: float(val) for name, val in zip(names, mean_grads)}


if __name__ == "__main__":
    cfg = {
        "state_dim": 23,
        "action_dim": len(ACTION_FEATURES),
        "hidden_dim": 128,
        "dropout": 0.1,
        "model_type": 0,
        "learning_rate": 3e-4,
    }
    sys = MCEIRLSystem(cfg)
    print("MCE-IRL system initialised")
    print(sys.network)
