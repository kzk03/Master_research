"""
継続予測のための逆強化学習システム

優秀な開発者（継続した開発者）の行動パターンから
継続に寄与する要因の報酬関数を学習し、
それを基に継続確率を予測する。
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# common_features.py の特徴量定義をインポート
from review_predictor.IRL.features.common_features import (
    extract_common_features,
    STATE_FEATURES,
    ACTION_FEATURES,
)

logger = logging.getLogger(__name__)


@dataclass
class DeveloperState:
    """開発者の状態表現（14次元版 - マルチプロジェクト対応）"""
    developer_id: str
    experience_days: int
    total_changes: int
    total_reviews: int
    recent_activity_frequency: float
    avg_activity_gap: float
    activity_trend: str
    collaboration_score: float
    code_quality_score: float
    recent_acceptance_rate: float  # 直近30日のレビュー受諾率
    review_load: float  # レビュー負荷（直近30日 / 平均）
    # マルチプロジェクト対応: 以下4つの特徴量を追加
    project_count: int  # 参加プロジェクト数
    project_activity_distribution: float  # プロジェクト間の活動分散度（0-1）
    main_project_contribution_ratio: float  # メインプロジェクトへの貢献率（0-1）
    cross_project_collaboration_score: float  # プロジェクト横断協力スコア（0-1）
    timestamp: datetime


@dataclass
class DeveloperAction:
    """開発者の行動表現（7次元版 - マルチプロジェクト対応）"""
    action_type: str  # 'commit', 'review', 'merge', 'documentation', etc.
    intensity: float  # 行動の強度（変更ファイル数ベース）
    collaboration: float  # 協力度
    response_time: float   # レスポンス時間（日数）
    review_size: float  # レビュー規模（変更行数）
    # マルチプロジェクト対応: 以下2つの特徴量を追加
    project_id: str  # 行動が発生したプロジェクトID
    is_cross_project: bool  # プロジェクト横断的な行動かどうか
    timestamp: datetime


class RetentionIRLNetwork(nn.Module):
    """
    継続予測のための逆強化学習ニューラルネットワーク (時系列対応版)

    このネットワークは以下の2つを学習します:
    1. 報酬関数: 開発者の状態・行動から、その行動の「継続への寄与度」を予測
    2. 継続確率: 学習した報酬関数をもとに、将来の継続確率を予測

    アーキテクチャ:
        状態エンコーダー: state_dim → 128 → 64
        行動エンコーダー: action_dim → 128 → 64
        LSTM (時系列): hidden_dim=128, 1層
        報酬予測器: 128 → 64 → 1
        継続確率予測器: 128 → 64 → 1 → Sigmoid
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        """
        ネットワークの初期化

        Args:
            state_dim: 状態の次元数（デフォルト: 14次元）
            action_dim: 行動の次元数（デフォルト: 5次元）
            hidden_dim: 隠れ層の次元数（デフォルト: 128）
            dropout: Dropout率（過学習防止、デフォルト: 0.1）
        """
        super().__init__()

        # 状態エンコーダー: state_dim → hidden_dim → hidden_dim//2
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 行動エンコーダー: action_dim → hidden_dim → hidden_dim//2
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM: 状態・行動を連結（hidden_dim//2 + hidden_dim//2 = hidden_dim）して入力
        # 2層LSTMでdropoutが有効化される
        self.lstm = nn.LSTM(
            hidden_dim,      # concat後の次元
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # LSTM出力のLayerNorm（出力安定化）
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # 報酬予測器
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 継続確率予測器
        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ニューラルネットワークの前向き計算 (時系列対応・可変長対応)

        Args:
            state: 開発者状態テンソル [batch_size, seq_len, state_dim]
            action: 開発者行動テンソル [batch_size, seq_len, action_dim]
            lengths: 各シーケンスの実際の長さ [batch_size] (必須)

        Returns:
            reward: 予測報酬スコア [batch_size, 1]
            continuation_prob: 継続確率 [batch_size, 1]
        """
        batch_size, seq_len, _ = state.shape

        # ステップ1: 状態と行動をエンコード
        # 各タイムステップの状態・行動を独立にエンコード
        # view(-1, dim)で2次元に変換 → エンコード → view(batch, seq, dim)で3次元に戻す
        state_encoded = self.state_encoder(state.view(-1, state.shape[-1])).view(batch_size, seq_len, -1)
        action_encoded = self.action_encoder(action.view(-1, action.shape[-1])).view(batch_size, seq_len, -1)

        # ステップ2: 状態と行動を連結（concat → hidden_dim次元）
        combined = torch.cat([state_encoded, action_encoded], dim=-1)

        # ========================================
        # 可変長シーケンス処理: pack_padded_sequenceを使用
        # 異なる長さのシーケンスを効率的に処理
        # PyTorchのLSTMは降順にソートされたシーケンスを要求
        # ========================================
        lengths_cpu = lengths.cpu()
        sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
        _, unsort_idx = sorted_idx.sort()  # 元の順序に戻すためのインデックス

        # ソート後のシーケンスをpack
        combined_sorted = combined[sorted_idx]
        packed = nn.utils.rnn.pack_padded_sequence(
            combined_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
        )

        # LSTMで処理
        lstm_out_packed, _ = self.lstm(packed)

        # unpack して元の順序に戻す
        lstm_out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out_packed, batch_first=True
        )
        lstm_out = lstm_out_sorted[unsort_idx]

        # 各シーケンスの実際の最終ステップを取得
        hidden = torch.zeros(batch_size, lstm_out.size(-1), device=state.device)
        for i in range(batch_size):
            actual_len = lengths[i].item()
            hidden[i] = lstm_out[i, actual_len - 1, :]

        hidden = self.lstm_norm(hidden)  # LayerNorm で安定化

        reward = self.reward_predictor(hidden)
        continuation_prob = self.continuation_predictor(hidden)

        return reward, continuation_prob
    
    def forward_all_steps(self, state: torch.Tensor, action: torch.Tensor,
                          lengths: torch.Tensor,
                          return_reward: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        全ステップで継続確率（と報酬）を予測（可変長対応）

        Args:
            state: [batch_size, max_seq_len, state_dim]
            action: [batch_size, max_seq_len, action_dim]
            lengths: [batch_size] 各シーケンスの実際の長さ (必須)
            return_reward: Trueの場合、報酬も返す

        Returns:
            return_reward=False: predictions [batch_size, max_seq_len] 各ステップの継続確率
            return_reward=True: (reward, continuation) のタプル
        """
        batch_size, max_seq_len, _ = state.shape
        state_encoded = self.state_encoder(state.view(-1, state.shape[-1])).view(batch_size, max_seq_len, -1)
        action_encoded = self.action_encoder(action.view(-1, action.shape[-1])).view(batch_size, max_seq_len, -1)

        combined = torch.cat([state_encoded, action_encoded], dim=-1)

        # 可変長シーケンスの処理
        lengths_cpu = lengths.cpu()
        sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
        _, unsort_idx = sorted_idx.sort()

        combined_sorted = combined[sorted_idx]
        packed = nn.utils.rnn.pack_padded_sequence(
            combined_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
        )
        lstm_out_packed, _ = self.lstm(packed)
        lstm_out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out_packed, batch_first=True, total_length=max_seq_len
        )
        lstm_out = lstm_out_sorted[unsort_idx]
        lstm_out = self.lstm_norm(lstm_out)  # LayerNorm

        # 各ステップで予測
        lstm_out_flat = lstm_out.reshape(-1, lstm_out.size(-1))
        continuation_flat = self.continuation_predictor(lstm_out_flat).squeeze(-1)
        continuation = continuation_flat.view(batch_size, max_seq_len)
        
        if return_reward:
            reward_flat = self.reward_predictor(lstm_out_flat).squeeze(-1)
            reward = reward_flat.view(batch_size, max_seq_len)
            return reward, continuation
        
        return continuation


class RetentionIRLSystem:
    """継続予測IRL システム (拡張: 時系列対応)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # ========================================
        # ネットワーク設定（マルチプロジェクト対応）
        # - 状態次元: 10 → 14次元（プロジェクト特徴量4つ追加）
        # - 行動次元: 4 → 5次元（プロジェクト特徴量1つ追加）
        # ========================================
        self.state_dim = config.get('state_dim', len(STATE_FEATURES))  # 動的に取得
        self.action_dim = config.get('action_dim', len(ACTION_FEATURES))  # 動的に取得
        self.hidden_dim = config.get('hidden_dim', 128)
        self.dropout = config.get('dropout', 0.1)
        # 予測確率の温度スケーリング（1.0で無効、<1でシャープ、>1でフラット）
        self.output_temperature = float(config.get('output_temperature', 1.0))

        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ネットワーク初期化
        self.model_type = config.get('model_type', 0)
        if self.model_type == 0:
            self.network = RetentionIRLNetwork(
                self.state_dim, self.action_dim, self.hidden_dim, self.dropout
            ).to(self.device)
        else:
            from .network_variants import create_network
            self.network = create_network(
                self.model_type, self.state_dim, self.action_dim,
                self.hidden_dim, self.dropout,
            ).to(self.device)
        
        # オプティマイザー
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # LRスケジューラ（エポック数は訓練開始時に設定）
        self._scheduler: Optional[Any] = None
        
        # 損失関数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # Focal Loss のパラメータ（デフォルト値、調整: gamma 2.0 → 1.0）
        self.focal_alpha = 0.25  # クラス重み（ポジティブクラスの重み）
        self.focal_gamma = 1.0   # フォーカスパラメータ（学習安定化のため削減）
        
        logger.info("継続予測IRLシステムを初期化しました")
    
    def set_focal_loss_params(self, alpha: float, gamma: float):
        """
        Focal Loss のパラメータを動的に設定
        
        Args:
            alpha: クラス重み（0～1、小さいほど正例重視）
            gamma: フォーカスパラメータ（0～5、大きいほど難しい例重視）
        """
        self.focal_alpha = alpha
        self.focal_gamma = gamma
        logger.info(f"Focal Loss パラメータ更新: alpha={alpha:.3f}, gamma={gamma:.3f}")
    
    def auto_tune_focal_loss(self, positive_rate: float):
        """
        正例率に応じて Focal Loss パラメータを自動調整
        
        Args:
            positive_rate: 訓練データの正例率（0～1）
        
        調整ロジック（gamma削減: 学習安定化のため）:
        - 正例率が高い（≥0.6）: alpha=0.4, gamma=1.0（バランス重視）
        - 正例率が中程度（0.3～0.6）: alpha=0.3, gamma=1.0（標準）
        - 正例率が低い（<0.3）: alpha=0.25, gamma=1.5（Recall 重視）
        """
        if positive_rate >= 0.6:
            # 正例が多い区間ではほぼバランスに近い重み
            alpha = 0.40
            gamma = 1.0
            strategy = "バランス重視（正例率≥60%・軽い正例優先)"
        elif positive_rate >= 0.3:
            # 標準帯域では適度に正例へ重みを寄せ、precision低下を防ぐ
            alpha = 0.45
            gamma = 1.0
            strategy = "継続重視（正例率30-60%・適度な正例ウェイト)"
        else:
            # 正例が希少な期間は追加で正例を持ち上げるがgammaは抑える
            alpha = 0.55
            gamma = 1.1
            strategy = "継続重視（正例率<30%・正例ウェイト中)"
        
        self.set_focal_loss_params(alpha, gamma)
        logger.info(f"正例率 {positive_rate:.1%} に基づき自動調整: {strategy}")
    
    def focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                   sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Focal Loss の計算（クラス不均衡対策）

        Focal Lossは、クラス不均衡問題に対処するための損失関数です。
        - 簡単な例（正しく予測できている例）の損失を減らす
        - 難しい例（間違って予測している例）の損失を増やす
        - 少数クラス（正例）により多くの重みを与える

        数式: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) * sample_weight

        パラメータ:
        - alpha: クラス重み（0～1、小さいほど正例重視）
          例: alpha=0.25の場合、正例の重み=0.25、負例の重み=0.75
        - gamma: フォーカスパラメータ（0～5、大きいほど難しい例重視）
          例: gamma=1.0の場合、p_t=0.9の例は重みが0.1^1.0=0.1に減少
        - sample_weight: サンプルごとの重み
          例: 依頼なし=0.5、依頼あり=1.0

        Args:
            predictions: 予測確率 [batch_size] or [batch_size, 1]
                         値の範囲: 0～1（Sigmoidの出力）
            targets: ターゲットラベル [batch_size] or [batch_size, 1]
                     値: 0（負例: 離脱）または 1（正例: 継続）
            sample_weights: サンプル重み [batch_size] or None
                           依頼なし（拡張期間のみ依頼あり）=0.1、依頼あり=1.0

        Returns:
            Focal Loss（スカラー値）
        """
        # ========================================
        # ステップ1: テンソルの形状を整える
        # [batch_size, 1] → [batch_size]
        # ========================================
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # ========================================
        # ステップ2: Binary Cross Entropy (BCE) Lossを計算
        # BCE: -[y*log(p) + (1-y)*log(1-p)]
        # ========================================
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')

        # ========================================
        # ステップ3: p_t を計算（正しいクラスの予測確率）
        # p_t = p (y=1の場合) または 1-p (y=0の場合)
        # 例: y=1, p=0.8 → p_t=0.8（正しく予測）
        #     y=0, p=0.2 → p_t=0.8（正しく予測）
        #     y=1, p=0.2 → p_t=0.2（間違って予測）
        # ========================================
        p_t = predictions * targets + (1 - predictions) * (1 - targets)

        # ========================================
        # ステップ4: alpha_t を計算（クラスごとの重み）
        # alpha_t = alpha (y=1の場合) または 1-alpha (y=0の場合)
        # 例: alpha=0.25, y=1 → alpha_t=0.25（正例の重み）
        #     alpha=0.25, y=0 → alpha_t=0.75（負例の重み）
        # ========================================
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)

        # ========================================
        # ステップ5: Focal Lossの重みを計算
        # focal_weight = alpha_t * (1 - p_t)^gamma
        # 例: p_t=0.9, gamma=1.0 → (1-0.9)^1.0 = 0.1（簡単な例は重みが小さい）
        #     p_t=0.2, gamma=1.0 → (1-0.2)^1.0 = 0.8（難しい例は重みが大きい）
        # ========================================
        focal_weight = alpha_t * torch.pow(1 - p_t, self.focal_gamma)
        focal_loss = focal_weight * bce_loss

        # ========================================
        # ステップ6: サンプル重みを適用（オプション）
        # 依頼なし（拡張期間のみ依頼あり）のサンプルは重みを下げる
        # ========================================
        if sample_weights is not None:
            sample_weights = sample_weights.squeeze()
            focal_loss = focal_loss * sample_weights

        # 平均を返す（バッチ全体の損失）
        return focal_loss.mean()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # common_features.py を使った特徴量抽出（v2: マルチプロジェクト特徴量を廃止）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _history_to_df(self, email: str, activity_history: List[Dict]) -> pd.DataFrame:
        """activity_history (list of dicts) を common_features 用 DataFrame に変換"""
        rows = []
        for act in activity_history:
            ts = act.get('timestamp')
            if ts is None:
                continue
            if act.get('action_type') == 'authored':
                # 自分が作成したPR: owner_email=自分, email=レビュアー
                # → owner_data (df[owner_email==email]) にヒットして total_changes / reciprocity_score が計算される
                rows.append({
                    'email': act.get('reviewer_email', ''),
                    'timestamp': pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts,
                    'label': 0,
                    'owner_email': email,  # 自分がオーナー
                    'change_insertions': act.get('lines_added', act.get('change_insertions', 0)) or 0,
                    'change_deletions': act.get('lines_deleted', act.get('change_deletions', 0)) or 0,
                    'change_files_count': act.get('files_changed', act.get('change_files_count', 0)) or 0,
                    'first_response_time': None,
                })
            else:
                rows.append({
                    'email': email,
                    'timestamp': pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts,
                    'label': 1 if act.get('accepted', False) else 0,
                    'owner_email': act.get('owner_email', ''),
                    'change_insertions': act.get('lines_added', act.get('change_insertions', 0)) or 0,
                    'change_deletions': act.get('lines_deleted', act.get('change_deletions', 0)) or 0,
                    'change_files_count': act.get('files_changed', act.get('change_files_count', 0)) or 0,
                    'first_response_time': act.get('response_time', act.get('first_response_time')),
                })
        return pd.DataFrame(rows)

    def extract_features_tensor(
        self,
        email: str,
        activity_history: List[Dict],
        context_date: datetime,
        total_project_reviews: int = 0,
        path_features_vec: Optional[np.ndarray] = None,
        event_features_vec: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        activity_history から common_features.py で STATE / ACTION テンソルを生成。

        Args:
            total_project_reviews: プロジェクト全体のレビュー依頼数（core_reviewer_ratio計算用）
                                   0の場合はdf内のデータから計算（不正確になる場合あり）
            path_features_vec: ディレクトリ固有のパス特徴量（3次元）。
                               指定時は state に連結し��� state_dim=23 にする。
        Returns:
            (state_tensor [state_dim], action_tensor [action_dim])
        """
        df = self._history_to_df(email, activity_history)

        if len(df) == 0:
            return (
                torch.zeros(self.state_dim, device=self.device),
                torch.zeros(self.action_dim, device=self.device),
            )

        feature_start = df['timestamp'].min()
        feature_end = pd.Timestamp(context_date)

        try:
            features = extract_common_features(
                df, email, feature_start, feature_end,
                normalize=True,
                total_project_reviews=total_project_reviews,
            )
        except Exception as e:
            logger.warning(f"common_features 計算エラー ({email}): {e}")
            return (
                torch.zeros(self.state_dim, device=self.device),
                torch.zeros(self.action_dim, device=self.device),
            )

        state_vec = [float(features.get(f, 0.0)) for f in STATE_FEATURES]
        if path_features_vec is not None:
            state_vec.extend(float(v) for v in path_features_vec)
        if event_features_vec is not None:
            state_vec.extend(float(event_features_vec.get(k, 0.0)) for k in [
                'event_lines_changed', 'event_response_time',
                'event_accepted', 'time_since_prev_event',
            ])
        action_vec = [float(features.get(f, 0.0)) for f in ACTION_FEATURES]

        return (
            torch.tensor(state_vec, dtype=torch.float32, device=self.device),
            torch.tensor(action_vec, dtype=torch.float32, device=self.device),
        )

    def extract_developer_state(self,
                               developer: Dict[str, Any],
                               activity_history: List[Dict[str, Any]],
                               context_date: datetime) -> DeveloperState:
        """開発者の状態を抽出（マルチプロジェクト対応版）"""

        # 経験日数
        first_seen = developer.get('first_seen', context_date.isoformat())
        if isinstance(first_seen, str):
            first_date = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
        else:
            first_date = first_seen
        experience_days = (context_date - first_date).days

        # 活動統計
        total_changes = developer.get('changes_authored', 0)
        total_reviews = developer.get('changes_reviewed', 0)

        # ========================================
        # マルチプロジェクト対応: プロジェクト関連の特徴量を計算
        # ========================================

        # プロジェクト数を取得
        projects = developer.get('projects', [])
        project_count = len(projects) if isinstance(projects, list) else 0

        # プロジェクトごとに活動をグループ化
        project_activities = self._group_by_project(activity_history)

        # プロジェクト間の活動分散度（0-1）
        project_activity_distribution = self._calculate_activity_distribution(project_activities)

        # メインプロジェクト（最も活動が多い）への貢献率（0-1）
        main_project_contribution_ratio = self._calculate_main_project_ratio(project_activities)

        # プロジェクト横断協力スコア（0-1）
        cross_project_collaboration_score = self._calculate_cross_project_collaboration(activity_history)

        # ========================================
        # 既存の特徴量計算
        # ========================================

        # 最近の活動パターン
        recent_activities = self._get_recent_activities(activity_history, context_date, days=30)
        recent_activity_frequency = len(recent_activities) / 30.0

        # 活動間隔
        activity_gaps = self._calculate_activity_gaps(activity_history)
        avg_activity_gap = np.mean(activity_gaps) if activity_gaps else 30.0

        # 活動トレンド
        activity_trend = self._analyze_activity_trend(activity_history, context_date)

        # 協力スコア（簡易版）
        collaboration_score = self._calculate_collaboration_score(activity_history)

        # コード品質スコア（簡易版）
        code_quality_score = self._calculate_code_quality_score(activity_history)

        # 最近のレビュー受諾率（直近30日）
        recent_acceptance_rate = self._calculate_recent_acceptance_rate(activity_history, context_date, days=30)

        # レビュー負荷（直近30日 / 平均）
        review_load = self._calculate_review_load(activity_history, context_date, days=30)

        return DeveloperState(
            developer_id=developer.get('developer_id', 'unknown'),
            experience_days=experience_days,
            total_changes=total_changes,
            total_reviews=total_reviews,
            recent_activity_frequency=recent_activity_frequency,
            avg_activity_gap=avg_activity_gap,
            activity_trend=activity_trend,
            collaboration_score=collaboration_score,
            code_quality_score=code_quality_score,
            recent_acceptance_rate=recent_acceptance_rate,
            review_load=review_load,
            # マルチプロジェクト対応: 新しい特徴量を追加
            project_count=project_count,
            project_activity_distribution=project_activity_distribution,
            main_project_contribution_ratio=main_project_contribution_ratio,
            cross_project_collaboration_score=cross_project_collaboration_score,
            timestamp=context_date
        )
    
    def extract_developer_actions(self,
                                activity_history: List[Dict[str, Any]],
                                context_date: datetime) -> List[DeveloperAction]:
        """開発者の行動を抽出（マルチプロジェクト対応版）"""

        actions = []

        for activity in activity_history:
            try:
                # 行動タイプ
                action_type = activity.get('type', 'unknown')

                # 行動の強度（変更ファイル数ベース）
                intensity = self._calculate_action_intensity(activity)

                # 協力度
                collaboration = self._calculate_action_collaboration(activity)

                # レスポンス時間（レビューリクエストから応答までの日数）
                response_time = self._calculate_response_time(activity)

                # レビュー規模（変更行数ベース）
                review_size = self._calculate_review_size(activity)

                # ========================================
                # マルチプロジェクト対応: プロジェクト関連の情報を抽出
                # ========================================

                # プロジェクトID
                project_id = activity.get('project_id', 'unknown')

                # プロジェクト横断的な行動かどうか
                is_cross_project = activity.get('is_cross_project', False)

                # タイムスタンプ
                timestamp_str = activity.get('timestamp', context_date.isoformat())
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str

                actions.append(DeveloperAction(
                    action_type=action_type,
                    intensity=intensity,
                    collaboration=collaboration,
                    response_time=response_time,
                    review_size=review_size,
                    # マルチプロジェクト対応: 新しい特徴量を追加
                    project_id=project_id,
                    is_cross_project=is_cross_project,
                    timestamp=timestamp
                ))

            except Exception as e:
                logger.warning(f"行動抽出エラー: {e}")
                continue

        return actions
    
    def state_to_tensor(self, state: DeveloperState) -> torch.Tensor:
        """状態をテンソルに変換（14次元版 - マルチプロジェクト対応）"""

        # 活動トレンドのエンコーディング
        trend_encoding = {
            'increasing': 1.0,
            'stable': 0.5,
            'decreasing': 0.0,
            'unknown': 0.25
        }

        # 全特徴量を0-1の範囲に正規化（上限でクリップ）
        features = [
            # 既存の特徴量（10次元）
            min(state.experience_days / 730.0, 1.0),  # 2年でキャップ
            min(state.total_changes / 500.0, 1.0),    # 500件でキャップ
            min(state.total_reviews / 500.0, 1.0),    # 500件でキャップ
            min(state.recent_activity_frequency, 1.0), # 既に0-1
            min(state.avg_activity_gap / 60.0, 1.0),  # 60日でキャップ
            trend_encoding.get(state.activity_trend, 0.25), # 既に0-1
            min(state.collaboration_score, 1.0),      # 既に0-1
            min(state.code_quality_score, 1.0),       # 既に0-1
            min(state.recent_acceptance_rate, 1.0),   # 既に0-1（直近30日の受諾率）
            min(state.review_load, 1.0),              # 既に0-1（負荷比率、正規化済み）
            # マルチプロジェクト対応: 新しい特徴量（4次元）
            min(state.project_count / 5.0, 1.0),      # 5プロジェクトでキャップ
            min(state.project_activity_distribution, 1.0),  # 既に0-1（活動分散度）
            min(state.main_project_contribution_ratio, 1.0),  # 既に0-1（メイン貢献率）
            min(state.cross_project_collaboration_score, 1.0)  # 既に0-1（横断協力スコア）
        ]

        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def action_to_tensor(self, action: DeveloperAction) -> torch.Tensor:
        """行動をテンソルに変換（5次元版 - マルチプロジェクト対応）"""

        # レスポンス時間を「素早さ」に変換（0-1の範囲に正規化）
        # response_time が短い（素早い）ほど値が大きくなる
        # 3日でおよそ0.5、即日で1.0に近い値
        response_speed = 1.0 / (1.0 + action.response_time / 3.0)

        # 全特徴量を0-1の範囲に正規化
        features = [
            # 既存の特徴量（4次元）
            min(action.intensity, 1.0),        # 強度（変更ファイル数、0-1）
            min(action.collaboration, 1.0),    # 協力度（0-1）
            min(response_speed, 1.0),          # レスポンス速度（素早いほど大きい、0-1）
            min(action.review_size, 1.0),      # レビュー規模（変更行数、0-1）
            # マルチプロジェクト対応: 新しい特徴量（1次元）
            1.0 if action.is_cross_project else 0.0,  # クロスプロジェクト行動フラグ（0 or 1）
        ]

        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def predict_continuation_probability(self,
                                       developer: Dict[str, Any],
                                       activity_history: List[Dict[str, Any]],
                                       context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        継続確率を予測（時系列対応）

        Args:
            developer: 開発者データ
            activity_history: 活動履歴
            context_date: 基準日

        Returns:
            Dict[str, Any]: 予測結果
        """
        if context_date is None:
            context_date = datetime.now()

        self.network.eval()

        with torch.no_grad():
            if not activity_history:
                return {
                    'continuation_probability': 0.5,
                    'confidence': 0.0,
                    'reasoning': '活動履歴が不足しているため、デフォルト確率を返します'
                }

            email = developer.get('email', developer.get('developer_id', developer.get('reviewer', '')))

            # 各ステップまでの履歴で common_features テンソルを生成
            state_tensors = []
            action_tensors = []
            for i in range(len(activity_history)):
                step_history = activity_history[:i + 1]
                s_t, a_t = self.extract_features_tensor(email, step_history, context_date)
                state_tensors.append(s_t)
                action_tensors.append(a_t)

            state_seq = torch.stack(state_tensors).unsqueeze(0)  # [1, seq_len, state_dim]
            action_seq = torch.stack(action_tensors).unsqueeze(0)  # [1, seq_len, action_dim]

            # 可変長シーケンスとして処理
            lengths = torch.tensor([len(activity_history)], dtype=torch.long, device=self.device)
            predicted_reward, predicted_continuation = self.network(
                state_seq, action_seq, lengths
            )

            continuation_prob = predicted_continuation.item()
            reward_score = predicted_reward.item()

            confidence = min(abs(continuation_prob - 0.5) * 2, 1.0)
            reasoning = f"IRL予測継続確率: {continuation_prob:.1%}"

            return {
                'continuation_probability': continuation_prob,
                'reward_score': reward_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'state_features': state_tensors[-1].tolist(),
            }
    
    def _get_recent_activities(self, 
                             activity_history: List[Dict[str, Any]], 
                             context_date: datetime, 
                             days: int = 30) -> List[Dict[str, Any]]:
        """最近の活動を取得"""
        
        cutoff_date = context_date - timedelta(days=days)
        recent_activities = []
        
        for activity in activity_history:
            try:
                timestamp_str = activity.get('timestamp', context_date.isoformat())
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = timestamp_str
                
                if timestamp >= cutoff_date:
                    recent_activities.append(activity)
            except:
                continue
        
        return recent_activities
    
    def _calculate_activity_gaps(self, activity_history: List[Dict[str, Any]]) -> List[float]:
        """活動間隔を計算"""
        
        timestamps = []
        for activity in activity_history:
            try:
                timestamp_str = activity.get('timestamp')
                if timestamp_str:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timestamp_str
                    timestamps.append(timestamp)
            except:
                continue
        
        if len(timestamps) < 2:
            return []
        
        timestamps.sort()
        gaps = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).days
            gaps.append(gap)
        
        return gaps
    
    def _analyze_activity_trend(self, 
                              activity_history: List[Dict[str, Any]], 
                              context_date: datetime) -> str:
        """活動トレンドを分析"""
        
        # 最近30日と過去30-60日を比較
        recent_activities = self._get_recent_activities(activity_history, context_date, 30)
        past_activities = self._get_recent_activities(activity_history, context_date - timedelta(days=30), 30)
        
        recent_count = len(recent_activities)
        past_count = len(past_activities)
        
        if past_count == 0:
            return 'unknown'
        
        ratio = recent_count / past_count
        
        if ratio > 1.2:
            return 'increasing'
        elif ratio < 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_collaboration_score(self, activity_history: List[Dict[str, Any]]) -> float:
        """協力スコアを計算（簡易版）"""
        
        collaboration_activities = ['review', 'merge', 'collaboration', 'mentoring']
        total_activities = len(activity_history)
        
        if total_activities == 0:
            return 0.0
        
        collaboration_count = sum(
            1 for activity in activity_history 
            if activity.get('type', '').lower() in collaboration_activities
        )
        
        return collaboration_count / total_activities
    
    def _calculate_code_quality_score(self, activity_history: List[Dict[str, Any]]) -> float:
        """コード品質スコアを計算（簡易版）"""
        
        quality_indicators = ['test', 'documentation', 'refactor', 'fix']
        total_activities = len(activity_history)
        
        if total_activities == 0:
            return 0.5
        
        quality_count = 0
        for activity in activity_history:
            message = activity.get('message', '').lower()
            if any(indicator in message for indicator in quality_indicators):
                quality_count += 1
        
        return min(quality_count / total_activities + 0.3, 1.0)
    
    def _calculate_recent_acceptance_rate(self, activity_history: List[Dict[str, Any]], 
                                         context_date: datetime, days: int = 30) -> float:
        """
        直近N日のレビュー受諾率を計算
        
        Args:
            activity_history: 活動履歴
            context_date: 基準日
            days: 対象期間（日数）
        
        Returns:
            受諾率（0.0～1.0）、依頼がない場合は0.5（中立）
        """
        cutoff_date = context_date - timedelta(days=days)
        
        # 直近の活動のみフィルタ
        recent_activities = [
            activity for activity in activity_history
            if activity.get('timestamp', context_date) >= cutoff_date
        ]
        
        if not recent_activities:
            return 0.5  # データなし → 中立
        
        # レビュー依頼とその受諾を集計
        review_requests = 0
        accepted_reviews = 0
        
        for activity in recent_activities:
            # レビュー関連の活動かチェック
            if activity.get('action_type') == 'review' or 'review' in activity.get('type', '').lower():
                review_requests += 1
                # 受諾したかチェック
                if activity.get('accepted', False):
                    accepted_reviews += 1
        
        if review_requests == 0:
            return 0.5  # レビュー依頼なし → 中立
        
        return accepted_reviews / review_requests
    
    def _calculate_review_load(self, activity_history: List[Dict[str, Any]], 
                              context_date: datetime, days: int = 30) -> float:
        """
        レビュー負荷を計算（直近N日の依頼数 / 平均依頼数）
        
        Args:
            activity_history: 活動履歴
            context_date: 基準日
            days: 対象期間（日数）
        
        Returns:
            負荷比率（0.0～）、1.0が平均、>1.0が過負荷
        """
        cutoff_date = context_date - timedelta(days=days)
        
        # 直近の活動のみフィルタ
        recent_activities = [
            activity for activity in activity_history
            if activity.get('timestamp', context_date) >= cutoff_date
        ]
        
        # 直近のレビュー依頼数
        recent_requests = sum(
            1 for activity in recent_activities
            if activity.get('action_type') == 'review' or 'review' in activity.get('type', '').lower()
        )
        
        # 全期間の平均依頼数を計算（月あたり）
        if not activity_history:
            return 0.0
        
        total_requests = sum(
            1 for activity in activity_history
            if activity.get('action_type') == 'review' or 'review' in activity.get('type', '').lower()
        )
        
        # 活動履歴の期間を計算
        timestamps = [activity.get('timestamp', context_date) for activity in activity_history]
        if len(timestamps) < 2:
            return 1.0  # データ不足 → 標準負荷
        
        period_days = (max(timestamps) - min(timestamps)).days + 1
        avg_requests_per_period = total_requests / max(period_days / days, 1.0)
        
        if avg_requests_per_period == 0:
            return 0.0
        
        # 負荷比率を正規化（0-1の範囲にクリップ）
        load_ratio = recent_requests / avg_requests_per_period
        return min(load_ratio / 2.0, 1.0)  # 2倍の負荷を1.0にマッピング
    
    def _calculate_action_intensity(self, activity: Dict[str, Any]) -> float:
        """行動の強度を計算（変更ファイル数ベース）"""
        
        files_changed = activity.get('files_changed', 0)
        
        # 変更ファイル数で強度を計算（正規化）
        intensity = min(files_changed / 20.0, 1.0)  # 20ファイルで1.0
        return max(intensity, 0.0)
    
    def _calculate_action_collaboration(self, activity: Dict[str, Any]) -> float:
        """行動の協力度を計算"""
        
        action_type = activity.get('type', '').lower()
        collaboration_types = {
            'review': 0.8,
            'merge': 0.7,
            'collaboration': 1.0,
            'mentoring': 0.9,
            'documentation': 0.6
        }
        
        return collaboration_types.get(action_type, 0.3)
    
    def _calculate_review_size(self, activity: Dict[str, Any]) -> float:
        """レビュー規模を計算（変更行数ベース）"""
        
        lines_added = activity.get('lines_added', 0)
        lines_deleted = activity.get('lines_deleted', 0)
        total_lines = lines_added + lines_deleted
        
        # 変更行数で規模を計算（正規化）
        review_size = min(total_lines / 500.0, 1.0)  # 500行で1.0
        return max(review_size, 0.0)
    
    def _calculate_response_time(self, activity: Dict[str, Any]) -> float:
        """レスポンス時間を計算（日数）"""
        
        # レビューリクエストの作成日時と応答日時の差を計算
        request_time = activity.get('request_time')
        response_time = activity.get('response_time')  # 変更: timestamp → response_time
        
        if request_time and response_time:
            try:
                if isinstance(request_time, str):
                    request_dt = datetime.fromisoformat(request_time.replace('Z', '+00:00'))
                else:
                    request_dt = request_time
                    
                if isinstance(response_time, str):
                    response_dt = datetime.fromisoformat(response_time.replace('Z', '+00:00'))
                else:
                    response_dt = response_time
                
                # 日数で計算
                days_diff = (response_dt - request_dt).days
                return max(0.0, days_diff)  # 負の値は0に
                
            except Exception:
                return 14.0  # 最大レスポンス時間（未応答/非アクティブ）
        
        # デフォルト値（データがない場合）: 未応答を表す最大値を使用
        return 14.0  # 最大レスポンス時間（未応答/非アクティブ）
    
    def _group_by_project(self, activity_history: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        活動履歴をプロジェクトごとにグループ化

        Args:
            activity_history: 活動履歴のリスト

        Returns:
            プロジェクトIDをキーとした活動履歴の辞書
        """
        project_activities: Dict[str, List[Dict[str, Any]]] = {}

        for activity in activity_history:
            project_id = activity.get('project_id', 'unknown')
            if project_id not in project_activities:
                project_activities[project_id] = []
            project_activities[project_id].append(activity)

        return project_activities

    def _calculate_activity_distribution(self, project_activities: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        プロジェクト間の活動分散度を計算（0-1の範囲）

        活動が1つのプロジェクトに集中している場合は0に近く、
        複数のプロジェクトに均等に分散している場合は1に近い値を返す。

        Args:
            project_activities: プロジェクトごとの活動履歴

        Returns:
            分散度（0.0-1.0）
        """
        if len(project_activities) <= 1:
            return 0.0  # プロジェクトが1つ以下なら分散なし

        # 各プロジェクトの活動数を取得
        counts = [len(activities) for activities in project_activities.values()]
        total = sum(counts)

        if total == 0:
            return 0.0

        # 標準偏差を使った分散度（正規化版）
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        # 標準偏差を平均で割って正規化（変動係数）
        # 0に近い = 均等分散、1に近い = 偏った分散
        coefficient_of_variation = std_count / (mean_count + 1e-6)

        # 0-1の範囲に正規化（CV=1.0を最大分散とする）
        distribution_score = min(coefficient_of_variation, 1.0)

        return distribution_score

    def _calculate_main_project_ratio(self, project_activities: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        メインプロジェクト（最も活動が多い）への貢献率を計算

        Args:
            project_activities: プロジェクトごとの活動履歴

        Returns:
            メインプロジェクトへの貢献率（0.0-1.0）
        """
        if not project_activities:
            return 0.0

        # 各プロジェクトの活動数を取得
        counts = [len(activities) for activities in project_activities.values()]
        total = sum(counts)

        if total == 0:
            return 0.0

        # 最も活動が多いプロジェクトの活動数
        max_count = max(counts)

        # 全体に対する割合
        return max_count / total

    def _calculate_cross_project_collaboration(self, activity_history: List[Dict[str, Any]]) -> float:
        """
        プロジェクト横断的な協力スコアを計算

        異なるプロジェクトのメンバーとのレビューや協力活動を検出し、
        プロジェクトを跨いだ協力の度合いを0-1で返す。

        Args:
            activity_history: 活動履歴のリスト

        Returns:
            クロスプロジェクト協力スコア（0.0-1.0）
        """
        if not activity_history:
            return 0.0

        # プロジェクト横断的な活動をカウント
        cross_project_count = 0

        for activity in activity_history:
            # is_cross_projectフラグがある場合はそれを使用
            if activity.get('is_cross_project', False):
                cross_project_count += 1
            # または、複数のproject_idsが関連している場合
            elif 'related_projects' in activity:
                related = activity.get('related_projects', [])
                if isinstance(related, list) and len(related) > 1:
                    cross_project_count += 1

        # 全活動に対する割合
        return cross_project_count / len(activity_history)

    def _generate_irl_reasoning(self, 
                              state: DeveloperState, 
                              action: DeveloperAction, 
                              continuation_prob: float,
                              reward_score: float) -> str:
        """IRL予測の理由を生成"""
        
        reasoning_parts = []
        
        # 経験レベル
        if state.experience_days > 365:
            reasoning_parts.append("豊富な経験により継続確率が向上")
        elif state.experience_days < 90:
            reasoning_parts.append("経験が浅いため継続確率がやや低下")
        
        # 活動パターン
        if state.recent_activity_frequency > 0.1:
            reasoning_parts.append("高い活動頻度により継続確率が向上")
        elif state.recent_activity_frequency < 0.03:
            reasoning_parts.append("低い活動頻度により継続確率が低下")
        
        # 協力度
        if state.collaboration_score > 0.5:
            reasoning_parts.append("高い協力度により継続確率が向上")
        
        # 最近の行動（属性チェック）
        if hasattr(action, 'quality') and action.quality > 0.7:
            reasoning_parts.append("高品質な最近の行動により継続確率が向上")
        
        # 報酬スコア
        if reward_score > 0.7:
            reasoning_parts.append("学習された報酬関数により高い継続価値を予測")
        elif reward_score < 0.3:
            reasoning_parts.append("学習された報酬関数により低い継続価値を予測")
        
        reasoning_parts.append(f"IRL予測継続確率: {continuation_prob:.1%}")
        
        return "。".join(reasoning_parts)
    
    def train_irl_temporal_trajectories(self,
                                       expert_trajectories: List[Dict[str, Any]],
                                       epochs: int = 50,
                                       patience: int = 5,
                                       val_ratio: float = 0.2,
                                       batch_size: int = 32) -> Dict[str, Any]:
        """
        時系列軌跡データを用いた逆強化学習（IRL）モデルの訓練

        訓練の流れ:
        1. 各レビュアーの月次活動履歴から状態・行動シーケンスを構築
        2. LSTMで時系列パターンを学習
        3. 各月時点での継続確率を予測
        4. Focal Lossで損失を計算（クラス不均衡対策）
        5. バックプロパゲーションで重みを更新

        軌跡データの構造:
        {
            'developer_info': {...},                    # 開発者の基本情報
            'activity_history': [...],                  # 全期間の活動履歴
            'monthly_activity_histories': [[...], ...], # 各月時点の活動履歴（LSTM用）
            'step_labels': [0, 1, 1, 0, ...],          # 各月の継続ラベル（0=離脱, 1=継続）
            'sample_weight': 1.0 or 0.1                # サンプル重み（依頼あり=1.0, なし=0.1）
        }

        実装の詳細:
        - 各月時点での状態を動的に計算（その月までの履歴のみを使用）
        - データリーク防止: 将来のデータは一切使用しない
        - 可変長シーケンスに対応（pack_padded_sequence使用）
        - 月次集約ラベル: 各月から将来窓を見て継続判定

        Args:
            expert_trajectories: 時系列軌跡データのリスト
                                 各要素は1レビュアーの時系列データ
            epochs: 訓練エポック数（デフォルト: 50）

        Returns:
            訓練結果を含む辞書
            {
                'training_losses': [float, ...],  # 各エポックの損失
                'final_loss': float,              # 最終エポックの損失
                'epochs_trained': int             # 訓練したエポック数
            }
        """
        # ── train / val 分割 ──
        import random
        import copy
        indices = list(range(len(expert_trajectories)))
        random.seed(42)
        random.shuffle(indices)
        val_size = max(1, int(len(indices) * val_ratio))
        val_indices = set(indices[:val_size])
        train_trajectories = [t for i, t in enumerate(expert_trajectories) if i not in val_indices]
        val_trajectories = [t for i, t in enumerate(expert_trajectories) if i in val_indices]

        logger.info("=" * 60)
        logger.info("時系列IRL訓練開始")
        logger.info(f"軌跡数: {len(expert_trajectories)} (train={len(train_trajectories)}, val={len(val_trajectories)})")
        logger.info(f"エポック数: {epochs} (patience={patience})")
        logger.info("=" * 60)

        # CosineAnnealingWarmRestarts: T_0ごとにLRをリセットして局所最小から脱出
        t0 = max(epochs // 4, 10)
        self._scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t0, T_mult=1, eta_min=1e-5
        )

        # ── 特徴量の事前計算（全軌跡・全ステップ、並列化） ──
        # エポック間で不変なので1回だけ計算してキャッシュ
        from .network_variants import is_multitask
        from joblib import Parallel, delayed
        import multiprocessing
        n_cpus = multiprocessing.cpu_count()
        logger.info(f"特徴量を事前計算中... (CPUs={n_cpus})")

        def _precompute_features(trajectories_list, split_name=""):
            """軌跡リストから事前計算済みテンソルのリストを返す（joblib並列）。"""
            state_dim = self.state_dim
            action_dim = self.action_dim
            device = self.device

            # CPU重い部分: trajectory → numpy配列のリストを返す（self不使用）
            def _extract_one(trajectory):
                developer = trajectory.get('developer', trajectory.get('developer_info', {}))
                activity_history = trajectory.get('activity_history', [])
                step_labels = trajectory.get('step_labels', [])
                monthly_histories = trajectory.get('monthly_activity_histories', [])

                if (not activity_history and not monthly_histories) or not step_labels or not monthly_histories:
                    return None

                email = developer.get('email', developer.get('developer_id', developer.get('reviewer', '')))
                step_context_dates = trajectory.get('step_context_dates', [])
                step_total_project_reviews = trajectory.get('step_total_project_reviews', [])
                path_features_per_step = trajectory.get('path_features_per_step', [])
                event_features_per_step = trajectory.get('event_features', [])

                min_len = min(len(monthly_histories), len(step_labels))
                state_vecs = []
                action_vecs = []

                for i in range(min_len):
                    month_history = monthly_histories[i]
                    if not month_history:
                        state_vecs.append(np.zeros(state_dim, dtype=np.float32))
                        action_vecs.append(np.zeros(action_dim, dtype=np.float32))
                        continue

                    if step_context_dates and i < len(step_context_dates):
                        month_context_date = step_context_dates[i]
                    else:
                        month_context_date = month_history[-1]['timestamp']

                    total_proj = step_total_project_reviews[i] if i < len(step_total_project_reviews) else 0
                    pf = path_features_per_step[i] if i < len(path_features_per_step) else None
                    ef = event_features_per_step[i] if i < len(event_features_per_step) else None

                    # _history_to_df のインライン版（self不使用）
                    rows = []
                    for act in month_history:
                        ts = act.get('timestamp')
                        if ts is None:
                            continue
                        if act.get('action_type') == 'authored':
                            rows.append({
                                'email': act.get('reviewer_email', ''),
                                'timestamp': pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts,
                                'label': 0,
                                'owner_email': email,
                                'change_insertions': act.get('lines_added', act.get('change_insertions', 0)) or 0,
                                'change_deletions': act.get('lines_deleted', act.get('change_deletions', 0)) or 0,
                                'change_files_count': act.get('files_changed', act.get('change_files_count', 0)) or 0,
                                'first_response_time': None,
                            })
                        else:
                            rows.append({
                                'email': email,
                                'timestamp': pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts,
                                'label': 1 if act.get('accepted', False) else 0,
                                'owner_email': act.get('owner_email', ''),
                                'change_insertions': act.get('lines_added', act.get('change_insertions', 0)) or 0,
                                'change_deletions': act.get('lines_deleted', act.get('change_deletions', 0)) or 0,
                                'change_files_count': act.get('files_changed', act.get('change_files_count', 0)) or 0,
                                'first_response_time': act.get('response_time', act.get('first_response_time')),
                            })
                    df = pd.DataFrame(rows)

                    if len(df) == 0:
                        state_vecs.append(np.zeros(state_dim, dtype=np.float32))
                        action_vecs.append(np.zeros(action_dim, dtype=np.float32))
                        continue

                    feature_start = df['timestamp'].min()
                    feature_end = pd.Timestamp(month_context_date)
                    try:
                        features = extract_common_features(
                            df, email, feature_start, feature_end,
                            normalize=True,
                            total_project_reviews=total_proj,
                        )
                    except Exception:
                        state_vecs.append(np.zeros(state_dim, dtype=np.float32))
                        action_vecs.append(np.zeros(action_dim, dtype=np.float32))
                        continue

                    sv = [float(features.get(f, 0.0)) for f in STATE_FEATURES]
                    if pf is not None:
                        sv.extend(float(v) for v in pf)
                    if ef is not None:
                        sv.extend(float(ef.get(k, 0.0)) for k in [
                            'event_lines_changed', 'event_response_time',
                            'event_accepted', 'time_since_prev_event',
                        ])
                    av = [float(features.get(f, 0.0)) for f in ACTION_FEATURES]
                    state_vecs.append(np.array(sv, dtype=np.float32))
                    action_vecs.append(np.array(av, dtype=np.float32))

                if not state_vecs:
                    return None

                return {
                    'state_vecs': np.stack(state_vecs),    # [L, state_dim]
                    'action_vecs': np.stack(action_vecs),  # [L, action_dim]
                    'min_len': min_len,
                    'step_labels': step_labels,
                    'sample_weight': trajectory.get('sample_weight', 1.0),
                    'step_labels_per_window': trajectory.get('step_labels_per_window'),
                }

            # joblib で並列実行（CPU重い部分のみ）
            logger.info(f"  [{split_name}] {len(trajectories_list)} 軌跡を並列処理中...")
            raw_results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
                delayed(_extract_one)(t) for t in trajectories_list
            )

            # numpy → torch テンソルに変換（[L, dim] / [L] 形状で保持、バッチ次元はcollate時に追加）
            precomputed = []
            for raw in raw_results:
                if raw is None:
                    continue
                min_len = raw['min_len']
                state_seq = torch.tensor(raw['state_vecs'], dtype=torch.float32, device=device)   # [L, state_dim]
                action_seq = torch.tensor(raw['action_vecs'], dtype=torch.float32, device=device) # [L, action_dim]
                targets = torch.tensor(
                    [1.0 if l else 0.0 for l in raw['step_labels'][:min_len]],
                    device=device,
                )                                                                                  # [L]
                sample_w = torch.full([min_len], raw['sample_weight'], device=device)              # [L]

                entry = {
                    'state_seq': state_seq,
                    'action_seq': action_seq,
                    'targets': targets,
                    'sample_weights': sample_w,
                    'min_len': min_len,
                    'step_labels': raw['step_labels'],
                }
                if raw.get('step_labels_per_window'):
                    entry['step_labels_per_window'] = raw['step_labels_per_window']
                precomputed.append(entry)

            return precomputed

        train_data = _precompute_features(train_trajectories, "train")
        val_data = _precompute_features(val_trajectories, "val")
        logger.info(f"特徴量事前計算完了: train={len(train_data)}, val={len(val_data)}")
        total_train_steps = sum(e['min_len'] for e in train_data)
        total_val_steps = sum(e['min_len'] for e in val_data)
        logger.info(f"総ステップ数: train={total_train_steps}, val={total_val_steps} (batch_size={batch_size})")

        # ── ミニバッチ collate ヘルパー ──
        def _collate_batches(precomputed_list, bs, shuffle, rng):
            """[L, dim] エントリ群をパディングして [B, L_max, dim] のバッチにまとめる。"""
            n = len(precomputed_list)
            order = list(range(n))
            if shuffle:
                rng.shuffle(order)
            batches = []
            state_dim = precomputed_list[0]['state_seq'].shape[-1] if n > 0 else self.state_dim
            action_dim = precomputed_list[0]['action_seq'].shape[-1] if n > 0 else self.action_dim
            for start in range(0, n, bs):
                idx_chunk = order[start:start + bs]
                if not idx_chunk:
                    continue
                chunk = [precomputed_list[i] for i in idx_chunk]
                B = len(chunk)
                max_len = max(e['min_len'] for e in chunk)

                state_b = torch.zeros(B, max_len, state_dim, device=self.device)
                action_b = torch.zeros(B, max_len, action_dim, device=self.device)
                targets_b = torch.zeros(B, max_len, device=self.device)
                sample_w_b = torch.zeros(B, max_len, device=self.device)
                mask_b = torch.zeros(B, max_len, device=self.device)
                lengths = torch.zeros(B, dtype=torch.long, device=self.device)
                head_label_b = {}

                for bi, entry in enumerate(chunk):
                    L = entry['min_len']
                    state_b[bi, :L] = entry['state_seq']
                    action_b[bi, :L] = entry['action_seq']
                    targets_b[bi, :L] = entry['targets']
                    sample_w_b[bi, :L] = entry['sample_weights']
                    mask_b[bi, :L] = 1.0
                    lengths[bi] = L

                    slpw = entry.get('step_labels_per_window')
                    if slpw:
                        for head_idx in range(4):
                            window_key = f"{head_idx * 3}-{head_idx * 3 + 3}"
                            head_labels = slpw.get(window_key, entry['step_labels'][:L])
                            if head_idx not in head_label_b:
                                head_label_b[head_idx] = torch.zeros(B, max_len, device=self.device)
                            head_label_b[head_idx][bi, :L] = torch.tensor(
                                [1.0 if l else 0.0 for l in head_labels[:L]],
                                dtype=torch.float32, device=self.device,
                            )

                batches.append({
                    'state_seq': state_b,
                    'action_seq': action_b,
                    'lengths': lengths,
                    'targets': targets_b,
                    'sample_weights': sample_w_b,
                    'mask': mask_b,
                    'head_labels': head_label_b if head_label_b else None,
                })
            return batches

        def _masked_focal_loss(predictions, targets, sample_weights, mask):
            """マスク有り Focal Loss（パディング位置は重み0で除外）。"""
            pred = predictions.reshape(-1)
            targ = targets.reshape(-1)
            sw = sample_weights.reshape(-1)
            mk = mask.reshape(-1)
            bce = F.binary_cross_entropy(pred, targ, reduction='none')
            p_t = pred * targ + (1 - pred) * (1 - targ)
            alpha_t = self.focal_alpha * targ + (1 - self.focal_alpha) * (1 - targ)
            focal_w = alpha_t * torch.pow(1 - p_t, self.focal_gamma)
            loss = focal_w * bce * sw * mk
            denom = mk.sum().clamp(min=1.0)
            return loss.sum() / denom

        def _masked_mse(predictions, targets, mask):
            mse = (predictions.reshape(-1) - targets.reshape(-1)) ** 2
            mk = mask.reshape(-1)
            denom = mk.sum().clamp(min=1.0)
            return (mse * mk).sum() / denom

        rng = random.Random(42)

        self.network.train()
        training_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        best_state_dict = None
        patience_counter = 0

        # validation バッチは固定（毎エポック同じ順序）
        val_batches = _collate_batches(val_data, batch_size, shuffle=False, rng=rng)

        for epoch in range(epochs):
            self.network.train()
            epoch_loss = 0.0
            batch_count = 0

            train_batches = _collate_batches(train_data, batch_size, shuffle=True, rng=rng)
            for batch in train_batches:
                try:
                    predicted_reward, predicted_continuation = self.network.forward_all_steps(
                        batch['state_seq'], batch['action_seq'], batch['lengths'], return_reward=True
                    )

                    targets = batch['targets']
                    sample_weights = batch['sample_weights']
                    mask = batch['mask']
                    reward_targets = targets * 2.0 - 1.0

                    if self.model_type != 0 and is_multitask(self.model_type):
                        head_labels = batch['head_labels'] or {}
                        continuation_loss = torch.tensor(0.0, device=self.device)
                        n_active_heads = 0
                        for head_idx, head_pred in predicted_continuation.items():
                            if head_idx in head_labels:
                                continuation_loss = continuation_loss + _masked_focal_loss(
                                    head_pred, head_labels[head_idx], sample_weights, mask
                                )
                                n_active_heads += 1
                        if n_active_heads > 0:
                            continuation_loss = continuation_loss / n_active_heads
                        reward_loss = _masked_mse(predicted_reward, reward_targets, mask)
                    else:
                        continuation_loss = _masked_focal_loss(
                            predicted_continuation, targets, sample_weights, mask
                        )
                        reward_loss = _masked_mse(predicted_reward, reward_targets, mask)

                    loss_per_batch = continuation_loss + reward_loss

                    self.optimizer.zero_grad()
                    loss_per_batch.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    epoch_loss += loss_per_batch.item()
                    batch_count += 1

                except Exception as e:
                    logger.warning(f"バッチ処理エラー: {e}")
                    continue

            avg_loss = epoch_loss / max(batch_count, 1)
            training_losses.append(avg_loss)

            if self._scheduler is not None:
                self._scheduler.step(epoch)

            # ── Validation loss ──
            self.network.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_batches:
                    try:
                        predicted_reward, predicted_continuation = self.network.forward_all_steps(
                            batch['state_seq'], batch['action_seq'], batch['lengths'], return_reward=True
                        )
                        targets = batch['targets']
                        sample_weights = batch['sample_weights']
                        mask = batch['mask']
                        reward_targets = targets * 2.0 - 1.0

                        if self.model_type != 0 and is_multitask(self.model_type):
                            head_labels = batch['head_labels'] or {}
                            c_loss = torch.tensor(0.0, device=self.device)
                            n_heads = 0
                            for hi, hp in predicted_continuation.items():
                                if hi in head_labels:
                                    c_loss = c_loss + _masked_focal_loss(
                                        hp, head_labels[hi], sample_weights, mask
                                    )
                                    n_heads += 1
                            if n_heads > 0:
                                c_loss = c_loss / n_heads
                            r_loss = _masked_mse(predicted_reward, reward_targets, mask)
                        else:
                            c_loss = _masked_focal_loss(
                                predicted_continuation, targets, sample_weights, mask
                            )
                            r_loss = _masked_mse(predicted_reward, reward_targets, mask)
                        val_loss += (c_loss + r_loss).item()
                        val_count += 1
                    except Exception:
                        continue
            avg_val_loss = val_loss / max(val_count, 1)
            val_losses.append(avg_val_loss)

            # ── Early stopping check ──
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.network.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if True:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"エポック {epoch}: train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}, LR={current_lr:.6f}, patience={patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info(f"Early stopping: epoch {epoch}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
                break

        # ── best モデルを復元 ──
        if best_state_dict is not None:
            self.network.load_state_dict(best_state_dict)
            logger.info(f"ベストモデル復元: epoch {best_epoch}, val_loss={best_val_loss:.4f}")

        logger.info("時系列IRL訓練完了")

        return {
            'training_losses': training_losses,
            'val_losses': val_losses,
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epochs_trained': epoch + 1 if training_losses else 0
        }
    
    def predict_continuation_probability_snapshot(self,
                                                developer: Dict[str, Any],
                                                activity_history: List[Dict[str, Any]],
                                                context_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        スナップショット特徴量で継続確率を予測
        
        Args:
            developer: 開発者データ
            activity_history: 活動履歴
            context_date: 基準日
            
        Returns:
            予測結果
        """
        if context_date is None:
            context_date = datetime.now()
        
        self.network.eval()
        
        with torch.no_grad():
            if not activity_history:
                return {
                    'continuation_probability': 0.5,
                    'confidence': 0.0,
                    'reasoning': '活動履歴が不足しているため、デフォルト確率を返します'
                }

            email = developer.get('email', developer.get('developer_id', developer.get('reviewer', '')))
            state_tensor, action_tensor = self.extract_features_tensor(email, activity_history, context_date)

            # 3次元テンソルに変換（seq_len=1）
            state_seq = state_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
            action_seq = action_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
            lengths = torch.tensor([1], dtype=torch.long, device=self.device)

            # 予測
            predicted_reward, predicted_continuation = self.network(
                state_seq, action_seq, lengths
            )
            
            continuation_prob = predicted_continuation.item()
            # 温度スケーリングで確率分布の広がりを調整
            if self.output_temperature and abs(self.output_temperature - 1.0) > 1e-6:
                p = min(max(continuation_prob, 1e-6), 1.0 - 1e-6)
                # ロジットに変換して温度で割る（T<1でシャープ、T>1でフラット）
                import math
                logit = math.log(p / (1.0 - p))
                scaled_logit = logit / self.output_temperature
                continuation_prob = 1.0 / (1.0 + math.exp(-scaled_logit))
            confidence = abs(continuation_prob - 0.5) * 2
            
            # 理由生成
            reasoning = self._generate_snapshot_reasoning(
                developer, activity_history, continuation_prob, context_date
            )
            
            return {
                'continuation_probability': continuation_prob,
                'confidence': confidence,
                'reasoning': reasoning,
                'method': 'snapshot_features',
                'state_features': state_tensor.tolist(),
                'action_features': action_tensor.tolist()
            }
    
    def _generate_snapshot_reasoning(self,
                                   developer: Dict[str, Any],
                                   activity_history: List[Dict[str, Any]],
                                   continuation_prob: float,
                                   context_date: datetime) -> str:
        """スナップショット特徴量に基づく理由生成"""
        reasoning_parts = []
        
        # 活動履歴の分析
        if len(activity_history) > 10:
            reasoning_parts.append("豊富な活動履歴")
        elif len(activity_history) > 5:
            reasoning_parts.append("適度な活動履歴")
        else:
            reasoning_parts.append("限定的な活動履歴")
        
        # 継続確率に基づく判断
        if continuation_prob > 0.7:
            reasoning_parts.append("高い継続可能性")
        elif continuation_prob > 0.3:
            reasoning_parts.append("中程度の継続可能性")
        else:
            reasoning_parts.append("低い継続可能性")
        
        return f"スナップショット特徴量分析: {', '.join(reasoning_parts)}"


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
    ) -> Dict[str, Any]:
        """
        月次/イベント シーケンスを使った時系列予測（卒論設計準拠）。

        訓練と同じステップをLSTMに入力し、最終ステップの出力で予測する。

        Args:
            developer: 開発者情報
            monthly_activity_histories: 各月時点の累積活動履歴リスト
            step_context_dates: 各月ステップの基準日（month_end）リスト
            context_date: 最終基準日（フォールバック用）
            step_path_features: 各月ステップのパス特徴量（3次元）リスト

        Returns:
            予測結果 dict
        """
        if context_date is None:
            context_date = datetime.now()

        self.network.eval()

        with torch.no_grad():
            if not monthly_activity_histories:
                return {
                    'continuation_probability': 0.5,
                    'confidence': 0.0,
                    'reasoning': '月次履歴が不足しているため、デフォルト確率を返します'
                }

            email = developer.get('email', developer.get('developer_id', developer.get('reviewer', '')))

            state_tensors = []
            action_tensors = []

            for i, month_history in enumerate(monthly_activity_histories):
                if step_context_dates and i < len(step_context_dates):
                    month_ctx = step_context_dates[i]
                else:
                    month_ctx = context_date

                if not month_history:
                    state_tensors.append(torch.zeros(self.state_dim, device=self.device))
                    action_tensors.append(torch.zeros(self.action_dim, device=self.device))
                    continue

                total_proj = (step_total_project_reviews[i]
                              if step_total_project_reviews and i < len(step_total_project_reviews)
                              else 0)
                pf = (step_path_features[i]
                      if step_path_features and i < len(step_path_features)
                      else None)
                ef = (step_event_features[i]
                      if step_event_features and i < len(step_event_features)
                      else None)
                s_t, a_t = self.extract_features_tensor(
                    email, month_history, month_ctx,
                    total_project_reviews=total_proj,
                    path_features_vec=pf,
                    event_features_vec=ef,
                )
                state_tensors.append(s_t)
                action_tensors.append(a_t)

            if not state_tensors:
                return {
                    'continuation_probability': 0.5,
                    'confidence': 0.0,
                    'reasoning': '有効な月次ステップがありません'
                }

            state_seq = torch.stack(state_tensors).unsqueeze(0)   # [1, seq_len, state_dim]
            action_seq = torch.stack(action_tensors).unsqueeze(0)  # [1, seq_len, action_dim]
            lengths = torch.tensor([len(state_tensors)], dtype=torch.long, device=self.device)

            predicted_reward, predicted_continuation = self.network(state_seq, action_seq, lengths)

            # Multi-task モデルの場合は指定ヘッドの確率を取得
            from .network_variants import is_multitask
            if self.model_type != 0 and is_multitask(self.model_type):
                continuation_prob = predicted_continuation[head_index].item()
            else:
                continuation_prob = predicted_continuation.item()
            if self.output_temperature and abs(self.output_temperature - 1.0) > 1e-6:
                import math
                p = min(max(continuation_prob, 1e-6), 1.0 - 1e-6)
                logit = math.log(p / (1.0 - p))
                continuation_prob = 1.0 / (1.0 + math.exp(-logit / self.output_temperature))

            confidence = min(abs(continuation_prob - 0.5) * 2, 1.0)

            return {
                'continuation_probability': continuation_prob,
                'reward_score': predicted_reward.item(),
                'confidence': confidence,
                'reasoning': f"月次IRL予測継続確率: {continuation_prob:.1%}",
                'method': 'monthly_sequence',
            }

    # ------------------------------------------------------------------
    # Feature importance (gradient-based)
    # ------------------------------------------------------------------

    # IRL state feature names (14-dim, same order as STATE_FEATURES in common_features.py)
    IRL_STATE_FEATURE_NAMES = STATE_FEATURES

    # IRL action feature names (5-dim, same order as ACTION_FEATURES in common_features.py)
    IRL_ACTION_FEATURE_NAMES = ACTION_FEATURES

    def compute_gradient_importance(
        self,
        trajectories: List[Dict[str, Any]],
        max_samples: int = 200,
    ) -> Dict[str, float]:
        """Compute gradient-based feature importance for state+action inputs.

        For each sample, computes |d(continuation_prob)/d(input_feature)|
        averaged over all samples.

        Args:
            trajectories: evaluation trajectories (same format as predict input)
            max_samples: cap on number of samples to keep computation fast

        Returns:
            Dict mapping feature name -> mean absolute gradient importance
        """
        self.network.eval()

        all_grads: List[np.ndarray] = []
        used = 0

        for traj in trajectories:
            if used >= max_samples:
                break

            developer = traj.get('developer', traj.get('developer_info', {}))
            activity_history = traj['activity_history']
            context_date = traj.get('context_date', datetime.now())

            if not activity_history:
                continue

            email = developer.get('email', developer.get('developer_id', developer.get('reviewer', '')))

            # Build tensors with gradient tracking (per step)
            state_tensors = []
            action_tensors = []
            for i in range(len(activity_history)):
                step_history = activity_history[:i + 1]
                s_t, a_t = self.extract_features_tensor(email, step_history, context_date)
                state_tensors.append(s_t)
                action_tensors.append(a_t)

            state_seq = torch.stack(state_tensors).unsqueeze(0).requires_grad_(True)
            action_seq = torch.stack(action_tensors).unsqueeze(0).requires_grad_(True)
            lengths = torch.tensor([len(activity_history)], dtype=torch.long, device=self.device)

            _, continuation = self.network(state_seq, action_seq, lengths)
            continuation.backward()

            # Collect gradients
            s_grad = state_seq.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()   # [state_dim]
            a_grad = action_seq.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()  # [action_dim]
            all_grads.append(np.concatenate([s_grad, a_grad]))
            used += 1

        if not all_grads:
            return {}

        mean_grads = np.mean(all_grads, axis=0)
        total = mean_grads.sum()
        if total > 0:
            mean_grads = mean_grads / total  # normalize to proportions

        names = self.IRL_STATE_FEATURE_NAMES + self.IRL_ACTION_FEATURE_NAMES
        return {name: float(val) for name, val in zip(names, mean_grads)}


if __name__ == "__main__":
    # テスト用の設定（マルチプロジェクト対応版）
    config = {
        'state_dim': 15,  # 10 → 14 → 15（core_reviewer_ratio追加）
        'action_dim': 5,
        'hidden_dim': 128,
        'learning_rate': 0.001
    }

    # IRLシステムを初期化
    irl_system = RetentionIRLSystem(config)

    print("継続予測IRLシステムのテスト完了（マルチプロジェクト対応版）")
    print(f"ネットワーク: {irl_system.network}")
    print(f"デバイス: {irl_system.device}")
    print(f"状態次元: {irl_system.state_dim}")
    print(f"行動次元: {irl_system.action_dim}")