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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
            state_dim: 状態の次元数（デフォルト: 10次元）
            action_dim: 行動の次元数（デフォルト: 5次元）
            hidden_dim: 隠れ層の次元数（デフォルト: 128）
            dropout: Dropout率（過学習防止、デフォルト: 0.1）
        """
        super().__init__()

        # ========================================
        # 状態エンコーダー: 開発者の状態（10次元）を高次元空間（128次元）に埋め込み
        # その後、低次元（64次元）に圧縮して特徴抽出
        # Dropout: 過学習防止のため0.1に設定
        # ========================================
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),      # 10 → 128: 高次元埋め込み
            nn.ReLU(),                              # 非線形活性化
            nn.Dropout(dropout),                    # 過学習防止（0.1）
            nn.Linear(hidden_dim, hidden_dim // 2), # 128 → 64: 特徴圧縮
            nn.ReLU(),                              # 非線形活性化
            nn.Dropout(dropout)                     # 過学習防止（0.1）
        )

        # ========================================
        # 行動エンコーダー: 開発者の行動（5次元）を高次元空間（128次元）に埋め込み
        # その後、低次元（64次元）に圧縮して特徴抽出
        # 状態エンコーダーと同じ構造で対称性を保つ
        # ========================================
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),      # 5 → 128: 高次元埋め込み
            nn.ReLU(),                              # 非線形活性化
            nn.Dropout(dropout),                    # 過学習防止（0.1）
            nn.Linear(hidden_dim, hidden_dim // 2), # 128 → 64: 特徴圧縮
            nn.ReLU(),                              # 非線形活性化
            nn.Dropout(dropout)                     # 過学習防止（0.1）
        )

        # ========================================
        # LSTM: 時系列データ（月次の状態・行動）を処理
        # 可変長シーケンスに対応（pack_padded_sequenceを使用）
        # 入力: 64次元（エンコード後の状態+行動）、出力: 128次元
        # ========================================
        # LSTM: 時系列パターンを学習（1層、hidden_dim=128）
        self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim, num_layers=1, batch_first=True, dropout=dropout)

        # ========================================
        # 報酬予測器: LSTMの隠れ状態から報酬スコアを予測
        # 報酬スコア: その行動が継続にどれだけ寄与するか（-1〜1の範囲）
        # ========================================
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), # 128 → 64: 特徴抽出
            nn.ReLU(),                              # 非線形活性化
            nn.Dropout(dropout),                    # 過学習防止（0.1）
            nn.Linear(hidden_dim // 2, 1)           # 64 → 1: 報酬スコア出力
        )

        # ========================================
        # 継続確率予測器: LSTMの隠れ状態から継続確率を予測
        # 継続確率: 将来もレビュー依頼を承諾し続ける確率（0〜1の範囲）
        # Sigmoid: 出力を0〜1の範囲に制限
        # ========================================
        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), # 128 → 64: 特徴抽出
            nn.ReLU(),                              # 非線形活性化
            nn.Dropout(dropout),                    # 過学習防止（0.1）
            nn.Linear(hidden_dim // 2, 1),          # 64 → 1: 継続確率（ロジット）
            nn.Sigmoid()                            # 0〜1の範囲に正規化
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

        # ステップ2: 状態と行動を結合（要素ごとの加算）
        # 加算により、状態と行動の情報を融合（次元数は変わらず64次元を保持）
        combined = state_encoded + action_encoded

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
        # パディング部分を無視し、実際のデータの最終ステップのみを使用
        hidden = torch.zeros(batch_size, lstm_out.size(-1), device=state.device)
        for i in range(batch_size):
            actual_len = lengths[i].item()
            hidden[i] = lstm_out[i, actual_len - 1, :]  # 実際の最終ステップ

        # ========================================
        # 報酬と継続確率を予測
        # 隠れ状態から最終的な予測を行う
        # ========================================
        reward = self.reward_predictor(hidden)                # 報酬スコア（-1〜1の範囲）
        continuation_prob = self.continuation_predictor(hidden)  # 継続確率（0〜1の範囲）

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

        combined = state_encoded + action_encoded

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
        self.state_dim = config.get('state_dim', 14)  # 10 → 14
        self.action_dim = config.get('action_dim', 5)  # 4 → 5
        self.hidden_dim = config.get('hidden_dim', 128)
        self.dropout = config.get('dropout', 0.1)
        # 予測確率の温度スケーリング（1.0で無効、<1でシャープ、>1でフラット）
        self.output_temperature = float(config.get('output_temperature', 1.0))

        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ネットワーク初期化
        self.network = RetentionIRLNetwork(
            self.state_dim, self.action_dim, self.hidden_dim, self.dropout
        ).to(self.device)
        
        # オプティマイザー（学習率を増加: 0.001 → 0.0003がデフォルト、さらに調整可能）
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get('learning_rate', 0.0003)
        )
        
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
            # 行動を抽出
            actions = self.extract_developer_actions(activity_history, context_date)

            if not actions:
                return {
                    'continuation_probability': 0.5,
                    'confidence': 0.0,
                    'reasoning': '活動履歴が不足しているため、デフォルト確率を返します'
                }

            # 時系列モード: 全活動を可変長シーケンスとして使用
            # 状態テンソル: 各タイムステップで状態を構築
            state_tensors = []
            states = []  # デバッグ/理由生成用に保存
            for i in range(len(actions)):
                # 各ステップまでの履歴を使用
                step_history = activity_history[:i+1]
                step_state = self.extract_developer_state(developer, step_history, context_date)
                states.append(step_state)
                state_tensors.append(self.state_to_tensor(step_state))

            state_seq = torch.stack(state_tensors).unsqueeze(0)  # [1, seq_len, state_dim]

            # 行動テンソル: 時系列順序を保持
            action_tensors = [self.action_to_tensor(action) for action in actions]
            action_seq = torch.stack(action_tensors).unsqueeze(0)  # [1, seq_len, action_dim]

            # 可変長シーケンスとして処理
            lengths = torch.tensor([len(actions)], dtype=torch.long, device=self.device)
            predicted_reward, predicted_continuation = self.network(
                state_seq, action_seq, lengths
            )

            # 最終ステップの状態と行動を取得
            state = states[-1]
            recent_action = actions[-1]

            continuation_prob = predicted_continuation.item()
            reward_score = predicted_reward.item()

            # 信頼度計算（簡易版）
            confidence = min(abs(continuation_prob - 0.5) * 2, 1.0)

            # 理由生成
            reasoning = self._generate_irl_reasoning(
                state, recent_action, continuation_prob, reward_score
            )

            return {
                'continuation_probability': continuation_prob,
                'reward_score': reward_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'state_features': {
                    'experience_days': state.experience_days,
                    'recent_activity_frequency': state.recent_activity_frequency,
                    'collaboration_score': state.collaboration_score,
                    'code_quality_score': state.code_quality_score
                }
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
                                       epochs: int = 50) -> Dict[str, Any]:
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
        logger.info("=" * 60)
        logger.info("時系列IRL訓練開始")
        logger.info(f"軌跡数: {len(expert_trajectories)}")
        logger.info(f"エポック数: {epochs}")
        logger.info("=" * 60)
        
        self.network.train()
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for trajectory in expert_trajectories:
                try:
                    # 開発者情報と活動履歴を取得
                    developer = trajectory.get('developer', trajectory.get('developer_info', {}))
                    activity_history = trajectory.get('activity_history', [])
                    context_date = trajectory.get('context_date', datetime.now())
                    
                    if not activity_history:
                        continue

                    # 行動を抽出
                    actions = self.extract_developer_actions(activity_history, context_date)

                    if not actions:
                        continue

                    # 時系列データとして処理
                    # 各ステップで損失を計算（月次集約ラベル）
                    step_labels = trajectory.get('step_labels', [])
                    monthly_histories = trajectory.get('monthly_activity_histories', [])

                    if not step_labels or not monthly_histories:
                        continue

                    # 各月の時点での状態を計算（LSTM用）
                    state_tensors = []
                    action_tensors = []

                    min_len = min(len(monthly_histories), len(step_labels))

                    for i in range(min_len):
                        month_history = monthly_histories[i]
                        if not month_history:
                            # 活動履歴がない場合はゼロ状態
                            state_tensors.append(torch.zeros(self.state_dim, device=self.device))
                            action_tensors.append(torch.zeros(self.action_dim, device=self.device))
                            continue

                        # この月時点での状態を計算
                        month_context_date = month_history[-1]['timestamp']  # 最新の活動時刻
                        month_state = self.extract_developer_state(developer, month_history, month_context_date)
                        month_actions = self.extract_developer_actions(month_history, month_context_date)

                        state_tensors.append(self.state_to_tensor(month_state))

                        # 行動は最新のものを使用
                        if month_actions:
                            action_tensors.append(self.action_to_tensor(month_actions[-1]))
                        else:
                            action_tensors.append(torch.zeros(self.action_dim, device=self.device))

                    if not state_tensors or not action_tensors:
                        continue

                    # 3Dテンソルとして構築（LSTM用: [batch=1, sequence_length, features]）
                    state_sequence = torch.stack(state_tensors).unsqueeze(0)  # [1, seq_len, state_dim]
                    action_sequence = torch.stack(action_tensors).unsqueeze(0)  # [1, seq_len, action_dim]
                    lengths = torch.tensor([min_len], dtype=torch.long, device=self.device)

                    # forward_all_stepsで報酬と継続確率を予測
                    predicted_reward, predicted_continuation = self.network.forward_all_steps(
                        state_sequence, action_sequence, lengths, return_reward=True
                    )
                    # [1, seq_len] → [seq_len] にflatten
                    predicted_reward_flat = predicted_reward.squeeze(0)
                    predicted_continuation_flat = predicted_continuation.squeeze(0)

                    # 損失計算（月次集約ラベルを使用）
                    targets = torch.tensor([1.0 if label else 0.0 for label in step_labels[:min_len]], device=self.device)

                    # サンプル重みを取得（依頼なし=0.5、依頼あり=1.0）
                    sample_weight = trajectory.get('sample_weight', 1.0)
                    sample_weights = torch.full([min_len], sample_weight, device=self.device)

                    # 継続予測損失（Focal Loss を使用、重み付き）
                    continuation_loss = self.focal_loss(
                        predicted_continuation_flat, targets, sample_weights
                    )

                    # 報酬損失（月次集約ラベルから逆算）
                    reward_loss = F.mse_loss(
                        predicted_reward_flat, targets * 2.0 - 1.0  # -1 to 1
                    )

                    loss_per_step = continuation_loss + reward_loss

                    # バックプロパゲーション
                    self.optimizer.zero_grad()
                    loss_per_step.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss_per_step.item()
                    batch_count += 1
                
                except Exception as e:
                    logger.warning(f"軌跡処理エラー: {e}")
                    continue
            
            avg_loss = epoch_loss / max(batch_count, 1)
            training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"エポック {epoch}: 平均損失 = {avg_loss:.4f}")
        
        logger.info("時系列IRL訓練完了")
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'epochs_trained': epochs
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
            # スナップショット時点での状態と行動を抽出
            state = self.extract_developer_state(developer, activity_history, context_date)
            actions = self.extract_developer_actions(activity_history, context_date)
            
            if not actions:
                return {
                    'continuation_probability': 0.5,
                    'confidence': 0.0,
                    'reasoning': '活動履歴が不足しているため、デフォルト確率を返します'
                }
            
            # 最新の行動を使用（スナップショット特徴量）
            latest_action = actions[-1]
            state_tensor = self.state_to_tensor(state)
            action_tensor = self.action_to_tensor(latest_action)

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


    # ------------------------------------------------------------------
    # Feature importance (gradient-based)
    # ------------------------------------------------------------------

    # IRL state feature names (14-dim, same order as state_to_tensor)
    IRL_STATE_FEATURE_NAMES = [
        'experience_days', 'total_changes', 'total_reviews',
        'recent_activity_frequency', 'avg_activity_gap', 'activity_trend',
        'collaboration_score', 'code_quality_score',
        'recent_acceptance_rate', 'review_load',
        'project_count', 'project_activity_distribution',
        'main_project_contribution_ratio', 'cross_project_collaboration_score',
    ]

    # IRL action feature names (5-dim, same order as action_to_tensor)
    IRL_ACTION_FEATURE_NAMES = [
        'intensity', 'collaboration', 'response_speed',
        'review_size', 'is_cross_project',
    ]

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

            actions = self.extract_developer_actions(activity_history, context_date)
            if not actions:
                continue

            # Build tensors with gradient tracking
            state_tensors = []
            for i in range(len(actions)):
                step_history = activity_history[:i + 1]
                step_state = self.extract_developer_state(developer, step_history, context_date)
                state_tensors.append(self.state_to_tensor(step_state))

            state_seq = torch.stack(state_tensors).unsqueeze(0).requires_grad_(True)
            action_tensors = [self.action_to_tensor(a) for a in actions]
            action_seq = torch.stack(action_tensors).unsqueeze(0).requires_grad_(True)
            lengths = torch.tensor([len(actions)], dtype=torch.long, device=self.device)

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
        'state_dim': 14,  # 10 → 14（プロジェクト特徴量4つ追加）
        'action_dim': 5,  # 4 → 5（プロジェクト特徴量1つ追加）
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