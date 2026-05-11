"""
報酬関数モジュール（RL/reward/reward.py）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このファイルの役割
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
強化学習（RL）において「その行動はどれだけ良かったか」を数値で返すのが
報酬関数の役割。このファイルでは：

  1. RewardFunction  ← 報酬関数の「型」を定義する抽象クラス
  2. IRLReward       ← 卒論の IRL モデルを使った具体的な実装
  3. CustomReward    ← 将来の別実装のためのテンプレート

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 抽象クラスとは？（初学者向け）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
通常のクラスは「設計図」だが、抽象クラスは「設計書のルール」。
@abstractmethod がついたメソッドは「必ずサブクラスで実装しなさい」という
強制ルール。実装しないと TypeError が発生する。

    class Animal(ABC):          # ABC = Abstract Base Class
        @abstractmethod
        def speak(self): ...    # ← このメソッドは必ず実装が必要

    class Dog(Animal):
        def speak(self):        # ← 実装しないと TypeError
            return "ワン"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ なぜ抽象クラスを使うのか？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ReviewEnv（環境）は reward_fn.compute() を呼ぶだけでよく、
中身が IRLReward だろうと CustomReward だろうと気にしない。
→ 報酬関数を差し替えても ReviewEnv の変更ゼロ。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 使用例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from RL.reward.reward import IRLReward
    from RL.env.review_env import ReviewEnv

    # IRL モデルを報酬関数として使う
    reward_fn = IRLReward(model_path="outputs/train_0-3m/irl_model.pt")

    # ReviewEnv に渡すだけ（IRLReward の中身を知らなくてよい）
    env = ReviewEnv(df=df, reward_fn=reward_fn, ...)
"""

# Python 3.9 以前でも `str | Path` などの型ヒントを使えるようにするおまじない
from __future__ import annotations

import logging
from abc import ABC, abstractmethod  # 抽象クラスの仕組みを提供するモジュール
from pathlib import Path             # ファイルパスをオブジェクトとして扱う
from typing import Dict, Optional    # 型ヒント用

import numpy as np   # 数値計算ライブラリ（状態ベクトルは np.ndarray）
import torch         # PyTorch（IRL モデルの実行に使う）

# このモジュール専用のロガー（ログ出力用）
# logging.getLogger(__name__) により "RL.reward.reward" という名前のロガーになる
logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ブロック A: RewardFunction（抽象基底クラス）                          ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  【このブロックの役割】                                                 ║
# ║  「報酬関数はこういう形でなければならない」というルールを定義する。         ║
# ║  ABC(Abstract Base Class)を継承することで、@abstractmethod のついた     ║
# ║  メソッドを実装しないサブクラスは TypeError を出す仕組みになる。           ║
# ║                                                                      ║
# ║  【変更できること】                                                     ║
# ║  ・compute_total() の集計ロジック                                      ║
# ║      現在: 全員の報酬の総和 → 変更例: 最小値・重み付き平均              ║
# ║      方法: IRLReward や CustomReward の中で compute_total() を         ║
# ║            オーバーライド（上書き）する                                  ║
# ║                                                                      ║
# ║  【変更してはいけないこと】                                              ║
# ║  ・compute() の引数と戻り値の型シグネチャ                               ║
# ║      (state_vector, developer_id, action) → float                    ║
# ║      ここを変えると ReviewEnv が壊れる                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

class RewardFunction(ABC):
    """
    報酬関数の抽象基底クラス。

    全ての報酬関数はこのクラスを継承して `compute()` を実装する。
    これにより ReviewEnv は具体的な報酬関数の中身を知らなくてよい。

    継承の書き方:
        class MyReward(RewardFunction):   # ← (RewardFunction) で継承
            def compute(self, ...):
                return 何らかのスコア
    """

    @abstractmethod  # ← このデコレータが「必ず実装しなさい」という強制マーク
    def compute(
        self,
        state_vector: np.ndarray,  # 開発者1人の状態ベクトル（14次元 or それ以上）
        developer_id: str,         # 対象開発者のメールアドレス
        action: Dict,              # エージェントが選んだ行動（辞書形式）
    ) -> float:
        """
        1人の開発者に対する報酬を計算する。

        Args:
            state_vector: StateBuilder.build() が返す numpy 配列
                          例: [0.5, 0.3, 0.8, ...] （14次元）
            developer_id: 例 "alice@example.com"
            action: 例 {"developer_id": "alice@example.com", "step": 42}

        Returns:
            報酬値（float）。大きいほど「良い状態」を意味する。
        """
        # ... は「サブクラスで実装してください」という意味のプレースホルダ
        ...

    def compute_total(
        self,
        all_state_vectors: Dict[str, np.ndarray],  # {メール: 状態ベクトル} の辞書
        action: Dict,
    ) -> float:
        """
        全開発者の報酬の総和を計算する。

        これが ReviewEnv から呼ばれる主要メソッド。
        各開発者の compute() を呼んで合計する。
        サブクラスで上書きしなければこのデフォルト実装が使われる。

        数式イメージ:
            R_total = Σ_i compute(state_i, dev_i, action)
                    = 全員の継続確率の合計

        Args:
            all_state_vectors: StateBuilder.build_all() の出力
                               例: {"alice@...": array([...]), "bob@...": array([...])}
            action: 今ステップでエージェントが選んだ行動

        Returns:
            全開発者分の報酬の総和
        """
        total = 0.0
        for dev_id, vec in all_state_vectors.items():
            # 各開発者の報酬を足し合わせる
            total += self.compute(vec, dev_id, action)
        return total


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ブロック B: IRLReward（IRL モデルを使った報酬関数の実装）               ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  【このブロックの役割】                                                 ║
# ║  卒論で学習した IRL モデルを「審判」として RL に流用する。               ║
# ║  IRL モデルが出力する「継続確率スコア（0〜1）」= RL の報酬。             ║
# ║  負荷集中を防ぐペナルティも内包している。                                ║
# ║                                                                      ║
# ║  【変更できること】                                                     ║
# ║  ・model_path: 別の期間で学習した IRL モデルに差し替えられる             ║
# ║      例: "outputs/train_3-6m/irl_model.pt" に変えると                  ║
# ║          3-6ヶ月期間で学習したモデルを審判にできる                        ║
# ║  ・workload_penalty_weight: 0.0〜1.0 の範囲で調整                      ║
# ║      0.0: 負荷ペナルティなし（純粋に継続確率を最大化）                    ║
# ║      1.0: 負荷が高い人への推薦を強く罰する（負荷分散重視）                ║
# ║  ・device: GPU がある環境では "cuda" に変更すると推論が速くなる           ║
# ║  ・_load_model(): TODO 部分。irl_predictor.py の API に合わせて実装     ║
# ║  ・compute() の irl_score 計算部分: TODO 部分                          ║
# ║  ・ペナルティの計算式: 現在は線形だが二乗などに変更できる                  ║
# ║      現在: penalty = weight * review_load                             ║
# ║      変更: penalty = weight * review_load ** 2  （高負荷を強く罰す）    ║
# ║                                                                      ║
# ║  【変更してはいけないこと】                                              ║
# ║  ・compute() の引数と戻り値の型（RewardFunction の規約）                ║
# ║  ・self._load_model() の呼び出し（なければモデルが使えない）              ║
# ║  ・self._model へのモデルの代入（後続処理が依存している）                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

class IRLReward(RewardFunction):
    """
    卒論で学習済みの IRL モデルを報酬関数として使用する実装。

    ■ 報酬の設計
        報酬 = IRL が出力する「継続確率スコア」- 負荷ペナルティ

        - IRL スコア: 0〜1 の値で「この開発者が継続して貢献する確率」
        - 負荷ペナルティ: workload_penalty_weight > 0 のとき、
                         review_load が高い開発者に推薦すると報酬が下がる
                         → 自動的に負荷分散を学習させる仕組み

    ■ 遅延ロード（Lazy Loading）
        __init__ でモデルをロードせず、初回の compute() 呼び出し時に
        _load_model() でロードする。理由: モデルが大きく、
        オブジェクト作成時にメモリを消費したくないため。

    Args:
        model_path: 学習済みモデルのファイルパス（例: "outputs/train_0-3m/irl_model.pt"）
        workload_penalty_weight: 負荷ペナルティの強さ（0.0 なら無効）
        device: 推論に使うデバイス（"cpu" か "cuda"）
    """

    def __init__(
        self,
        model_path: str | Path,          # str でも Path オブジェクトでも受け取れる
        workload_penalty_weight: float = 0.0,
        device: str = "cpu",
    ) -> None:
        # Path() でラップすることで、文字列でも Path でも統一して扱える
        self.model_path = Path(model_path)

        self.workload_penalty_weight = workload_penalty_weight

        # torch.device オブジェクトに変換（"cpu" → device(type='cpu')）
        self.device = torch.device(device)

        # モデルは最初 None。_load_model() で初めてロードされる（遅延ロード）
        self._model: Optional[torch.nn.Module] = None

    def _load_model(self) -> None:
        """
        IRL モデル（v2）をファイルからロードする（遅延ロード）。

        self._model が None でなければ既にロード済みなので何もしない。
        → 何度 compute() を呼んでもロードは1回だけ。

        v39 系の保存形式:
            torch.save(irl_system.network.state_dict(), "irl_model.pt")
        を `RetentionIRLSystem` で復元し、network のみを self._model に保持する。
        """
        if self._model is not None:
            return  # 既にロード済みなので何もしない

        # 遅延 import（reward.py 単体の依存を軽くするため）
        from IRL.model.irl_predictor_v2 import RetentionIRLSystem
        from IRL.features.common_features import STATE_FEATURES, ACTION_FEATURES

        config = {
            "state_dim": len(STATE_FEATURES),
            "action_dim": len(ACTION_FEATURES),
            "hidden_dim": 128,
            "dropout": 0.1,
        }
        irl_system = RetentionIRLSystem(config)
        # device は IRLReward 側の指定に揃える（v2 のデフォルトは cuda を見るので上書き）
        irl_system.device = self.device
        irl_system.network = irl_system.network.to(self.device)

        state_dict = torch.load(self.model_path, map_location=self.device)
        irl_system.network.load_state_dict(state_dict)
        irl_system.network.eval()

        self._model = irl_system.network
        # state/action の次元数を保持しておく（compute() で分割に使う）
        self._state_dim = config["state_dim"]
        self._action_dim = config["action_dim"]

        logger.info(
            f"IRL モデルをロード: {self.model_path} "
            f"(state_dim={self._state_dim}, action_dim={self._action_dim})"
        )

    def compute(
        self,
        state_vector: np.ndarray,  # 1人の開発者の状態ベクトル
        developer_id: str,
        action: Dict,
    ) -> float:
        """
        IRL モデルで開発者の継続確率スコアを計算し、報酬として返す。

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        TODO: irl_score の計算部分を実装する
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        state_vector（numpy 配列）を PyTorch テンソルに変換して
        IRL モデルに渡す。

        実装例:
            # numpy 配列 → PyTorch テンソルに変換
            # unsqueeze(0) でバッチ次元を追加: shape (14,) → (1, 14)
            tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

            # 勾配計算不要（推論のみ）なので no_grad() でメモリを節約
            with torch.no_grad():
                irl_score = float(self._model.predict_score(tensor))
        """
        # まずモデルをロード（初回のみ実際にロードされる）
        self._load_model()

        # state_vector は StateBuilder.build() の出力（25次元 = STATE 20 + ACTION 5）
        # IRL ネットワークは state と action を別テンソルとして受け取るので分割する。
        expected_dim = self._state_dim + self._action_dim
        if len(state_vector) < expected_dim:
            # 想定より短い場合はゼロパディングして次元を揃える（早期エラーを避ける）
            padded = np.zeros(expected_dim, dtype=np.float32)
            padded[: len(state_vector)] = state_vector
            state_vector = padded

        state_part = state_vector[: self._state_dim]
        action_part = state_vector[self._state_dim : self._state_dim + self._action_dim]

        # [state_dim] → [batch=1, seq=1, state_dim]
        state_tensor = (
            torch.as_tensor(state_part, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        action_tensor = (
            torch.as_tensor(action_part, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        lengths = torch.tensor([1], dtype=torch.long, device=self.device)

        with torch.no_grad():
            _, continuation_prob = self._model(state_tensor, action_tensor, lengths)

        irl_score: float = float(continuation_prob.item())

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 負荷ペナルティの計算（実装済み・変更不要）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        penalty = 0.0

        # workload_penalty_weight > 0 かつ、今回の推薦対象がこの開発者のとき
        # action["developer_id"] == developer_id → 「この人に仕事を振った」
        if (
            self.workload_penalty_weight > 0.0
            and action.get("developer_id") == developer_id
        ):
            # review_load は状態ベクトルの9番目の要素
            # （IRL/features/common_features.py の FEATURE_NAMES の順番から決まる）
            # FEATURE_NAMES = ['experience_days', ..., 'review_load', ...]
            #                   index:  0                   9
            REVIEW_LOAD_IDX = 9

            if len(state_vector) > REVIEW_LOAD_IDX:
                # review_load が高いほどペナルティが大きくなる
                penalty = self.workload_penalty_weight * float(state_vector[REVIEW_LOAD_IDX])

        # 報酬 = IRL スコア - 負荷ペナルティ
        return irl_score - penalty


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ブロック C: CustomReward（独自報酬関数のテンプレート）                  ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  【このブロックの役割】                                                 ║
# ║  IRL とは全く異なる報酬設計を試したい場合のひな形。                       ║
# ║  compute() を実装して ReviewEnv に渡す reward_fn を差し替えるだけ。      ║
# ║                                                                      ║
# ║  【変更できること（= このクラスを継承して実装できる報酬の例）】              ║
# ║  ・承諾率を最大化:                                                      ║
# ║      return float(state_vector[8])  # recent_acceptance_rate          ║
# ║  ・活動頻度の高い人を優先:                                               ║
# ║      return float(state_vector[3])  # recent_activity_frequency       ║
# ║  ・経験年数に比例した報酬:                                               ║
# ║      return float(state_vector[0])  # experience_days                 ║
# ║  ・複数指標の組み合わせ:                                                 ║
# ║      score = 0.5 * state_vector[8] + 0.5 * state_vector[6]           ║
# ║      return float(score)  # 承諾率 + 協力スコアの平均                   ║
# ║                                                                      ║
# ║  【state_vector の各インデックスの意味】                                 ║
# ║   0: experience_days          1: total_changes                        ║
# ║   2: total_reviews            3: recent_activity_frequency            ║
# ║   4: avg_activity_gap         5: activity_trend                       ║
# ║   6: collaboration_score      7: overall_acceptance_rate                   ║
# ║   8: recent_acceptance_rate   9: review_load                          ║
# ║  10: avg_action_intensity    11: avg_collaboration                    ║
# ║  12: avg_response_time       13: avg_review_size                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

class CustomReward(RewardFunction):
    """
    IRL とは全く別の報酬設計を試したい場合のテンプレート。

    使い方:
        1. このクラスを継承して compute() を実装する
        2. ReviewEnv に渡す reward_fn を差し替えるだけで切り替えられる

    例: 承諾数の最大化を目的にする場合
        class AcceptanceReward(RewardFunction):
            def compute(self, state_vector, developer_id, action):
                # 今月の承諾率をそのまま報酬にする
                ACCEPTANCE_RATE_IDX = 8  # recent_acceptance_rate のインデックス
                return float(state_vector[ACCEPTANCE_RATE_IDX])
    """

    def compute(
        self,
        state_vector: np.ndarray,
        developer_id: str,
        action: Dict,
    ) -> float:
        # TODO: ここに独自の報酬ロジックを実装する
        raise NotImplementedError
