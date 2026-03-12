"""
RL エージェントモジュール（RL/agent/agent.py）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このファイルの役割
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ReviewEnv で学習する RL エージェントの定義。
「どの開発者に次のタスクを振るか」を試行錯誤しながら学習する。

    ReviewEnv（環境）:  「今の状況」と「報酬」を提供する場所
    ReviewAgent（agent）: 「どう行動するか」を決めて学習するもの

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ Stable-Baselines3 とは？（初学者向け）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RL アルゴリズムを自前で実装する必要はない。
Stable-Baselines3（SB3）という有名ライブラリが PPO・SAC・DQN などを
すでに実装しており、gymnasium 互換の環境があれば数行で使える。

    インストール: pip install stable-baselines3

    使い方（M2 で実装後）:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1)  # MlpPolicy = 全結合ネット
        model.learn(total_timesteps=100_000)       # 学習
        model.save("outputs/rl_agent/ppo_model")   # 保存

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ アルゴリズムの選定指針
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ReviewEnv の行動空間は Discrete（整数）なので PPO か DQN を使う。

  PPO（Proximal Policy Optimization）:
    - 離散・連続どちらでも使える汎用アルゴリズム
    - 学習が安定しやすい ← まず試すならこれ
    - SB3 での実装が最も充実している

  DQN（Deep Q-Network）:
    - 離散行動のみ対応
    - 開発者が少ない（<50人）場合に向いている
    - メモリ使用量が少ない

  SAC（Soft Actor-Critic）:
    - 連続行動のみ対応 → ReviewEnv の Discrete 空間には使えない
    - 将来行動空間を連続値に変えた場合の選択肢

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ M2 での実装手順
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: stable-baselines3 をインストール
    pip install stable-baselines3

Step 2: ReviewEnv が動作することを確認（env.reset(), env.step() が通るか）

Step 3: ReviewAgent の __init__() を実装
    from stable_baselines3 import PPO
    self._model = PPO("MlpPolicy", env, verbose=1)

Step 4: train(), evaluate(), save(), load() を実装

Step 5: scripts/train/train_rl_agent.py を作って学習を実行

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 使用例（M2 実装後）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from RL.agent.agent import ReviewAgent
    from RL.env.review_env import ReviewEnv

    env = ReviewEnv(...)
    agent = ReviewAgent(env=env, algorithm="PPO")

    # 学習（時間がかかる）
    agent.train(total_timesteps=100_000)

    # 評価（学習済みエージェントを評価する）
    results = agent.evaluate(n_episodes=10)
    print(results)  # {"rewards": [...], "accepted_counts": [...]}

    # 保存・ロード
    agent.save("outputs/rl_agent/ppo_model")
    agent.load("outputs/rl_agent/ppo_model")
"""

from __future__ import annotations

# TODO (M2): stable-baselines3 をインストールしてから以下のコメントを外す
# from stable_baselines3 import PPO, DQN
# from stable_baselines3.common.evaluation import evaluate_policy

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ブロック A: アルゴリズム定義（ALGORITHMS マッピング）                    ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  【このブロックの役割】                                                 ║
# ║  アルゴリズム名（文字列）→ SB3 クラスへのマッピングを定義する。            ║
# ║  ReviewAgent("PPO") のように文字列でアルゴリズムを切り替えられる。         ║
# ║                                                                      ║
# ║  【変更できること】                                                     ║
# ║  ・新しいアルゴリズムを追加できる（SB3 に含まれるもの）                   ║
# ║      例: from stable_baselines3 import A2C                            ║
# ║          ALGORITHMS = {"PPO": PPO, "DQN": DQN, "A2C": A2C}           ║
# ║  ・SB3 以外のライブラリのクラスも追加できる                              ║
# ║      例: RLlib, CleanRL など                                           ║
# ║                                                                      ║
# ║  【変更してはいけないこと】                                              ║
# ║  ・DiscreteAction 非対応のアルゴリズムを Discrete 環境に使う             ║
# ║      SAC は連続行動空間のみ対応 → ReviewEnv では使えない                ║
# ╚══════════════════════════════════════════════════════════════════════╝

# SB3 が対応するアルゴリズムの名前→クラスのマッピング（M2 で有効化）
# ALGORITHMS = {"PPO": PPO, "DQN": DQN}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ブロック B: ReviewAgent クラス                                        ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  【このブロックの役割】                                                 ║
# ║  ReviewEnv と SB3 を橋渡しするラッパー。train/evaluate/save/load の     ║
# ║  統一 API を提供する。                                                  ║
# ║                                                                      ║
# ║  【サブブロック構成】                                                   ║
# ║  B-1: __init__()    SB3 モデルの初期化（M2 で実装）                    ║
# ║  B-2: train()       学習の実行（SB3 の learn() を呼ぶだけ）             ║
# ║  B-3: evaluate()    学習済みエージェントの評価                           ║
# ║  B-4: save()        モデルのファイル保存                                ║
# ║  B-5: load()        保存済みモデルのロード                              ║
# ║                                                                      ║
# ║  【変更できること】                                                     ║
# ║  ・__init__() の SB3 モデル設定（ネットワーク構造など）                  ║
# ║      例: policy_kwargs={"net_arch": [128, 128]} で層の幅を変える        ║
# ║  ・evaluate() の評価指標（報酬・承諾数以外に AUC-ROC なども計測できる）   ║
# ║  ・train() にコールバックを追加してログ・早期終了を実装できる              ║
# ║      例: SB3 の EvalCallback, CheckpointCallback                      ║
# ║                                                                      ║
# ║  【変更してはいけないこと】                                              ║
# ║  ・self.env への参照（SB3 の load() で必要）                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

class ReviewAgent:
    """
    レビュータスク推薦 RL エージェント。

    ReviewEnv と Stable-Baselines3 を橋渡しするラッパークラス。
    train() / evaluate() / save() / load() の統一 API を提供する。

    ■ なぜラッパークラスを作るのか？
    SB3 の PPO や DQN を直接使ってもよいが、ラッパーにまとめることで：
      - アルゴリズムの切り替えが1行（algorithm="PPO" → "DQN"）で済む
      - 評価ロジックをカスタマイズしやすい
      - 保存・ロードのパス管理を一元化できる

    Args:
        env:       ReviewEnv のインスタンス（学習する環境）
        algorithm: 使用するアルゴリズム名（"PPO" or "DQN"）
    """

    # ── B-1: __init__() ──────────────────────────────────────────────────
    # 【変更できること】
    # ・policy_kwargs でネットワーク構造を変えられる（M2 実装後）
    #   例: "MlpPolicy" のデフォルトは [64, 64] の2層全結合
    #       policy_kwargs={"net_arch": [256, 256]} で大きくできる
    # ・learning_rate, n_steps など SB3 のハイパーパラメータを調整できる
    #   例: PPO("MlpPolicy", env, learning_rate=1e-4, n_steps=2048)
    # ・verbose=0 にするとログ出力を抑制できる（本番実行時など）
    # 【変更してはいけないこと】
    # ・self.env と self.algorithm の保存（後続メソッドが参照する）

    def __init__(self, env, algorithm: str = "PPO") -> None:
        self.env = env
        self.algorithm = algorithm

        # TODO (M2): SB3 のモデルをここで初期化する
        # 実装例:
        #   AlgorithmClass = ALGORITHMS[algorithm]
        #   self._model = AlgorithmClass(
        #       policy="MlpPolicy",  # 全結合ニューラルネット（状態→行動）
        #       env=env,
        #       verbose=1,           # 学習ログを標準出力に表示
        #   )
        #   logger.info(f"エージェント初期化完了: algorithm={algorithm}")
        self._model = None  # TODO (M2): 上記の実装に置き換える

    # ── B-2: train() ─────────────────────────────────────────────────────
    # 【変更できること】
    # ・コールバックを追加して学習中にモデルを定期保存したり評価できる
    #   例（M2 実装後）:
    #     from stable_baselines3.common.callbacks import CheckpointCallback
    #     cb = CheckpointCallback(save_freq=10_000, save_path="outputs/rl_checkpoints/")
    #     self._model.learn(total_timesteps=total_timesteps, callback=cb)
    # ・total_timesteps のデフォルト値を変えてよい
    #   小さい値（1000〜10000）でまず動作確認 → 大きい値（100万〜）で本学習

    def train(self, total_timesteps: int = 100_000) -> None:
        """
        エージェントを訓練する。

        total_timesteps 回だけ env.step() を繰り返し、
        報酬が大きくなるような行動方策を学習する。

        Args:
            total_timesteps: 学習に使う総ステップ数
                             多いほど精度が上がるが時間もかかる
                             目安: 10万〜100万ステップ

        TODO (M2): 以下を実装する
            self._model.learn(total_timesteps=total_timesteps)
            logger.info(f"学習完了: {total_timesteps} ステップ")
        """
        # TODO (M2): 実装する
        raise NotImplementedError(
            "TODO (M2): train() を実装してください。\n"
            "self._model.learn(total_timesteps=total_timesteps) を呼ぶだけです。"
        )

    # ── B-3: evaluate() ──────────────────────────────────────────────────
    # 【変更できること】
    # ・評価指標を追加できる
    #   例: 史実の推薦と比較した AUC-ROC を計算する
    #       変動型開発者（33.3%）の承諾率を別途集計する
    # ・deterministic=False にすると確率的な行動で評価できる
    #   → 学習中の中間評価に使える（ランダム性があるため複数回平均を取る）

    def evaluate(self, n_episodes: int = 10) -> Dict:
        """
        訓練済みエージェントを評価する。

        n_episodes エピソード分だけシミュレーションを走らせ、
        各エピソードの合計報酬と承諾数を記録して返す。

        Args:
            n_episodes: 評価するエピソード数（多いほど安定した評価になる）

        Returns:
            {"rewards": [各エピソードの合計報酬], "accepted_counts": [各エピソードの承諾数]}

        TODO (M2): 以下を実装する
            results = {"rewards": [], "accepted_counts": []}
            for _ in range(n_episodes):
                obs, _ = self.env.reset()
                total_reward = 0.0
                accepted_count = 0
                done = False

                while not done:
                    # deterministic=True: 確率的でなく決定的に行動を選ぶ（評価時）
                    action, _ = self._model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    if info.get("accepted"):
                        accepted_count += 1
                    done = terminated or truncated

                results["rewards"].append(total_reward)
                results["accepted_counts"].append(accepted_count)

            return results
        """
        # TODO (M2): 実装する
        raise NotImplementedError(
            "TODO (M2): evaluate() を実装してください。\n"
            "docs/rl_implementation_guide.md の Section 4 を参照してください。"
        )

    # ── B-4: save() / B-5: load() ────────────────────────────────────────
    # 【変更できること】
    # ・保存先のパス規則を決める（outputs/ 配下に揃えると管理しやすい）
    #   例: "outputs/rl_agent/{algorithm}_{timestamp}" の形式
    # ・load() 後に追加学習（ファインチューニング）もできる
    #   env が変わった場合（例: 別プロジェクト）は env=new_env を渡す

    def save(self, path: str) -> None:
        """
        学習済みモデルをファイルに保存する。

        Args:
            path: 保存先のファイルパス（拡張子 .zip は自動付与される）
                  例: "outputs/rl_agent/ppo_model"
                  → "outputs/rl_agent/ppo_model.zip" として保存される

        TODO (M2): 以下を実装する
            self._model.save(path)
            logger.info(f"モデルを保存: {path}")
        """
        # TODO (M2): 実装する
        raise NotImplementedError

    def load(self, path: str) -> None:
        """
        保存済みモデルをファイルからロードする。

        save() で保存したモデルを再ロードして、
        追加学習や評価に使えるようにする。

        Args:
            path: ロードするファイルパス（.zip は省略可）
                  例: "outputs/rl_agent/ppo_model"

        TODO (M2): 以下を実装する
            # SB3 の load() は クラスメソッドなので type() でクラスを取得して呼ぶ
            AlgorithmClass = type(self._model)
            self._model = AlgorithmClass.load(path, env=self.env)
            logger.info(f"モデルをロード: {path}")
        """
        # TODO (M2): 実装する
        raise NotImplementedError
