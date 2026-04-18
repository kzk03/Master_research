"""
レビュー推薦 RL 環境（RL/env/review_env.py）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このファイルの役割
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ReviewEnv は「シミュレータの司令塔」。RL エージェントが学習するための
ゲームボードのような存在。

    エージェント: 「開発者 A に次のレビューを振ろう」（行動）
          ↓
    ReviewEnv:  「A は承諾した。全体の継続確率は 8.3 だ」（報酬）
          ↓
    エージェント: 次の行動を選ぶ（観測を見て）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ gymnasium とは？（初学者向け）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gymnasium は RL 環境の標準ライブラリ（OpenAI Gym の後継）。
gym.Env を継承して reset() と step() を実装するだけで、
Stable-Baselines3 などの RL ライブラリと自動的に連携できる。

エピソードの基本ループ:
    obs, info = env.reset()          # ← 毎エピソードの最初に呼ぶ
    while True:
        action = agent.predict(obs)  # エージェントが行動を選ぶ
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break                    # エピソード終了

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このクラスの行動空間と観測空間
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
行動 (action):
    整数 0〜n_developers-1 のどれか
    → action=2 なら「developer_ids[2] に推薦する」という意味

観測 (observation):
    全開発者の状態ベクトルを横に並べた 1 次元配列
    例: 開発者 30 人 × 14 次元 = 420 次元の配列

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 使用例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from RL.env.review_env import ReviewEnv
    from RL.reward.reward import IRLReward
    from RL.state.state_builder import StateBuilder

    reward_fn = IRLReward(model_path="outputs/train_0-3m/irl_model.pt")
    state_builder = StateBuilder(window_days=90)

    env = ReviewEnv(
        df=df,
        reward_fn=reward_fn,
        state_builder=state_builder,
        eval_start=datetime(2013, 1, 1),
        eval_end=datetime(2013, 4, 1),
    )

    obs, info = env.reset()
    print(obs.shape)  # (n_developers * 14,)

    # ランダムな行動でステップを試す
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym          # RL 環境の標準ライブラリ
from gymnasium import spaces     # 行動空間・観測空間の定義に使う

from RL.reward.reward import RewardFunction   # 報酬関数の抽象クラス
from RL.state.state_builder import StateBuilder  # 状態ベクトル構築クラス

# タスク特徴量の定義（obs の先頭に追加される）
TASK_FEATURE_NAMES = [
    "change_insertions",
    "change_deletions",
    "change_files_count",
    "is_cross_project",
]
TASK_FEATURE_DIM = len(TASK_FEATURE_NAMES)

# Step 3: df に 'dirs' 列がある場合に使う空集合の定数
_EMPTY_DIRS: frozenset = frozenset()

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ブロック A: ReviewEnv クラス全体                                      ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  【このブロックの役割】                                                 ║
# ║  OpenStack の履歴データを「ゲームボード」として再現する RL 環境。          ║
# ║  gymnasium.Env を継承しているため Stable-Baselines3 などと直結できる。    ║
# ║                                                                      ║
# ║  【サブブロック構成】                                                   ║
# ║  A-1: __init__()           環境の初期化（開発者リスト・空間定義など）     ║
# ║  A-2: reset()              エピソード開始時の初期化                     ║
# ║  A-3: step()               1ステップの処理（行動→報酬→次の状態）        ║
# ║  A-4: _get_obs()           状態辞書 → 1次元観測ベクトルへの変換         ║
# ║  A-5: _get_info()          デバッグ情報の生成                          ║
# ║  A-6: _build_event_queue() イベントキューの構築                        ║
# ║  A-7: get_developer_index/id 開発者ID ↔ インデックス変換ユーティリティ  ║
# ║                                                                      ║
# ║  【変更できること】                                                     ║
# ║  ・reward_fn: IRLReward → CustomReward など差し替え自由                ║
# ║  ・state_builder: window_days や use_macro の設定を変えるだけ           ║
# ║  ・developer_ids: 活動中の開発者だけに絞る（TODO部分）                   ║
# ║  ・max_steps: エピソードの長さを制限してデバッグしやすくできる             ║
# ║  ・_build_event_queue(): イベント種別（レビューのみ、等）でフィルタできる  ║
# ║  ・action_space: Discrete（整数）→ MultiDiscrete（複数推薦）に変更可    ║
# ║                                                                      ║
# ║  【変更してはいけないこと】                                              ║
# ║  ・reset() の戻り値の型: (np.ndarray, dict) → SB3 が要求する形式        ║
# ║  ・step() の戻り値の型: (obs, reward, terminated, truncated, info)     ║
# ║  ・observation_space と action_space の定義（gymnasium のルール）       ║
# ╚══════════════════════════════════════════════════════════════════════╝

class ReviewEnv(gym.Env):
    """
    レビュータスク推薦のための RL 環境。

    gym.Env を継承することで Stable-Baselines3 などの RL ライブラリと
    自動的に連携できる。必須メソッドは reset() と step() の2つ。

    ■ 設計のポイント
    reward_fn と state_builder を外から渡す構造にしているため、
    このクラス自体を変えずに報酬・状態の設計を差し替えられる。

    Args:
        df:            OpenStack の全レビューデータ（全期間）
        reward_fn:     報酬関数（RewardFunction のサブクラスのインスタンス）
        state_builder: 状態ベクトル構築クラスのインスタンス
        developer_ids: 推薦候補の開発者リスト（None なら df から全員抽出）
        eval_start:    シミュレーション開始日時（None なら df の最古日時）
        eval_end:      シミュレーション終了日時（None なら df の最新日時）
        max_steps:     1エピソードの最大ステップ数（None なら全イベント数）
        active_window_days: action_masks() で「アクティブ」と判定する直近日数
        min_candidates: action_masks() の有効候補数の下限（少なすぎる場合は
                       直近活動者で補充する）
    """

    # gymnasium が要求するクラス変数（レンダリングモードの定義）
    # 今回は可視化不要なので空リスト
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        reward_fn: RewardFunction,
        state_builder: StateBuilder,
        developer_ids: Optional[List[str]] = None,
        eval_start: Optional[datetime] = None,
        eval_end: Optional[datetime] = None,
        max_steps: Optional[int] = None,
        active_window_days: int = 90,
        min_candidates: int = 5,
        hit_bonus_weight: float = 0.0,
        irl_reward_weight: float = 1.0,
        use_task_features: bool = False,
    ) -> None:
        # ── A-1: __init__ ────────────────────────────────────────────────
        # 【変更できること】
        # ・developer_ids を明示的に渡すと推薦候補を絞り込める
        #   例: 活動中の上位 N 人だけを対象にする
        #       top_devs = df.groupby("email").size().nlargest(50).index.tolist()
        #       env = ReviewEnv(df=df, developer_ids=top_devs, ...)
        # ・max_steps を設定するとエピソードが短くなりデバッグが楽になる
        #   例: ReviewEnv(..., max_steps=100)
        # 【変更してはいけないこと】
        # ・observation_space と action_space の定義（gymnasium のルール）
        #   これを変えると SB3 が環境を認識できなくなる

        # gym.Env の初期化を必ず呼ぶ（gymnasium のルール）
        super().__init__()

        # 引数をインスタンス変数として保存
        self.df = df
        self.reward_fn = reward_fn
        self.state_builder = state_builder
        self.max_steps = max_steps
        self.active_window_days = active_window_days
        self.min_candidates = min_candidates
        self.hit_bonus_weight = hit_bonus_weight
        self.irl_reward_weight = irl_reward_weight
        self.use_task_features = use_task_features

        # ── 開発者リストの確定 ──────────────────────────────────────────
        if developer_ids is not None:
            # 明示的に指定された場合はそのまま使う
            self.developer_ids = developer_ids
        else:
            # TODO: eval_start〜eval_end の間に活動した開発者だけを抽出する
            #       現在は全開発者を使っているため、非活動者が混入する可能性あり
            #
            # 改善例:
            #   active_mask = (df["timestamp"] >= eval_start) & (df["timestamp"] < eval_end)
            #   self.developer_ids = sorted(df[active_mask]["email"].unique().tolist())
            self.developer_ids = sorted(df["email"].unique().tolist())

        # 開発者の人数（行動空間のサイズになる）
        self.n_developers = len(self.developer_ids)

        # ── シミュレーション期間の確定 ────────────────────────────────────
        # eval_start/eval_end が None なら df の最古・最新の日時を使う
        # pd.to_datetime().min() → DataFrame 内の最古のタイムスタンプ
        # .to_pydatetime()       → Python の datetime オブジェクトに変換
        self.eval_start = eval_start or pd.to_datetime(df["timestamp"]).min().to_pydatetime()
        self.eval_end   = eval_end   or pd.to_datetime(df["timestamp"]).max().to_pydatetime()

        # ── イベントキューの構築 ───────────────────────────────────────────
        # eval_start〜eval_end の間に発生した全タスクを時系列順に並べる
        # → 1ステップ = イベントキューの1行を処理すること
        self._events: pd.DataFrame = self._build_event_queue()

        # ── gymnasium の空間定義 ────────────────────────────────────────
        # 観測空間: (タスク特徴) + 全開発者の状態ベクトルを連結した 1 次元配列
        obs_dim = self.n_developers * self.state_builder.obs_dim
        if self.use_task_features:
            obs_dim += TASK_FEATURE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf,   # 最小値（正規化済みなら実際は 0 以上だが余裕を持たせる）
            high=np.inf,   # 最大値
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 行動空間: 整数 0〜n_developers-1
        # action=k → developer_ids[k] に推薦する
        self.action_space = spaces.Discrete(self.n_developers)

        # ── 内部状態（reset() で初期化される）──────────────────────────
        self._step_idx: int = 0                          # 現在のステップ番号
        self._current_time: datetime = self.eval_start   # シミュレータの現在時刻
        self._current_state: Dict[str, np.ndarray] = {}  # {dev_id: 状態ベクトル}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # gymnasium 必須インターフェース
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── A-2: reset() ─────────────────────────────────────────────────────
    # 【変更できること】
    # ・reset() で返す obs に追加情報を含めたい場合は _get_obs() を修正する
    # ・エピソードごとに開始時刻をランダムにしたい場合:
    #   self._current_time = self.eval_start + timedelta(
    #       days=np.random.randint(0, 30))  ← ランダムオフセット
    # 【変更してはいけないこと】
    # ・super().reset(seed=seed) の呼び出し（gymnasium のルール）
    # ・戻り値が (obs, info) のタプルであること

    def reset(
        self,
        *,
        seed: Optional[int] = None,    # 乱数シード（再現性のために使う）
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        エピソードを初期状態にリセットする。

        毎エピソードの最初に呼ばれる。
        シミュレーション時刻を eval_start に戻し、
        全開発者の状態ベクトルを再計算する。

        Returns:
            obs:  初期観測ベクトル（全開発者の状態を連結）
            info: デバッグ情報の辞書
        """
        # gymnasium のルール: 乱数シードをスーパークラスに渡す
        super().reset(seed=seed)

        # 内部状態を初期化
        self._step_idx = 0
        self._current_time = self.eval_start

        # 全開発者の初期状態ベクトルを計算
        # StateBuilder.build_all() が {dev_id: array} の辞書を返す
        self._current_state = self.state_builder.build_all(
            df=self.df,
            developer_ids=self.developer_ids,
            current_time=self._current_time,
            task_dirs=self._get_current_task_dirs(),
        )

        obs = self._get_obs()     # 辞書 → 1次元配列に変換
        info = self._get_info()   # デバッグ情報
        return obs, info

    # ── A-3: step() ──────────────────────────────────────────────────────
    # 【変更できること】
    # ・Step 3（承諾シミュレーション）: TODO 部分を実装する
    #   承諾確率の計算式や、承諾した場合のボーナス報酬なども追加できる
    #   例: accepted かつ developer が LTC なら報酬に +0.5 のボーナスを加算
    # ・Step 5（報酬計算）: 承諾の有無を報酬に反映する
    #   例: accepted=True のとき reward += 1.0 のボーナス
    # ・Step 6（終了判定）: 追加の終了条件を設けられる
    #   例: 承諾率が一定値を下回ったらエピソードを終了
    # ・action_dict に情報を追加して reward_fn に渡せる
    #   例: action_dict["task_type"] = event["action_type"] を加える
    # 【変更してはいけないこと】
    # ・戻り値の5タプル形式: (obs, reward, terminated, truncated, info)
    # ・_step_idx のインクリメント（ステップ管理に必須）

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        1ステップ進める（1タスク発生 = 1ステップ）。

        処理の流れ:
            1. action（整数）→ 推薦する開発者のメールアドレスに変換
            2. タスク発生時刻を現在のイベントの時刻に進める
            3. 開発者が承諾/拒否するかを確率的にシミュレート（TODO）
            4. 全開発者の状態ベクトルを更新
            5. 報酬を計算（全開発者の継続確率の総和）
            6. 終了判定

        Args:
            action: 推薦する開発者のインデックス（0〜n_developers-1 の整数）
                    例: action=2 → developer_ids[2] に推薦

        Returns:
            obs:        次の観測ベクトル
            reward:     このステップの報酬（全員の継続確率の総和）
            terminated: True なら eval_end に達してエピソード終了
            truncated:  True なら max_steps に達してエピソード終了
            info:       デバッグ情報
        """
        # ── Step 1: action（整数）→ 開発者 ID に変換 ─────────────────────
        # action=2 → developer_ids[2] = "bob@example.com" のような変換
        recommended_dev = self.developer_ids[action]

        # 報酬関数・ログに渡す辞書形式の行動表現
        action_dict = {
            "developer_id": recommended_dev,
            "step": self._step_idx,
        }

        # ── Step 2: シミュレータの時刻を進める ────────────────────────────
        # イベントキューから現在のタスク発生時刻を取得
        if self._step_idx < len(self._events):
            event = self._events.iloc[self._step_idx]  # 現在のイベント行
            # pandas の Timestamp → Python の datetime に変換
            self._current_time = pd.to_datetime(event["timestamp"]).to_pydatetime()

        # ── Step 3: 承諾/拒否のシミュレーション ──────────────────────────
        # 推薦された開発者が実際に承諾するかを確率的に決める。
        # IRL の継続確率を承諾確率として使う（reward_fn.compute() は
        # IRL の continuation_prob - 負荷ペナルティを返すので、それを 0〜1 に
        # クリップして Bernoulli サンプリングする）。
        state_vec = self._current_state.get(
            recommended_dev,
            np.zeros(self.state_builder.obs_dim, dtype=np.float32),
        )
        acceptance_prob = float(
            np.clip(
                self.reward_fn.compute(state_vec, recommended_dev, action_dict),
                0.0,
                1.0,
            )
        )
        # gymnasium の seed と整合するように self.np_random（reset で初期化済）を使う
        accepted: bool = bool(self.np_random.random() < acceptance_prob)

        # ── Step 4: 状態を更新 ────────────────────────────────────────────
        # 時刻が進んだので、新しい時刻で全員の状態ベクトルを再計算する
        # （window_days 分の過去データのウィンドウがスライドする）
        self._current_state = self.state_builder.build_all(
            df=self.df,
            developer_ids=self.developer_ids,
            current_time=self._current_time,
            task_dirs=self._get_current_task_dirs(),
        )

        # ── Step 5: 報酬の計算 ────────────────────────────────────────────
        # 全開発者の継続確率の総和 = エコシステム全体の健全度
        # RewardFunction.compute_total() が全員分の compute() を合計して返す
        irl_reward = self.reward_fn.compute_total(
            all_state_vectors=self._current_state,
            action=action_dict,
        )

        # 史実一致シェーピング: 推薦先 == 史実 reviewer なら +bonus
        # Hit Rate と reward を直結させ、推薦精度向上に学習信号を流す
        hit_bonus = 0.0
        true_dev: Optional[str] = None
        if self._step_idx < len(self._events):
            true_dev = self._events.iloc[self._step_idx].get("email")
        is_hit = bool(true_dev is not None and recommended_dev == true_dev)
        if self.hit_bonus_weight != 0.0 and is_hit:
            hit_bonus = self.hit_bonus_weight

        reward = self.irl_reward_weight * irl_reward + hit_bonus

        # ステップカウンタを進める
        self._step_idx += 1

        # ── Step 6: 終了判定 ──────────────────────────────────────────────
        # terminated: 自然な終了（シミュレーション期間が終わった）
        terminated = self._current_time >= self.eval_end

        # truncated: 強制終了（max_steps に達した）
        # max_steps が None の場合は never truncate
        truncated = (
            self.max_steps is not None and self._step_idx >= self.max_steps
        )

        obs = self._get_obs()
        info = self._get_info(
            accepted=accepted,
            recommended_dev=recommended_dev,
            acceptance_prob=acceptance_prob,
            is_hit=is_hit,
            true_dev=true_dev,
            irl_reward=float(irl_reward),
            hit_bonus=float(hit_bonus),
        )
        return obs, reward, terminated, truncated, info

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 内部ユーティリティメソッド（A-4〜A-7）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ── A-4: _get_obs() ──────────────────────────────────────────────────
    # 【変更できること】
    # ・観測に含める情報を変えたい場合はここを修正する
    #   例: 全開発者ではなく「推薦候補 Top-K 人」だけに絞ると obs が小さくなる
    #   例: 時系列情報（前ステップの状態との差分）を追加する
    # 【変更してはいけないこと】
    # ・出力の shape が observation_space.shape と一致していること
    #   ズレると SB3 が例外を出す

    # ── A-5: _get_info() ─────────────────────────────────────────────────
    # 【変更できること】
    # ・ログ・デバッグに必要な情報を自由に追加できる
    #   例: info["event_type"] = event["action_type"] を追加

    # ── A-6: _build_event_queue() ────────────────────────────────────────
    # 【変更できること】
    # ・イベントの種類で絞り込む
    #   例: mask &= (df["action_type"] == "review")
    # ・イベントをランダムにサンプリングしてエピソードを短くする
    #   events = events.sample(frac=0.1, random_state=42)
    # 【変更してはいけないこと】
    # ・時系列順（sort_values("timestamp")）は必須
    #   順番がズレると過去のデータで未来を学習することになる

    def _get_task_features(self) -> np.ndarray:
        """
        現在処理中のイベントからタスク特徴量 (4次元) を抽出する。

        change_insertions, change_deletions, change_files_count, is_cross_project
        を正規化して返す。イベントが範囲外ならゼロベクトル。
        """
        if self._step_idx >= len(self._events):
            return np.zeros(TASK_FEATURE_DIM, dtype=np.float32)
        event = self._events.iloc[self._step_idx]
        return np.array([
            min(float(event.get("change_insertions", 0)) / 1000.0, 1.0),
            min(float(event.get("change_deletions", 0)) / 1000.0, 1.0),
            min(float(event.get("change_files_count", 0)) / 50.0, 1.0),
            float(event.get("is_cross_project", False)),
        ], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        現在の状態辞書 → gymnasium が受け取れる 1 次元配列に変換する。

        use_task_features=True のとき、先頭にタスク特徴量 (4次元) を追加する。
        その後ろに全開発者の状態ベクトルを連結する。

        開発者の順番は self.developer_ids のリスト順で固定。
        """
        vectors = [
            self._current_state.get(
                dev_id,
                np.zeros(self.state_builder.obs_dim, dtype=np.float32),
            )
            for dev_id in self.developer_ids
        ]
        dev_obs = np.concatenate(vectors, axis=0)
        if self.use_task_features:
            task_feat = self._get_task_features()
            return np.concatenate([task_feat, dev_obs], axis=0)
        return dev_obs

    def _get_info(self, **kwargs) -> Dict[str, Any]:
        """
        デバッグ・ログ用の情報辞書を作って返す。

        **kwargs は可変キーワード引数。
        呼び出し側から追加情報（accepted, recommended_dev など）を渡せる。

        使い方:
            info = self._get_info(accepted=True, recommended_dev="alice@...")
            # → {"step": 5, "current_time": ..., "accepted": True, ...}
        """
        return {
            "step": self._step_idx,
            "current_time": self._current_time,
            "n_developers": self.n_developers,
            **kwargs,  # 追加情報をそのまま展開して入れる
        }

    def _get_current_task_dirs(self) -> Optional[frozenset]:
        """
        現在処理中のイベントのタスクが触るディレクトリ集合を返す。

        df に 'dirs' 列が無い場合 (path features 無効の実験) は None を返す。
        'dirs' 列はあるが現在イベントが範囲外の場合は空 frozenset を返す。

        Returns:
            - None: path features 無効 (df に 'dirs' 列なし)
            - frozenset({...}): 現在イベントのディレクトリ集合
            - frozenset(): 現在イベントなし or 該当データなし
        """
        if "dirs" not in self._events.columns:
            return None
        if self._step_idx >= len(self._events):
            return _EMPTY_DIRS
        val = self._events.iloc[self._step_idx].get("dirs", _EMPTY_DIRS)
        if isinstance(val, frozenset):
            return val
        if isinstance(val, (set, list, tuple)):
            return frozenset(val)
        return _EMPTY_DIRS

    def _build_event_queue(self) -> pd.DataFrame:
        """
        シミュレーション期間（eval_start〜eval_end）内の
        全タスク発生イベントを時系列順に並べて返す。

        これが「1ステップ = 1タスク発生」の元となるリスト。
        イベントキューの長さ = 1エピソードのステップ数の上限。

        TODO（任意）: イベントの種類で絞り込みたい場合はここに追加
            例: レビュー依頼だけに絞る場合
                mask &= (df["action_type"] == "review")
        """
        df = self.df.copy()

        # timestamp 列を pandas の datetime 型に変換（文字列の場合に必要）
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # eval_start〜eval_end の間のデータだけを抽出
        mask = (df["timestamp"] >= self.eval_start) & (df["timestamp"] < self.eval_end)

        # 時刻順に並べて行インデックスをリセット
        events = df[mask].sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"イベントキュー構築完了: {len(events)} 件のタスク "
            f"({self.eval_start} 〜 {self.eval_end})"
        )
        return events

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 行動マスキング（sb3-contrib MaskablePPO 連携）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def action_masks(self) -> np.ndarray:
        """
        現在のステップで「有効」な行動のブールマスクを返す。

        sb3-contrib の MaskablePPO はこのメソッドを自動的に呼び出し、
        valid=False の行動を選ばないように学習する。

        マスク条件:
            1. 現在のイベントの owner_email と異なる開発者であること（自分の
               PR を自分にレビュー推薦するのは無意味なので除外）
            2. 直近 active_window_days 日以内に何らかの活動がある開発者であること
               （= active reviewer set）
            3. 上記で候補が min_candidates を下回る場合は、直近活動順で補充する

        Returns:
            shape (n_developers,) の bool 配列
        """
        # 現在の時刻基準で「直近 active_window_days」を判定
        active_start = self._current_time - pd.Timedelta(days=self.active_window_days)
        recent_mask = (
            (self.df["timestamp"] >= active_start)
            & (self.df["timestamp"] < self._current_time)
        )
        recent = self.df.loc[recent_mask]
        active_devs = set(recent["email"].unique().tolist())

        # 現在処理中のイベントの owner（PR 作成者）を除外
        owner_email: Optional[str] = None
        if self._step_idx < len(self._events):
            event = self._events.iloc[self._step_idx]
            if "owner_email" in event.index:
                owner_email = event.get("owner_email")

        mask = np.array(
            [
                (dev_id in active_devs) and (dev_id != owner_email)
                for dev_id in self.developer_ids
            ],
            dtype=bool,
        )

        # 候補が少なすぎる場合は直近活動が多い順に補充して min_candidates を確保
        if int(mask.sum()) < self.min_candidates and len(self.developer_ids) > 0:
            # 直近 active_window_days の活動回数で開発者をランキング
            counts = recent["email"].value_counts()
            ranked = [
                d for d in counts.index.tolist() if d in set(self.developer_ids)
            ]
            # フォールバック: 全開発者から先頭を補う
            if len(ranked) < self.min_candidates:
                for d in self.developer_ids:
                    if d not in ranked:
                        ranked.append(d)
                        if len(ranked) >= self.min_candidates:
                            break

            for d in ranked[: self.min_candidates]:
                if d == owner_email:
                    continue
                idx = self.developer_ids.index(d)
                mask[idx] = True

            # それでも全員 False になる場合は最初の1人を強制的に有効化（学習が止まらないように）
            if not mask.any():
                mask[0] = True

        return mask

    # ── 便利メソッド ─────────────────────────────────────────────────────

    def get_developer_index(self, developer_id: str) -> int:
        """
        開発者のメールアドレス → 行動インデックスに変換する。

        使い方:
            idx = env.get_developer_index("alice@example.com")
            # → 例: 3  （developer_ids の3番目にいる場合）
        """
        return self.developer_ids.index(developer_id)

    def get_developer_id(self, index: int) -> str:
        """
        行動インデックス → 開発者のメールアドレスに変換する。

        使い方:
            dev = env.get_developer_id(3)
            # → "alice@example.com"
        """
        return self.developer_ids[index]
