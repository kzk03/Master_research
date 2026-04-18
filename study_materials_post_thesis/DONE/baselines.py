"""
ベースライン推薦エージェント（RL/agent/baselines.py）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このファイルの役割
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RL エージェント (MaskablePPO) の比較対象となる、学習なしの単純な
推薦戦略をまとめたもの。Step 2 の評価軸A（史実再現率）で、
RL がベースラインに勝てているかを定量的に確認するために使う。

提供する戦略:
  1. RandomBaseline   - マスクされていない候補からランダムに選ぶ
  2. RoundRobinBaseline - マスクされた候補を順番に回す
  3. RecencyBaseline  - 直近活動が多い開発者を選ぶ

全てのベースラインは ReviewAgent と同じ predict() / evaluate() の
インターフェースを持つので、評価コードから差し替えるだけで使える。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from RL.env.review_env import ReviewEnv

logger = logging.getLogger(__name__)


class BaselineAgent:
    """ベースラインの共通インターフェース。"""

    def __init__(self, env: ReviewEnv, seed: int = 0) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)

    def select_action(self, mask: np.ndarray) -> int:
        raise NotImplementedError

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[int, None]:
        mask = self.env.action_masks()
        return self.select_action(mask), None

    def evaluate(self, n_episodes: int = 5) -> Dict[str, list]:
        results: Dict[str, list] = {
            "rewards": [],
            "accepted_counts": [],
            "n_steps": [],
        }
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=int(self.rng.integers(0, 1_000_000)))
            total_reward = 0.0
            accepted_count = 0
            steps = 0
            done = False
            while not done:
                action, _ = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                if info.get("accepted"):
                    accepted_count += 1
                steps += 1
                done = terminated or truncated
            results["rewards"].append(total_reward)
            results["accepted_counts"].append(accepted_count)
            results["n_steps"].append(steps)
        return results


class RandomBaseline(BaselineAgent):
    """マスク後の候補からランダムに1人選ぶ。"""

    def select_action(self, mask: np.ndarray) -> int:
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return 0
        return int(self.rng.choice(valid_indices))


class RoundRobinBaseline(BaselineAgent):
    """マスク後の候補を順番に回す（負荷分散の単純戦略）。"""

    def __init__(self, env: ReviewEnv, seed: int = 0) -> None:
        super().__init__(env, seed=seed)
        self._cursor = 0

    def select_action(self, mask: np.ndarray) -> int:
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return 0
        # 候補の中でカーソル以降の最初の有効インデックスを選ぶ
        self._cursor = (self._cursor + 1) % len(valid_indices)
        return int(valid_indices[self._cursor])


class RecencyBaseline(BaselineAgent):
    """
    現在時刻から見て直近 `window_days` 日に最も活発だった開発者を選ぶ。

    「最近触ってる人に振る」というよくある経験則を表現している。
    マスクが適用された候補に限定して選ぶ。
    """

    def __init__(
        self, env: ReviewEnv, window_days: int = 30, seed: int = 0
    ) -> None:
        super().__init__(env, seed=seed)
        self.window_days = window_days

    def select_action(self, mask: np.ndarray) -> int:
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return 0

        current_time = self.env._current_time
        start = current_time - pd.Timedelta(days=self.window_days)
        recent = self.env.df[
            (self.env.df["timestamp"] >= start)
            & (self.env.df["timestamp"] < current_time)
        ]
        counts = recent["email"].value_counts()

        # マスク済み候補を活動回数で降順ソート
        candidate_devs: List[Tuple[int, int]] = []
        for idx in valid_indices:
            dev_id = self.env.developer_ids[idx]
            candidate_devs.append((idx, int(counts.get(dev_id, 0))))

        candidate_devs.sort(key=lambda x: -x[1])
        return int(candidate_devs[0][0])


class PathAffinityBaseline(BaselineAgent):
    """
    現在のタスクが触るディレクトリを過去にレビューしたことが多い開発者を選ぶ。

    df に 'dirs' 列が必要 (path_features.attach_dirs_to_df で付与)。
    Step 3 で導入したファイルパス特徴量の有効性を学習なしで確認する
    シンプルな比較対象。
    """

    def __init__(
        self,
        env: ReviewEnv,
        window_days: int = 180,
        seed: int = 0,
    ) -> None:
        super().__init__(env, seed=seed)
        self.window_days = window_days
        if "dirs" not in env.df.columns:
            raise ValueError(
                "PathAffinityBaseline には df['dirs'] 列が必要です。"
                "IRL.features.path_features.attach_dirs_to_df() を先に呼んでください。"
            )

    def select_action(self, mask: np.ndarray) -> int:
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return 0

        task_dirs = self.env._get_current_task_dirs() or frozenset()
        if not task_dirs:
            # フォールバック: Recency と同じ振る舞い
            return int(valid_indices[0])

        current_time = self.env._current_time
        start = current_time - pd.Timedelta(days=self.window_days)
        sub = self.env.df[
            (self.env.df["timestamp"] >= start)
            & (self.env.df["timestamp"] < current_time)
        ]
        # ディレクトリ交差を持つ行に絞る
        overlap = sub[sub["dirs"].map(
            lambda ds: bool(ds) and not ds.isdisjoint(task_dirs)
        )]
        counts = overlap["email"].value_counts()

        # 0 件しかいない場合は Recency にフォールバック
        if counts.empty:
            return int(valid_indices[0])

        candidate_devs: List[Tuple[int, int]] = []
        for idx in valid_indices:
            dev_id = self.env.developer_ids[idx]
            candidate_devs.append((idx, int(counts.get(dev_id, 0))))

        candidate_devs.sort(key=lambda x: -x[1])
        return int(candidate_devs[0][0])
