"""
ReviewEnv のスモークテスト

目的: M1配線（IRLReward._load_model / .compute、ReviewEnv.step の承諾シミュレーション）
が壊れていないか最小限の動作確認をする。

使い方:
    cd /Users/kazuki-h/Master_research
    python -m pytest tests/RL/test_env.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# RL/IRL モジュールは src/review_predictor を sys.path に置く前提で
# `from IRL.xxx`, `from RL.xxx` の形で書かれているのでここで補う。
ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT / "src" / "review_predictor"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from RL.env.review_env import ReviewEnv  # noqa: E402
from RL.reward.reward import IRLReward, RewardFunction  # noqa: E402
from RL.state.state_builder import StateBuilder  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# テスト用の最小フィクスチャ
# ─────────────────────────────────────────────────────────────────────


def _make_dummy_df() -> pd.DataFrame:
    """
    ReviewEnv が要求する最小カラムを持つダミー DataFrame を生成。
    common_features.extract_common_features が参照するカラムだけ用意する。
    """
    base = datetime(2024, 1, 1)
    rows = []
    devs = ["alice@example.com", "bob@example.com", "carol@example.com"]
    for i in range(60):
        dev = devs[i % 3]
        owner = devs[(i + 1) % 3]
        rows.append(
            {
                "email": dev,
                "owner_email": owner,
                "timestamp": base + timedelta(days=i),
                "label": int(i % 2 == 0),
                "response_latency_days": float(i % 7),
                "change_id": f"chg{i}",
                "lines_added": 10 + i,
                "lines_deleted": 5,
                "files_changed": 2,
                "subject": f"test {i}",
            }
        )
    return pd.DataFrame(rows)


class _ConstReward(RewardFunction):
    """IRL モデルに依存しない定数報酬。step ループ自体の動作確認に使う。"""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def compute(self, state_vector, developer_id, action) -> float:  # type: ignore[override]
        return self.value


# ─────────────────────────────────────────────────────────────────────
# テスト
# ─────────────────────────────────────────────────────────────────────


def test_state_builder_obs_dim_matches_features():
    """StateBuilder.obs_dim が common_features の次元数（25）と一致する。"""
    builder = StateBuilder(window_days=30, normalize=True)
    assert builder.obs_dim == 25


def test_review_env_random_episode_with_const_reward():
    """
    定数報酬で 1 エピソードがエラーなく回ること。
    （IRL モデル不要のスモークテスト）
    """
    df = _make_dummy_df()
    builder = StateBuilder(window_days=30, normalize=True)
    reward_fn = _ConstReward(value=0.7)
    env = ReviewEnv(
        df=df,
        reward_fn=reward_fn,
        state_builder=builder,
        eval_start=datetime(2024, 2, 1),
        eval_end=datetime(2024, 2, 20),
        max_steps=5,
    )

    obs, info = env.reset(seed=42)
    assert obs.shape == (env.n_developers * builder.obs_dim,)

    total_reward = 0.0
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (env.n_developers * builder.obs_dim,)
        assert isinstance(reward, float)
        assert "accepted" in info
        assert "acceptance_prob" in info
        total_reward += reward
        if terminated or truncated:
            break

    # 定数報酬 (0.7) × 開発者数 でステップ毎に発生するため、最低でも > 0
    assert total_reward > 0.0


@pytest.mark.skipif(
    not (
        Path(
            "/Users/kazuki-h/Master_research/outputs/cross_temporal_v39/train_0-3m/irl_model.pt"
        ).exists()
    ),
    reason="IRL モデル成果物が無いとロード不可",
)
def test_irl_reward_loads_and_computes():
    """
    実際の IRL モデル (.pt) をロードして compute() が 0〜1 のスコアを返すこと。
    """
    model_path = (
        "/Users/kazuki-h/Master_research/outputs/cross_temporal_v39/train_0-3m/irl_model.pt"
    )
    reward_fn = IRLReward(model_path=model_path, workload_penalty_weight=0.0, device="cpu")

    # state_dim 20 + action_dim 5 = 25 次元
    state_vector = np.linspace(0.1, 0.9, 25).astype(np.float32)
    score = reward_fn.compute(
        state_vector=state_vector,
        developer_id="alice@example.com",
        action={"developer_id": "alice@example.com", "step": 0},
    )
    assert isinstance(score, float)
    # 継続確率 (0〜1) - ペナルティ (0) なので 0〜1 範囲のはず
    assert -0.01 <= score <= 1.01


@pytest.mark.skipif(
    not (
        Path(
            "/Users/kazuki-h/Master_research/outputs/cross_temporal_v39/train_0-3m/irl_model.pt"
        ).exists()
    ),
    reason="IRL モデル成果物が無いとロード不可",
)
def test_review_env_with_irl_reward_smoke():
    """
    IRLReward をつないだ状態でランダム推薦エージェントが 1 エピソード完走できること。
    """
    df = _make_dummy_df()
    builder = StateBuilder(window_days=30, normalize=True)
    reward_fn = IRLReward(
        model_path=(
            "/Users/kazuki-h/Master_research/outputs/"
            "cross_temporal_v39/train_0-3m/irl_model.pt"
        ),
        workload_penalty_weight=0.1,
        device="cpu",
    )
    env = ReviewEnv(
        df=df,
        reward_fn=reward_fn,
        state_builder=builder,
        eval_start=datetime(2024, 2, 1),
        eval_end=datetime(2024, 2, 20),
        max_steps=3,
    )

    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.n_developers * builder.obs_dim,)

    accepted_count = 0
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        accepted_count += int(info["accepted"])
        assert 0.0 <= info["acceptance_prob"] <= 1.0
        assert obs.shape == (env.n_developers * builder.obs_dim,)
        if terminated or truncated:
            break

    # 必ずしも承諾するとは限らないが、少なくともキー情報が欠落していなければ OK
    assert accepted_count >= 0


def test_action_masks_shape_and_validity():
    """action_masks() が n_developers の bool 配列を返し、最低 1 人は valid。"""
    df = _make_dummy_df()
    builder = StateBuilder(window_days=30, normalize=True)
    env = ReviewEnv(
        df=df,
        reward_fn=_ConstReward(value=0.5),
        state_builder=builder,
        eval_start=datetime(2024, 2, 1),
        eval_end=datetime(2024, 2, 20),
        max_steps=5,
        active_window_days=30,
        min_candidates=2,
    )
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask.shape == (env.n_developers,)
    assert mask.dtype == bool
    assert mask.any(), "全候補が無効になっている"


def test_baseline_agents_run():
    """ベースラインエージェント 3 種が 1 エピソード完走できる。"""
    from RL.agent.baselines import (
        RandomBaseline,
        RecencyBaseline,
        RoundRobinBaseline,
    )

    df = _make_dummy_df()
    builder = StateBuilder(window_days=30, normalize=True)
    env = ReviewEnv(
        df=df,
        reward_fn=_ConstReward(value=0.5),
        state_builder=builder,
        eval_start=datetime(2024, 2, 1),
        eval_end=datetime(2024, 2, 20),
        max_steps=4,
        active_window_days=30,
        min_candidates=2,
    )
    for agent_cls in (RandomBaseline, RoundRobinBaseline, RecencyBaseline):
        agent = agent_cls(env, seed=0)
        results = agent.evaluate(n_episodes=1)
        assert len(results["rewards"]) == 1
        assert results["n_steps"][0] >= 1


def test_maskable_ppo_short_train():
    """
    MaskablePPO が ReviewAgent 経由で短時間でも学習が回ること。
    """
    from RL.agent.agent import ReviewAgent

    df = _make_dummy_df()
    builder = StateBuilder(window_days=30, normalize=True)
    env = ReviewEnv(
        df=df,
        reward_fn=_ConstReward(value=0.5),
        state_builder=builder,
        eval_start=datetime(2024, 2, 1),
        eval_end=datetime(2024, 2, 20),
        max_steps=4,
        active_window_days=30,
        min_candidates=2,
    )
    agent = ReviewAgent(env=env, n_steps=8, verbose=0)
    agent.train(total_timesteps=16)
    results = agent.evaluate(n_episodes=1)
    assert len(results["rewards"]) == 1
