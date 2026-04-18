"""
RL エージェント学習スクリプト（Step 2: MaskablePPO ベース）

使い方:
    python scripts/train/train_rl_agent.py \
        --data data/nova_raw.csv \
        --irl-model outputs/cross_temporal_v39/train_0-3m/irl_model.pt \
        --eval-start 2014-01-01 --eval-end 2014-02-01 \
        --max-steps 200 --total-timesteps 5000 \
        --save outputs/rl_agent/maskable_ppo

学習が回るかの動作確認用にハイパラはかなり小さめのデフォルトにしている。
本格学習の際は --total-timesteps を大きくすること。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT / "src" / "review_predictor"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from IRL.features.path_features import (  # noqa: E402
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map,
)
from RL.agent.agent import ReviewAgent  # noqa: E402
from RL.agent.baselines import (  # noqa: E402
    PathAffinityBaseline,
    RandomBaseline,
    RecencyBaseline,
    RoundRobinBaseline,
)
from RL.env.review_env import ReviewEnv  # noqa: E402
from RL.reward.reward import IRLReward  # noqa: E402
from RL.state.state_builder import StateBuilder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/nova_raw.csv"))
    p.add_argument(
        "--irl-model",
        type=Path,
        default=Path("outputs/cross_temporal_v39/train_0-3m/irl_model.pt"),
    )
    p.add_argument("--eval-start", type=str, default="2014-01-01")
    p.add_argument("--eval-end", type=str, default="2014-02-01")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--total-timesteps", type=int, default=5_000)
    p.add_argument("--n-eval-episodes", type=int, default=3)
    p.add_argument("--workload-penalty", type=float, default=0.1)
    p.add_argument("--active-window-days", type=int, default=90)
    p.add_argument("--min-candidates", type=int, default=5)
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--raw-json",
        type=Path,
        default=Path("data/raw_json/openstack__nova.json"),
        help="ファイルパス特徴量のソース raw json",
    )
    p.add_argument(
        "--use-path-features",
        action="store_true",
        help="StateBuilder にディレクトリ親和度特徴量を含める",
    )
    p.add_argument("--path-window-days", type=int, default=180)
    p.add_argument("--path-depth", type=int, default=2)
    p.add_argument(
        "--use-task-features",
        action="store_true",
        help="obs の先頭にタスク特徴量 (insertions, deletions, files_count, is_cross_project) を追加",
    )
    p.add_argument(
        "--hit-bonus-weight",
        type=float,
        default=0.0,
        help="史実一致シェーピング: 推薦先 == 史実 reviewer のとき加算する報酬ボーナス",
    )
    p.add_argument(
        "--irl-reward-weight",
        type=float,
        default=1.0,
        help="IRL 継続確率総和に掛ける係数（hit_bonus と同スケールにしたい場合は小さくする）",
    )
    return p.parse_args()


def make_env(args: argparse.Namespace) -> ReviewEnv:
    logger.info(f"データを読み込み: {args.data}")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    eval_start = datetime.fromisoformat(args.eval_start)
    eval_end = datetime.fromisoformat(args.eval_end)

    # 評価期間内に活動した開発者だけに絞ると行動空間が現実的になる
    active_mask = (df["timestamp"] >= eval_start) & (df["timestamp"] < eval_end)
    developer_ids = sorted(df.loc[active_mask, "email"].dropna().unique().tolist())
    logger.info(
        f"評価期間 {eval_start} 〜 {eval_end} の活動開発者数: {len(developer_ids)}"
    )

    path_extractor: Optional[PathFeatureExtractor] = None
    if args.use_path_features:
        cdm = load_change_dir_map(args.raw_json, depth=args.path_depth)
        df = attach_dirs_to_df(df, cdm)
        path_extractor = PathFeatureExtractor(df, window_days=args.path_window_days)
        logger.info(
            f"path features 有効: window={args.path_window_days}d depth={args.path_depth}"
        )

    state_builder = StateBuilder(
        window_days=args.window_days,
        normalize=True,
        path_extractor=path_extractor,
    )
    reward_fn = IRLReward(
        model_path=args.irl_model,
        workload_penalty_weight=args.workload_penalty,
        device=args.device,
    )
    env = ReviewEnv(
        df=df,
        reward_fn=reward_fn,
        state_builder=state_builder,
        developer_ids=developer_ids,
        eval_start=eval_start,
        eval_end=eval_end,
        max_steps=args.max_steps,
        active_window_days=args.active_window_days,
        min_candidates=args.min_candidates,
        hit_bonus_weight=args.hit_bonus_weight,
        irl_reward_weight=args.irl_reward_weight,
        use_task_features=args.use_task_features,
    )
    return env


def evaluate_baselines(env: ReviewEnv, n_episodes: int, seed: int) -> None:
    logger.info("=== ベースライン評価 ===")
    for name, agent in [
        ("Random", RandomBaseline(env, seed=seed)),
        ("RoundRobin", RoundRobinBaseline(env, seed=seed)),
        ("Recency", RecencyBaseline(env, window_days=30, seed=seed)),
    ]:
        results = agent.evaluate(n_episodes=n_episodes)
        avg_reward = sum(results["rewards"]) / max(len(results["rewards"]), 1)
        avg_accept = sum(results["accepted_counts"]) / max(
            len(results["accepted_counts"]), 1
        )
        logger.info(
            f"{name:>10}: avg_reward={avg_reward:.3f} avg_accepted={avg_accept:.2f}"
        )


def main() -> None:
    args = parse_args()
    env = make_env(args)

    # 1) ベースライン評価
    evaluate_baselines(env, args.n_eval_episodes, args.seed)

    # 2) MaskablePPO 学習
    logger.info("=== MaskablePPO 学習 ===")
    agent = ReviewAgent(env=env, verbose=1, n_steps=min(256, args.max_steps * 2))
    agent.train(total_timesteps=args.total_timesteps)

    # 3) 学習後評価
    logger.info("=== 学習後 RL エージェント評価 ===")
    rl_results = agent.evaluate(n_episodes=args.n_eval_episodes)
    avg_reward = sum(rl_results["rewards"]) / max(len(rl_results["rewards"]), 1)
    avg_accept = sum(rl_results["accepted_counts"]) / max(
        len(rl_results["accepted_counts"]), 1
    )
    logger.info(
        f"MaskablePPO: avg_reward={avg_reward:.3f} avg_accepted={avg_accept:.2f}"
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(args.save))


if __name__ == "__main__":
    main()
