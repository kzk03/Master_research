"""
史実再現率（Hit Rate）評価スクリプト

評価軸A: モード①（タスク推薦）の妥当性証拠として、
RL エージェントが選んだ開発者と「史実で実際に担当した開発者」の
一致率を計測する。

各イベント（タスク発生）について:
  1. ReviewEnv の現在状態をエージェントに渡す
  2. エージェントが推薦する開発者（または上位K名）を取得
  3. 史実の reviewer（df の email カラム）と一致するかを判定

出力指標:
  - Top-1 Hit Rate
  - Top-3 / Top-5 Hit Rate（MaskablePPO の確率分布から事後算出）
  - MRR (Mean Reciprocal Rank)

使い方:
    python scripts/analyze/eval/eval_hit_rate.py \
        --data data/nova_raw.csv \
        --irl-model outputs/cross_temporal_v39/train_0-3m/irl_model.pt \
        --rl-model outputs/rl_agent/maskable_ppo \
        --eval-start 2014-01-01 --eval-end 2014-02-01 --max-steps 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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
    BaselineAgent,
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
    p.add_argument("--rl-model", type=Path, default=None)
    p.add_argument("--eval-start", type=str, default="2014-01-01")
    p.add_argument("--eval-end", type=str, default="2014-02-01")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--active-window-days", type=int, default=90)
    p.add_argument("--min-candidates", type=int, default=5)
    p.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--workload-penalty", type=float, default=0.0)
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
        help="StateBuilder にディレクトリ親和度特徴量を含め、PathAffinity baseline も評価する",
    )
    p.add_argument(
        "--path-window-days",
        type=int,
        default=180,
        help="path features の集計ウィンドウ（日数）",
    )
    p.add_argument(
        "--path-depth", type=int, default=2, help="ファイルパスをディレクトリ化する階層"
    )
    p.add_argument(
        "--use-task-features",
        action="store_true",
        help="obs の先頭にタスク特徴量を追加",
    )
    return p.parse_args()


def build_env(args: argparse.Namespace) -> ReviewEnv:
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    eval_start = datetime.fromisoformat(args.eval_start)
    eval_end = datetime.fromisoformat(args.eval_end)

    active_mask = (df["timestamp"] >= eval_start) & (df["timestamp"] < eval_end)
    developer_ids = sorted(df.loc[active_mask, "email"].dropna().unique().tolist())

    path_extractor: Optional[PathFeatureExtractor] = None
    if args.use_path_features:
        cdm = load_change_dir_map(args.raw_json, depth=args.path_depth)
        df = attach_dirs_to_df(df, cdm)
        path_extractor = PathFeatureExtractor(
            df, window_days=args.path_window_days
        )
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
    return ReviewEnv(
        df=df,
        reward_fn=reward_fn,
        state_builder=state_builder,
        developer_ids=developer_ids,
        eval_start=eval_start,
        eval_end=eval_end,
        max_steps=args.max_steps,
        active_window_days=args.active_window_days,
        min_candidates=args.min_candidates,
        use_task_features=getattr(args, "use_task_features", False),
    )


def topk_indices_from_scores(
    scores: np.ndarray, mask: np.ndarray, k: int
) -> List[int]:
    """マスク済みスコアから上位 k のインデックスを返す。"""
    masked = np.where(mask, scores, -np.inf)
    if k >= len(masked):
        order = np.argsort(-masked)
    else:
        order = np.argpartition(-masked, k)[:k]
        order = order[np.argsort(-masked[order])]
    return [int(i) for i in order if mask[i]]


def evaluate_hit_rate(
    env: ReviewEnv,
    agent_name: str,
    pick_topk_fn,
    top_ks: List[int],
) -> Dict[str, float]:
    """
    pick_topk_fn(obs, mask, k) → 上位 k インデックスのリスト を渡すと、
    各イベントで史実 reviewer との一致を計測する。
    """
    obs, _ = env.reset(seed=0)
    total = 0
    hits = {k: 0 for k in top_ks}
    reciprocal_ranks: List[float] = []

    done = False
    while not done:
        # 現在のイベントの真の reviewer
        if env._step_idx >= len(env._events):
            break
        true_event = env._events.iloc[env._step_idx]
        true_dev: Optional[str] = true_event.get("email")
        if true_dev is None or true_dev not in env.developer_ids:
            # 評価対象外（マスクで除外される可能性が高い）→ ステップは進める
            action = 0
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            continue

        mask = env.action_masks()
        max_k = max(top_ks)
        topk = pick_topk_fn(obs, mask, max_k)

        true_idx = env.developer_ids.index(true_dev)
        for k in top_ks:
            if true_idx in topk[:k]:
                hits[k] += 1
        # MRR
        if true_idx in topk:
            rank = topk.index(true_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        total += 1

        # 適当な行動でステップを進める（評価軸が hit rate なのでどれでもよいが、
        # Top-1 を選んで進めると一貫性がある）
        action = topk[0] if topk else 0
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    metrics = {
        f"hit@{k}": (hits[k] / total if total else 0.0) for k in top_ks
    }
    metrics["mrr"] = (
        sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    )
    metrics["n_events"] = float(total)
    logger.info(
        f"[{agent_name}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    )
    return metrics


def make_baseline_pick_fn(agent: BaselineAgent):
    """ベースラインを Top-K 評価に対応させるラッパー。"""

    def pick(obs, mask, k):
        # 1人だけ選ぶ → そこから他は activity 順で埋める
        primary = agent.select_action(mask)
        topk = [primary]
        valid = [i for i in np.flatnonzero(mask) if i != primary]
        topk.extend(valid[: k - 1])
        return topk[:k]

    return pick


def make_rl_pick_fn(agent: ReviewAgent):
    """MaskablePPO の方策確率から Top-K を取り出す。"""
    import torch

    def pick(obs, mask, k):
        # MaskablePPO は obs から行動分布を返せる
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            distribution = agent._model.policy.get_distribution(
                obs_tensor, action_masks=mask_tensor
            )
            probs = distribution.distribution.probs.squeeze(0).cpu().numpy()
        return topk_indices_from_scores(probs, mask, k)

    return pick


def main() -> None:
    args = parse_args()
    env = build_env(args)

    all_results: Dict[str, Dict[str, float]] = {}

    # ベースライン
    baselines: List[tuple] = [
        ("Random", RandomBaseline(env, seed=args.seed)),
        ("RoundRobin", RoundRobinBaseline(env, seed=args.seed)),
        ("Recency", RecencyBaseline(env, window_days=30, seed=args.seed)),
    ]
    if args.use_path_features:
        baselines.append(
            (
                "PathAffinity",
                PathAffinityBaseline(
                    env, window_days=args.path_window_days, seed=args.seed
                ),
            )
        )
    for name, base in baselines:
        all_results[name] = evaluate_hit_rate(
            env, name, make_baseline_pick_fn(base), args.top_k
        )

    # RL エージェント（モデルが指定されていれば）
    if args.rl_model is not None and Path(str(args.rl_model) + ".zip").exists():
        rl_agent = ReviewAgent(env=env, verbose=0)
        rl_agent.load(str(args.rl_model))
        all_results["MaskablePPO"] = evaluate_hit_rate(
            env, "MaskablePPO", make_rl_pick_fn(rl_agent), args.top_k
        )
    else:
        logger.info("RL モデル未指定、または見つからないためスキップ")

    # サマリ表示
    logger.info("=== サマリ ===")
    for name, metrics in all_results.items():
        logger.info(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
