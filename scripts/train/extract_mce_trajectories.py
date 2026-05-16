#!/usr/bin/env python3
"""
MCE-IRL 用の軌跡抽出スクリプト (joblib 並列化対応)

extract_trajectories.py とほぼ同じ軌跡を出力するが、MCE-IRL 用に二値 action 列
(step_actions = step_labels の 0/1 化) を併せて pickle にキャッシュする。
キャッシュ形式は MCEIRLSystem._precompute_trajectories と互換。

使い方:
    uv run python scripts/train/extract_mce_trajectories.py \
        --reviews data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --train-start 2019-01-01 --train-end 2022-01-01 \
        --future-window-start 0 --future-window-end 3 \
        --n-jobs -1 \
        --output outputs/mce_irl_trajectory_cache/mce_traj_0-3.pkl

NOTE: 出力パスは run_mce_irl_variant_single.sh の CACHE_DIR
(``outputs/mce_irl_trajectory_cache``) と必ず一致させる。Focal-supervised の
``outputs/trajectory_cache/`` とは別のディレクトリにすること。
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts" / "train"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="MCE-IRL 用軌跡抽出 & キャッシュ保存"
    )
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument("--train-start", type=str, default="2019-01-01")
    parser.add_argument("--train-end", type=str, default="2022-01-01")
    parser.add_argument("--future-window-start", type=int, required=True)
    parser.add_argument("--future-window-end", type=int, required=True)
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="joblib 並列数（-1=全コア）")
    parser.add_argument("--output", type=str, required=True, help="出力 .pkl パス")
    # Phase 2 (2026-05-15): co-change graph 由来の path 特徴量用 CSV を渡す
    # 未指定なら path_hub_score / path_neighbor_coverage は 0.0 (後方互換)
    parser.add_argument("--hub-scores", type=str, default=None,
                        help="experiments/dependency_analysis/results/hub_scores_main32.csv")
    parser.add_argument("--cochange-neighbors", type=str, default=None,
                        help="experiments/dependency_analysis/results/cochange_neighbors_main32.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        logger.info(f"キャッシュ済み、スキップ: {output_path}")
        return

    from train_model import (
        extract_directory_level_trajectories,
        load_review_requests,
    )
    from review_predictor.IRL.features.path_features import (
        PathFeatureExtractor,
        attach_dirs_to_df,
        load_change_dir_map,
        load_change_dir_map_multi,
    )

    # データ読み込み
    df = load_review_requests(args.reviews)
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)

    # ディレクトリマッピング
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    df = attach_dirs_to_df(df, cdm, column="dirs")

    df_for_path = df.rename(columns={
        "reviewer_email": "email",
        "request_time": "timestamp",
    })
    path_extractor = PathFeatureExtractor(
        df_for_path,
        window_days=180,
        hub_scores_path=args.hub_scores,
        cochange_neighbors_path=args.cochange_neighbors,
    )

    # 軌跡抽出（並列化）
    logger.info(f"軌跡抽出開始 (multitask={args.multitask}, n_jobs={args.n_jobs})")
    trajectories = extract_directory_level_trajectories(
        df,
        train_start=train_start,
        train_end=train_end,
        path_extractor=path_extractor,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        multitask=args.multitask,
        n_jobs=args.n_jobs,
    )

    if not trajectories:
        logger.error("軌跡が抽出できませんでした")
        return

    # MCE-IRL 用に step_actions (二値 action 列) を付与
    # step_labels は通常 0/1 だが、念のため bool キャストで 0/1 に正規化する。
    n_pos_steps = 0
    n_total_steps = 0
    for traj in trajectories:
        labels = traj.get("step_labels", []) or []
        actions = [int(bool(l)) for l in labels]
        traj["step_actions"] = actions
        n_pos_steps += sum(actions)
        n_total_steps += len(actions)
    if n_total_steps:
        logger.info(
            f"二値 action ステップ統計: 正例 {n_pos_steps}/{n_total_steps} "
            f"({n_pos_steps / n_total_steps * 100:.1f}%)"
        )

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    logger.info(
        f"保存完了 [MCE-IRL]: {output_path} ({len(trajectories)} 軌跡)"
    )


if __name__ == "__main__":
    main()
