"""
ランダムフォレスト (RF) ベースラインの Hit Rate 評価

教師あり学習で「タスク × 開発者 → 史実 reviewer かどうか」を予測し、
各タスクでスコア上位K名を推薦して Hit Rate を計測する。

特徴量:
  - 開発者特徴: StateBuilder.build() の出力 (25次元 or 28次元)
  - タスク特徴: change_insertions, change_deletions, change_files_count, is_cross_project
  - path 特徴 (optional): PathFeatureExtractor の出力 (3次元)

使い方:
    python scripts/analyze/eval/eval_rf_hit_rate.py \
        --data data/nova_raw.csv \
        --eval-start 2020-06-01 --eval-end 2020-06-15
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
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT / "src" / "review_predictor"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from IRL.features.common_features import extract_common_features  # noqa: E402
from IRL.features.path_features import (  # noqa: E402
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/nova_raw.csv"))
    p.add_argument("--eval-start", type=str, default="2020-06-01")
    p.add_argument("--eval-end", type=str, default="2020-06-15")
    p.add_argument("--train-months", type=int, default=6,
                   help="eval_start から何ヶ月前を訓練開始とするか")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--active-window-days", type=int, default=90)
    p.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--n-trees", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--raw-json",
        type=Path,
        default=Path("data/raw_json/openstack__nova.json"),
    )
    p.add_argument("--use-path-features", action="store_true")
    p.add_argument("--path-window-days", type=int, default=180)
    p.add_argument("--path-depth", type=int, default=2)
    return p.parse_args()


# ── 特徴量構築 ───────────────────────────────────────

TASK_FEATURE_NAMES = [
    "change_insertions",
    "change_deletions",
    "change_files_count",
    "is_cross_project",
]


def build_task_features(event: pd.Series) -> np.ndarray:
    """イベント行からタスク特徴量を抽出する。"""
    return np.array([
        float(event.get("change_insertions", 0)),
        float(event.get("change_deletions", 0)),
        float(event.get("change_files_count", 0)),
        float(event.get("is_cross_project", False)),
    ], dtype=np.float32)


def build_developer_features(
    df: pd.DataFrame,
    developer_id: str,
    current_time: datetime,
    window_days: int,
) -> np.ndarray:
    """common_features.extract_common_features を呼んで開発者特徴を返す。"""
    feature_end = current_time
    feature_start = current_time - pd.Timedelta(days=window_days)
    feats = extract_common_features(
        df=df,
        email=developer_id,
        feature_start=feature_start,
        feature_end=feature_end,
    )
    return np.array(list(feats.values()), dtype=np.float32)


def build_sample(
    df: pd.DataFrame,
    event: pd.Series,
    developer_id: str,
    current_time: datetime,
    window_days: int,
    path_extractor: Optional[PathFeatureExtractor] = None,
    task_dirs: Optional[frozenset] = None,
) -> np.ndarray:
    """1つの (タスク, 開発者) ペアの特徴ベクトルを構築する。"""
    task_feat = build_task_features(event)
    dev_feat = build_developer_features(df, developer_id, current_time, window_days)
    parts = [task_feat, dev_feat]

    if path_extractor is not None and task_dirs is not None:
        path_feat = path_extractor.compute(developer_id, task_dirs, current_time)
        parts.append(path_feat)

    return np.concatenate(parts)


# ── メイン ───────────────────────────────────────────

def main() -> None:
    args = parse_args()

    logger.info(f"データを読み込み: {args.data}")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    eval_start = datetime.fromisoformat(args.eval_start)
    eval_end = datetime.fromisoformat(args.eval_end)
    train_start = eval_start - pd.DateOffset(months=args.train_months)

    # path features
    path_extractor: Optional[PathFeatureExtractor] = None
    if args.use_path_features:
        cdm = load_change_dir_map(args.raw_json, depth=args.path_depth)
        df = attach_dirs_to_df(df, cdm)
        path_extractor = PathFeatureExtractor(df, window_days=args.path_window_days)
        logger.info(f"path features 有効: window={args.path_window_days}d")

    # 評価期間のアクティブ開発者
    active_mask = (df["timestamp"] >= eval_start) & (df["timestamp"] < eval_end)
    developer_ids = sorted(df.loc[active_mask, "email"].dropna().unique().tolist())
    logger.info(f"開発者数: {len(developer_ids)}")

    # ── 訓練データ構築 ──────────────────────────────────
    train_mask = (
        (df["timestamp"] >= train_start)
        & (df["timestamp"] < eval_start)
    )
    train_events = df[train_mask].sort_values("timestamp").reset_index(drop=True)
    logger.info(f"訓練イベント数: {len(train_events)}")

    # 訓練サンプリング: 各イベントについて正例(史実reviewer) + 負例(ランダム数人)
    rng = np.random.default_rng(args.seed)
    X_train_list: List[np.ndarray] = []
    y_train_list: List[int] = []
    n_neg = 4  # 負例の数

    for idx in range(0, len(train_events), 5):  # 5件おきにサンプル（速度のため）
        event = train_events.iloc[idx]
        true_dev = event.get("email")
        current_time = pd.to_datetime(event["timestamp"]).to_pydatetime()

        if true_dev is None or true_dev not in developer_ids:
            continue

        task_dirs = None
        if path_extractor is not None and "dirs" in train_events.columns:
            val = event.get("dirs")
            if isinstance(val, frozenset):
                task_dirs = val
            elif isinstance(val, (set, list, tuple)):
                task_dirs = frozenset(val)
            else:
                task_dirs = frozenset()

        # 正例
        x_pos = build_sample(df, event, true_dev, current_time, args.window_days,
                             path_extractor, task_dirs)
        X_train_list.append(x_pos)
        y_train_list.append(1)

        # 負例
        neg_candidates = [d for d in developer_ids if d != true_dev]
        if len(neg_candidates) > n_neg:
            neg_candidates = list(rng.choice(neg_candidates, size=n_neg, replace=False))
        for neg_dev in neg_candidates:
            x_neg = build_sample(df, event, neg_dev, current_time, args.window_days,
                                 path_extractor, task_dirs)
            X_train_list.append(x_neg)
            y_train_list.append(0)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    logger.info(f"訓練サンプル数: {len(y_train)} (正例: {y_train.sum()}, 負例: {(1-y_train).sum()})")

    # ── RF 学習 ─────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=args.n_trees,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    logger.info("RF 学習完了")

    # ── 評価 ────────────────────────────────────────────
    eval_events = df[active_mask].sort_values("timestamp").reset_index(drop=True)
    if args.max_steps is not None:
        eval_events = eval_events.iloc[:args.max_steps]

    max_k = max(args.top_k)
    total = 0
    hits = {k: 0 for k in args.top_k}
    reciprocal_ranks: List[float] = []

    for idx in range(len(eval_events)):
        event = eval_events.iloc[idx]
        true_dev = event.get("email")
        current_time = pd.to_datetime(event["timestamp"]).to_pydatetime()

        if true_dev is None or true_dev not in developer_ids:
            continue

        task_dirs = None
        if path_extractor is not None and "dirs" in eval_events.columns:
            val = event.get("dirs")
            if isinstance(val, frozenset):
                task_dirs = val
            elif isinstance(val, (set, list, tuple)):
                task_dirs = frozenset(val)
            else:
                task_dirs = frozenset()

        # 全候補をスコアリング
        scores = []
        for dev_id in developer_ids:
            x = build_sample(df, event, dev_id, current_time, args.window_days,
                             path_extractor, task_dirs)
            prob = clf.predict_proba(x.reshape(1, -1))[0, 1]
            scores.append(prob)

        scores = np.array(scores)
        topk_indices = np.argsort(-scores)[:max_k]
        true_idx = developer_ids.index(true_dev)

        for k in args.top_k:
            if true_idx in topk_indices[:k]:
                hits[k] += 1

        if true_idx in topk_indices:
            rank = list(topk_indices).index(true_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        total += 1
        if total % 50 == 0:
            logger.info(f"  evaluated {total} events...")

    metrics = {
        f"hit@{k}": (hits[k] / total if total else 0.0) for k in args.top_k
    }
    metrics["mrr"] = (
        sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    )
    metrics["n_events"] = float(total)

    logger.info(f"[RF] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    if args.use_path_features:
        logger.info("[RF+path] 上記は path features あり")
    logger.info(f"=== サマリ ===")
    logger.info(f"RF: {metrics}")


if __name__ == "__main__":
    main()
