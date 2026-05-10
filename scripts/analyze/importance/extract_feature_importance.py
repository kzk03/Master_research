#!/usr/bin/env python3
"""
保存済みIRLモデル (.pt) から特徴量重要度を事後的に抽出するスクリプト。

gradient-based importance を計算し、各 train_*/eval_*/ に
irl_feature_importance.json を保存する。
RF_Dir の Gini importance も同時に計算・保存できる。

使い方:
    # IRL のみ（variant_comparison_server の全モデル）
    uv run python scripts/analyze/importance/extract_feature_importance.py \
        --base-dir outputs/variant_comparison_server/lstm_baseline \
        --data data/combined_raw.csv \
        --raw-json data/raw_json/openstack__nova.json \
                   data/raw_json/openstack__cinder.json \
                   data/raw_json/openstack__neutron.json \
                   data/raw_json/openstack__ironic.json \
                   data/raw_json/openstack__glance.json \
                   data/raw_json/openstack__keystone.json \
                   data/raw_json/openstack__horizon.json \
                   data/raw_json/openstack__swift.json \
                   data/raw_json/openstack__heat.json \
                   data/raw_json/openstack__octavia.json

    # RF も同時に計算
    uv run python scripts/analyze/importance/extract_feature_importance.py \
        --base-dir outputs/variant_comparison_server/lstm_baseline \
        --data data/combined_raw.csv \
        --raw-json data/raw_json/openstack__nova.json ... \
        --include-rf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from review_predictor.IRL.features.common_features import (
    FEATURE_NAMES_WITH_PATH,
)
from review_predictor.IRL.features.directory_contributors import (
    get_directory_developers,
)
from review_predictor.IRL.features.path_features import (
    PATH_FEATURE_NAMES,
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map,
    load_change_dir_map_multi,
)
from review_predictor.IRL.model.batch_predictor import BatchContinuationPredictor
from review_predictor.IRL.model.rf_predictor import (
    extract_features_for_window_directory,
    prepare_rf_features_directory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 訓練窓の定義（run_variant_single.sh と同一）
TRAIN_WINDOWS = ["0-3m", "3-6m", "6-9m", "9-12m"]
FUTURE_START = [0, 3, 6, 9]
FUTURE_END = [3, 6, 9, 12]


def compute_irl_importance(
    model_path: Path,
    df: pd.DataFrame,
    path_extractor: PathFeatureExtractor,
    prediction_time: datetime,
    window_days: int = 180,
    max_samples: int = 300,
    head_index: int = 0,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """保存済み IRL モデルから gradient-based importance を計算する。

    directory-level モデル (state_dim=23) に対応するため、
    BatchContinuationPredictor の月次テンソル構築ロジックを利用し、
    path_features 付きで勾配を計算する。
    """
    import torch
    from review_predictor.IRL.features.common_features import (
        STATE_FEATURES,
        ACTION_FEATURES,
    )
    from review_predictor.IRL.features.path_features import PATH_FEATURE_NAMES

    history_start = (
        pd.Timestamp(prediction_time) - pd.DateOffset(months=24)
    ).to_pydatetime()

    predictor = BatchContinuationPredictor(
        model_path=model_path,
        df=df,
        history_start=history_start,
        device="cpu",
    )
    predictor._load_model()
    irl_system = predictor._irl_system
    is_dir_model = irl_system.state_dim > len(STATE_FEATURES)

    # ディレクトリ単位の開発者リストを取得
    window_start = prediction_time - pd.Timedelta(days=window_days)
    dir_developers = get_directory_developers(df, window_start, prediction_time)

    # (dev, dir) ペアから軌跡を構築
    pairs: List[tuple] = []
    for d, devs in dir_developers.items():
        for dev in devs:
            pairs.append((dev, d))

    # サンプル数を制限（ランダム選択）
    rng = np.random.RandomState(42)
    if len(pairs) > max_samples:
        indices = rng.choice(len(pairs), max_samples, replace=False)
        pairs = [pairs[i] for i in indices]

    logger.info(f"IRL importance: {len(pairs)} ペアで勾配計算 "
                f"(state_dim={irl_system.state_dim}, dir_model={is_dir_model})...")

    # 勾配の蓄積
    all_grads_abs: List[np.ndarray] = []
    all_grads_signed: List[np.ndarray] = []
    irl_system.network.eval()

    for dev, d in pairs:
        monthly_data = predictor._build_monthly_data(dev, prediction_time)
        monthly_histories, step_dates, step_reviews, dev_info = monthly_data
        if not monthly_histories or dev_info is None:
            continue

        # 各月ステップのパス特徴量
        step_path_features = None
        if is_dir_model:
            step_path_features = []
            task_dirs = frozenset({d})
            for ctx_date in step_dates:
                pf = path_extractor.compute(dev, task_dirs, ctx_date)
                step_path_features.append(pf)

        # 月次テンソルを構築
        email = dev_info.get("email", dev_info.get("developer_id", dev))
        state_tensors = []
        action_tensors = []
        for i, (month_history, month_ctx) in enumerate(
            zip(monthly_histories, step_dates)
        ):
            if not month_history:
                state_tensors.append(torch.zeros(irl_system.state_dim))
                action_tensors.append(torch.zeros(irl_system.action_dim))
                continue
            total_proj = step_reviews[i] if step_reviews and i < len(step_reviews) else 0
            pf = (step_path_features[i]
                  if step_path_features and i < len(step_path_features) else None)
            s_t, a_t = irl_system.extract_features_tensor(
                email, month_history, month_ctx,
                total_project_reviews=total_proj,
                path_features_vec=pf,
            )
            state_tensors.append(s_t)
            action_tensors.append(a_t)

        if not state_tensors:
            continue

        state_seq = torch.stack(state_tensors).unsqueeze(0).requires_grad_(True)
        action_seq = torch.stack(action_tensors).unsqueeze(0).requires_grad_(True)
        lengths = torch.tensor([len(state_tensors)], dtype=torch.long)

        _, continuation = irl_system.network(state_seq, action_seq, lengths)
        continuation.backward()

        s_grad_raw = state_seq.grad.mean(dim=(0, 1)).detach().cpu().numpy()
        a_grad_raw = action_seq.grad.mean(dim=(0, 1)).detach().cpu().numpy()
        all_grads_signed.append(np.concatenate([s_grad_raw, a_grad_raw]))

        s_grad_abs = state_seq.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
        a_grad_abs = action_seq.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
        all_grads_abs.append(np.concatenate([s_grad_abs, a_grad_abs]))

    if not all_grads_abs:
        logger.warning("勾配計算できたサンプルが0件")
        return {}, {}

    # 特徴量名の構築
    state_names = list(STATE_FEATURES)
    if is_dir_model:
        state_names = state_names + list(PATH_FEATURE_NAMES)
    action_names = list(ACTION_FEATURES)
    names = state_names + action_names

    # 絶対値版（影響度の大きさ、正規化済み）
    mean_abs = np.mean(all_grads_abs, axis=0)
    total = mean_abs.sum()
    if total > 0:
        mean_abs = mean_abs / total
    abs_importance = {name: float(val) for name, val in zip(names, mean_abs)}

    # 符号付き版（正=継続促進、負=離脱促進、正規化なし）
    mean_signed = np.mean(all_grads_signed, axis=0)
    signed_importance = {name: float(val) for name, val in zip(names, mean_signed)}

    logger.info(f"IRL importance: {len(abs_importance)} 特徴量, "
                f"{len(all_grads_abs)} サンプル")
    return abs_importance, signed_importance


def compute_rf_dir_importance(
    df: pd.DataFrame,
    path_extractor: PathFeatureExtractor,
    prediction_time: datetime,
    train_end: datetime,
    future_start_months: int,
    delta_months: int = 3,
    window_days: int = 180,
) -> Dict[str, float]:
    """RF_Dir の Gini importance を計算する。"""
    from sklearn.ensemble import RandomForestClassifier

    rf_train_end = pd.Timestamp(train_end)
    rf_train_start = rf_train_end - pd.Timedelta(days=window_days)
    rf_label_start = rf_train_end + pd.DateOffset(months=future_start_months)
    rf_label_end = rf_label_start + pd.DateOffset(months=delta_months)

    train_df = extract_features_for_window_directory(
        df,
        pd.Timestamp(rf_train_start),
        pd.Timestamp(rf_train_end),
        pd.Timestamp(rf_label_start),
        pd.Timestamp(rf_label_end),
        path_extractor=path_extractor,
    )

    if len(train_df) < 10 or len(train_df["label"].unique()) < 2:
        logger.warning(f"RF_Dir: 学習データ不足 ({len(train_df)} samples)")
        return {}

    X, y = prepare_rf_features_directory(train_df)
    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)

    importance = {
        name: float(val)
        for name, val in zip(FEATURE_NAMES_WITH_PATH, clf.feature_importances_)
    }
    logger.info(f"RF_Dir importance: {len(importance)} 特徴量, "
                f"学習サンプル={len(train_df)}, pos={int(y.sum())}")
    return importance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="保存済みモデルから特徴量重要度を抽出"
    )
    parser.add_argument(
        "--base-dir", type=Path, required=True,
        help="モデルのベースディレクトリ (e.g., outputs/variant_comparison_server/lstm_baseline)",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/combined_raw.csv"),
        help="レビュー依頼CSVファイルのパス",
    )
    parser.add_argument(
        "--raw-json", type=str, nargs="+",
        default=["data/raw_json/openstack__nova.json"],
        help="生JSONファイル（ディレクトリマッピング用）",
    )
    parser.add_argument(
        "--prediction-time", type=str, default="2023-01-01",
        help="予測時点 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end", type=str, default="2022-01-01",
        help="RF学習データの終了日 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--window-days", type=int, default=180,
        help="特徴量計算のウィンドウ幅（日数）",
    )
    parser.add_argument(
        "--max-samples", type=int, default=300,
        help="IRL勾配計算の最大サンプル数",
    )
    parser.add_argument(
        "--include-rf", action="store_true",
        help="RF_Dir の Gini importance も計算する",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="既存の importance ファイルを上書きする",
    )
    args = parser.parse_args()

    # データ読み込み
    logger.info(f"データ読み込み: {args.data}")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ディレクトリマッピング
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    df = attach_dirs_to_df(df, cdm)

    path_extractor = PathFeatureExtractor(df, window_days=args.window_days)
    prediction_time = datetime.fromisoformat(args.prediction_time)
    train_end = datetime.fromisoformat(args.train_end)

    # 各 train window について処理
    for i, win in enumerate(TRAIN_WINDOWS):
        model_dir = args.base_dir / f"train_{win}"
        model_path = model_dir / "irl_model.pt"

        if not model_path.exists():
            logger.warning(f"モデルが見つからない: {model_path}")
            continue

        logger.info(f"=== {win} ===")

        # 対角パターン (train==eval) の eval_dir に保存
        eval_dir = model_dir / f"eval_{win}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # IRL importance
        irl_out = eval_dir / "irl_feature_importance.json"
        irl_signed_out = eval_dir / "irl_feature_importance_signed.json"
        if irl_out.exists() and irl_signed_out.exists() and not args.overwrite:
            logger.info(f"スキップ（既存）: {irl_out}")
        else:
            head_index = i
            abs_importance, signed_importance = compute_irl_importance(
                model_path, df, path_extractor, prediction_time,
                window_days=args.window_days,
                max_samples=args.max_samples,
                head_index=head_index,
            )
            if abs_importance:
                with open(irl_out, "w") as f:
                    json.dump(abs_importance, f, indent=2, ensure_ascii=False)
                logger.info(f"保存: {irl_out}")
                with open(irl_signed_out, "w") as f:
                    json.dump(signed_importance, f, indent=2, ensure_ascii=False)
                logger.info(f"保存: {irl_signed_out}")

        # RF importance
        if args.include_rf:
            rf_out = eval_dir / "rf_metrics.json"
            if rf_out.exists() and not args.overwrite:
                logger.info(f"スキップ（既存）: {rf_out}")
            else:
                rf_importance = compute_rf_dir_importance(
                    df, path_extractor, prediction_time, train_end,
                    future_start_months=FUTURE_START[i],
                    delta_months=3,
                    window_days=args.window_days,
                )
                if rf_importance:
                    rf_data = {"feature_importance": rf_importance}
                    with open(rf_out, "w") as f:
                        json.dump(rf_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"保存: {rf_out}")

    logger.info("完了")


if __name__ == "__main__":
    main()
