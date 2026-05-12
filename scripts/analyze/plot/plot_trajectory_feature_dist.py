#!/usr/bin/env python3
"""軌跡キャッシュから実際のモデル入力特徴量の分布を可視化するスクリプト。

分布分析スクリプト (plot_feature_distributions.py) は月次スナップショットを
全レビュアー×全月で取るため、活動なしの月がゼロとして大量に含まれる。

本スクリプトは MCE-IRL が学習時に使うのと同じロジック
(mce_irl_predictor._precompute_trajectories) で特徴量を計算し、
**実際にモデルが見る分布**を可視化する。

- common_features 20次元 (STATE_FEATURES) + 5次元 (ACTION_FEATURES)
- path_features 3次元 (ディレクトリ親和度)
- 正例 / 負例の重ね描き

使い方:
    uv run python scripts/analyze/plot/plot_trajectory_feature_dist.py \
        --cache outputs/mce_irl_trajectory_cache/main32/mce_traj_0-3.pkl \
        --output-dir outputs/feature_dist_trajectory_main32
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from review_predictor.IRL.features.common_features import (
    STATE_FEATURES,
    ACTION_FEATURES,
    extract_common_features,
)
from review_predictor.IRL.features.path_features import PATH_FEATURE_NAMES

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
try:
    matplotlib.rcParams["font.family"] = "Hiragino Sans"
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────
# 特徴量の日本語名
# ────────────────────────────────────────────────────
FEATURE_JA: Dict[str, str] = {
    "window_tenure_days": "窓内tenure",
    "total_changes": "PR投稿数",
    "total_reviews": "レビュー受信数",
    "recent_activity_frequency": "直近活動頻度",
    "avg_activity_gap": "平均活動間隔",
    "activity_trend": "活動トレンド",
    "unique_collaborator_count": "ユニーク協力者数",
    "overall_acceptance_rate": "全期間承諾率",
    "recent_acceptance_rate": "直近承諾率",
    "recent_load_ratio_30d_all": "30日/全期間負荷比",
    "days_since_last_activity": "最終活動からの日数",
    "acceptance_trend": "承諾率トレンド",
    "reciprocity_score": "相互レビュー率",
    "recent_load_ratio_7d_30d": "7日/30日負荷比",
    "core_reviewer_ratio": "コアレビュアー度",
    "recent_rejection_streak": "連続拒否数",
    "acceptance_rate_last10": "直近10件承諾率",
    "active_months_ratio": "活動月割合",
    "response_time_trend": "応答速度トレンド",
    "complex_pr_bias": "複雑PRバイアス",
    "avg_action_intensity": "平均行動強度",
    "avg_change_lines": "平均変更行数",
    "avg_response_time": "平均応答速度",
    "avg_review_size": "平均レビューサイズ",
    "repeat_collaboration_rate": "リピート協力率",
    "path_review_count": "パスレビュー回数",
    "path_recency": "パスリセンシー",
    "path_acceptance_rate": "パス承諾率",
}


def extract_features_from_trajectory(
    trajectory: Dict[str, Any],
    state_dim_expected: int,
) -> Optional[Dict[str, Any]]:
    """1 軌跡 → 各ステップの特徴量ベクトルを抽出。

    Returns dict with:
      - state_vecs: np.ndarray [L, state_dim]
      - action_vecs: np.ndarray [L, action_dim]
      - step_labels: list[int]
      - future_acceptance: bool
      - sample_weight: float
    """
    developer = trajectory.get("developer", trajectory.get("developer_info", {}))
    step_labels = trajectory.get("step_labels", [])
    monthly_histories = trajectory.get("monthly_activity_histories", [])

    if not step_labels or not monthly_histories:
        return None

    email = developer.get(
        "email", developer.get("developer_id", developer.get("reviewer", ""))
    )
    step_context_dates = trajectory.get("step_context_dates", [])
    step_total_project_reviews = trajectory.get("step_total_project_reviews", [])
    path_features_per_step = trajectory.get("path_features_per_step", [])

    min_len = min(len(monthly_histories), len(step_labels))
    state_vecs: List[np.ndarray] = []
    action_vecs: List[np.ndarray] = []

    for i in range(min_len):
        month_history = monthly_histories[i]
        if not month_history:
            state_vecs.append(np.zeros(state_dim_expected, dtype=np.float32))
            action_vecs.append(np.zeros(len(ACTION_FEATURES), dtype=np.float32))
            continue

        if step_context_dates and i < len(step_context_dates):
            month_context_date = step_context_dates[i]
        else:
            month_context_date = month_history[-1]["timestamp"]

        total_proj = (
            step_total_project_reviews[i]
            if i < len(step_total_project_reviews)
            else 0
        )
        pf = (
            path_features_per_step[i]
            if i < len(path_features_per_step)
            else None
        )

        # Build DataFrame from monthly history (same as _precompute_trajectories)
        rows: List[Dict[str, Any]] = []
        for act in month_history:
            ts = act.get("timestamp")
            if ts is None:
                continue
            if act.get("action_type") == "authored":
                rows.append({
                    "email": act.get("reviewer_email", ""),
                    "timestamp": pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts,
                    "label": 0,
                    "owner_email": email,
                    "change_insertions": act.get("lines_added", act.get("change_insertions", 0)) or 0,
                    "change_deletions": act.get("lines_deleted", act.get("change_deletions", 0)) or 0,
                    "change_files_count": act.get("files_changed", act.get("change_files_count", 0)) or 0,
                    "first_response_time": None,
                })
            else:
                rows.append({
                    "email": email,
                    "timestamp": pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts,
                    "label": 1 if act.get("accepted", False) else 0,
                    "owner_email": act.get("owner_email", ""),
                    "change_insertions": act.get("lines_added", act.get("change_insertions", 0)) or 0,
                    "change_deletions": act.get("lines_deleted", act.get("change_deletions", 0)) or 0,
                    "change_files_count": act.get("files_changed", act.get("change_files_count", 0)) or 0,
                    "first_response_time": act.get("response_time", act.get("first_response_time")),
                })
        df = pd.DataFrame(rows)

        if len(df) == 0:
            state_vecs.append(np.zeros(state_dim_expected, dtype=np.float32))
            action_vecs.append(np.zeros(len(ACTION_FEATURES), dtype=np.float32))
            continue

        feature_start = df["timestamp"].min()
        feature_end = pd.Timestamp(month_context_date)
        try:
            features = extract_common_features(
                df, email, feature_start, feature_end,
                normalize=True, total_project_reviews=total_proj,
            )
        except Exception:
            state_vecs.append(np.zeros(state_dim_expected, dtype=np.float32))
            action_vecs.append(np.zeros(len(ACTION_FEATURES), dtype=np.float32))
            continue

        sv = [float(features.get(f, 0.0)) for f in STATE_FEATURES]
        if pf is not None:
            sv.extend(float(v) for v in pf)
        av = [float(features.get(f, 0.0)) for f in ACTION_FEATURES]
        state_vecs.append(np.array(sv, dtype=np.float32))
        action_vecs.append(np.array(av, dtype=np.float32))

    if not state_vecs:
        return None

    return {
        "state_vecs": np.stack(state_vecs),
        "action_vecs": np.stack(action_vecs),
        "step_labels": step_labels[:min_len],
        "future_acceptance": trajectory.get("future_acceptance", False),
        "sample_weight": trajectory.get("sample_weight", 1.0),
    }


def main():
    p = argparse.ArgumentParser(description="軌跡キャッシュから特徴量分布を可視化")
    p.add_argument("--cache", required=True, help="軌跡キャッシュ .pkl のパス")
    p.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    p.add_argument("--max-trajectories", type=int, default=None,
                   help="処理する軌跡数の上限 (デバッグ用)")
    p.add_argument("--n-jobs", type=int, default=-1, help="並列数")
    p.add_argument("--label-split", action="store_true",
                   help="正例/負例で色分けして重ね描きする (デフォルト: 全体分布のみ)")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("キャッシュ読み込み: %s", args.cache)
    with open(args.cache, "rb") as f:
        trajectories = pickle.load(f)
    logger.info("軌跡数: %d", len(trajectories))

    if args.max_trajectories:
        trajectories = trajectories[: args.max_trajectories]
        logger.info("上限適用後: %d 軌跡", len(trajectories))

    # Detect state_dim from first trajectory
    sample = trajectories[0]
    has_path = bool(sample.get("path_features_per_step"))
    state_dim = len(STATE_FEATURES) + (len(PATH_FEATURE_NAMES) if has_path else 0)
    logger.info("state_dim=%d (path_features=%s)", state_dim, has_path)

    # Feature names
    all_state_names = list(STATE_FEATURES) + (list(PATH_FEATURE_NAMES) if has_path else [])
    all_action_names = list(ACTION_FEATURES)
    all_names = all_state_names + all_action_names

    # Extract features in parallel
    from joblib import Parallel, delayed
    logger.info("特徴量抽出中 (n_jobs=%d)...", args.n_jobs)
    raw_results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(extract_features_from_trajectory)(t, state_dim)
        for t in trajectories
    )
    results = [r for r in raw_results if r is not None]
    logger.info("有効な軌跡: %d / %d", len(results), len(trajectories))

    # Collect all step-level feature vectors
    all_state_vecs = []
    all_action_vecs = []
    all_labels = []  # step labels (0/1)
    all_future = []  # trajectory-level future_acceptance (broadcast to steps)

    for r in results:
        n_steps = r["state_vecs"].shape[0]
        all_state_vecs.append(r["state_vecs"])
        all_action_vecs.append(r["action_vecs"])
        all_labels.extend(r["step_labels"])
        all_future.extend([r["future_acceptance"]] * n_steps)

    state_mat = np.concatenate(all_state_vecs, axis=0)  # [N, state_dim]
    action_mat = np.concatenate(all_action_vecs, axis=0)  # [N, action_dim]
    feat_mat = np.concatenate([state_mat, action_mat], axis=1)  # [N, all_dim]
    labels = np.array(all_labels)
    future = np.array(all_future)

    logger.info("全ステップ数: %d", feat_mat.shape[0])
    logger.info("  step_label=1: %d (%.1f%%)",
                labels.sum(), labels.mean() * 100)
    logger.info("  future_acceptance=True: %d (%.1f%%)",
                future.sum(), future.mean() * 100)

    # Save CSV
    df_feat = pd.DataFrame(feat_mat, columns=all_names)
    df_feat["__step_label"] = labels
    df_feat["__future_acceptance"] = future.astype(int)
    csv_path = out / "trajectory_feature_values.csv"
    df_feat.to_csv(csv_path, index=False)
    logger.info("CSV 保存: %s", csv_path)

    # Percentile table
    pct_rows = []
    for i, name in enumerate(all_names):
        v = feat_mat[:, i]
        v_nonzero = v[v != 0.0]
        pct_rows.append({
            "feature": name,
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)),
            "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(np.max(v)),
            "zero_rate": float((v == 0.0).mean()),
            "n_nonzero": int(len(v_nonzero)),
            "n": int(len(v)),
        })
    pct_df = pd.DataFrame(pct_rows).set_index("feature")
    pct_csv = out / "trajectory_feature_percentiles.csv"
    pct_df.to_csv(pct_csv)
    logger.info("percentile CSV 保存: %s", pct_csv)
    print(pct_df.to_string())

    # Plot
    for i, name in enumerate(all_names):
        ja = FEATURE_JA.get(name, name)
        v = feat_mat[:, i]

        fig, ax = plt.subplots(figsize=(7, 4.5))

        if args.label_split:
            # 正例 / 負例で重ね描き (step_label ベース)
            v_pos = v[labels == 1]
            v_neg = v[labels == 0]
            if len(v_pos) > 0:
                ax.hist(v_pos, bins=50, alpha=0.5, color="C2",
                        label=f"承諾 (n={len(v_pos)})", density=True)
            if len(v_neg) > 0:
                ax.hist(v_neg, bins=50, alpha=0.5, color="C3",
                        label=f"拒否/なし (n={len(v_neg)})", density=True)
            ax.legend(fontsize=9)
            ax.set_ylabel("密度 (Density)", fontsize=10)
        else:
            # 全体分布 (色分けなし)
            ax.hist(v, bins=50, alpha=0.7, color="C0")
            ax.set_ylabel("頻度 (Frequency)", fontsize=10)

        ax.set_title(f"{ja} ({name})\n全ステップ n={len(v)}, ゼロ率={((v==0).mean())*100:.1f}%", fontsize=11)
        ax.set_xlabel(ja, fontsize=10)
        ax.tick_params(labelsize=9)

        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=200)
        fig.savefig(out / f"{name}.pdf", format="pdf")
        plt.close(fig)

    logger.info("プロット完了: %s", out)


if __name__ == "__main__":
    main()

