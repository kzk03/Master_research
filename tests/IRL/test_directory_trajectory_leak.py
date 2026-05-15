"""extract_directory_level_trajectories のデータリーク防止テスト.

global の trajectory leak テストに対して、path 特徴量も含めた dir-level モデル
固有の不変条件を検証する:

  ▸ monthly_activity_histories は < month_end のデータのみ
  ▸ path_features_per_step[i] は current_time = step_context_dates[i] での
    PathFeatureExtractor.compute() 出力と一致 (= < current_time でフィルタ)
  ▸ path_extractor が「全データ込み」で初期化されていても、
    「訓練期間内データだけ」で初期化したものと同じ path_features を返す
    (= 各ステップで current_time フィルタが効いている)
  ▸ step_total_project_reviews も < month_end の範囲のみカウント
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_TRAIN = ROOT / "scripts" / "train"
if str(SCRIPTS_TRAIN) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_TRAIN))

from review_predictor.IRL.features.path_features import (  # noqa: E402
    PathFeatureExtractor,
    attach_dirs_to_df,
)
from train_model import extract_directory_level_trajectories  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# fixture
# ─────────────────────────────────────────────────────────────────────

TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2024-05-01")


def _row(reviewer, ts, label, change_id, owner="owner@x.com",
         project="openstack/nova"):
    return {
        "reviewer_email": reviewer,
        "email": reviewer,           # PathFeatureExtractor 用
        "request_time": pd.Timestamp(ts),
        "timestamp": pd.Timestamp(ts),  # PathFeatureExtractor 用
        "label": int(label),
        "change_id": change_id,
        "owner_email": owner,
        "project": project,
        "first_response_time": None,
        "change_insertions": 10,
        "change_deletions": 5,
        "change_files_count": 2,
        "is_cross_project": False,
    }


def _build_df_and_cdm():
    """alice: 訓練期間内 4 件 (nova/compute) + 訓練期間後 2 件 + 未来 1 件.

    change_id → dirs マップも返す.
    """
    rows = [
        _row("alice@x.com", "2024-01-15", 1, "c1"),
        _row("alice@x.com", "2024-02-15", 1, "c2"),
        _row("alice@x.com", "2024-03-15", 0, "c3"),
        _row("alice@x.com", "2024-04-15", 1, "c4"),
        # 訓練期間後 (label 期間内) — path_features に絶対漏れてはいけない
        _row("alice@x.com", "2024-06-01", 1, "c5"),
        _row("alice@x.com", "2024-06-15", 0, "c6"),
        # ありえない未来
        _row("alice@x.com", "2030-01-01", 1, "c99"),
    ]
    df = pd.DataFrame(rows)

    # 全 change_id を nova/compute に紐付け
    cdm = {f"c{i}": frozenset({"nova/compute"}) for i in [1, 2, 3, 4, 5, 6, 99]}
    df = attach_dirs_to_df(df, cdm, column="dirs")
    return df


@pytest.fixture(scope="module")
def trajectories_and_df():
    df = _build_df_and_cdm()
    path_extractor = PathFeatureExtractor(df, window_days=180)
    trajs = extract_directory_level_trajectories(
        df,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        path_extractor=path_extractor,
        future_window_start_months=0,
        future_window_end_months=3,
        n_jobs=1,    # 並列を避けてテスト中の loky overhead を回避
    )
    return trajs, df


def _alice_nova_compute(trajs):
    matched = [
        t for t in trajs
        if t.get("reviewer") == "alice@x.com" and t.get("directory") == "nova/compute"
    ]
    return matched[0] if matched else None


# ─────────────────────────────────────────────────────────────────────
# tests
# ─────────────────────────────────────────────────────────────────────


def test_alice_nova_compute_trajectory_exists(trajectories_and_df):
    trajs, _ = trajectories_and_df
    traj = _alice_nova_compute(trajs)
    assert traj is not None, "alice × nova/compute の軌跡が抽出されていない"
    assert traj["seq_len"] == len(traj["step_labels"])
    assert traj["seq_len"] >= 1


def test_monthly_history_respects_cutoff(trajectories_and_df):
    """月次履歴の timestamp < step_context_dates[i]."""
    trajs, _ = trajectories_and_df
    traj = _alice_nova_compute(trajs)
    contexts = traj["step_context_dates"]
    monthly = traj["monthly_activity_histories"]
    assert len(monthly) == len(contexts)
    for i, (history, ctx) in enumerate(zip(monthly, contexts)):
        cutoff = pd.Timestamp(ctx)
        for act in history:
            ts = pd.Timestamp(act["timestamp"])
            assert ts < cutoff, (
                f"step {i} cutoff={cutoff} に未来のレビュー混入: {ts}"
            )


def test_path_features_match_clean_extractor(trajectories_and_df):
    """全データ込み path_extractor の出力が、訓練期間内 df だけで作った
    extractor の出力と各ステップで一致 → current_time フィルタが効いている.
    """
    trajs, df = trajectories_and_df
    traj = _alice_nova_compute(trajs)

    # 訓練期間内データだけで作った「clean」extractor
    clean_df = df[df["timestamp"] < TRAIN_END].copy()
    clean_extractor = PathFeatureExtractor(clean_df, window_days=180)

    for i, ctx in enumerate(traj["step_context_dates"]):
        polluted_vec = traj["path_features_per_step"][i]
        clean_vec = clean_extractor.compute(
            developer_id="alice@x.com",
            task_dirs=frozenset({"nova/compute"}),
            current_time=pd.Timestamp(ctx).to_pydatetime(),
        )
        assert np.allclose(polluted_vec, clean_vec, atol=1e-6), (
            f"step {i} (ctx={ctx}) で path features がリークしている: "
            f"polluted={polluted_vec}, clean={clean_vec}"
        )


def test_step_total_project_reviews_matches_pre_cutoff_count(trajectories_and_df):
    """累積カウントも < month_end のみ."""
    trajs, _ = trajectories_and_df
    traj = _alice_nova_compute(trajs)
    # 訓練期間に入るのは alice の 4 件 (2024-01-15, 02-15, 03-15, 04-15)
    # 月ステップ: 02-01, 03-01, 04-01 (train_end=05-01 を含む step は drop)
    # cutoff=02-01: 1 件 / 03-01: 2 / 04-01: 3
    expected = [1, 2, 3]
    assert traj["step_total_project_reviews"] == expected


def test_no_far_future_review_in_any_step(trajectories_and_df):
    """2030-01-01 (遠い未来) は monthly_activity_histories のどこにも出てこない."""
    trajs, _ = trajectories_and_df
    traj = _alice_nova_compute(trajs)
    far_future = pd.Timestamp("2030-01-01")
    for history in traj["monthly_activity_histories"]:
        for act in history:
            assert pd.Timestamp(act["timestamp"]) != far_future


def test_label_period_reviews_not_in_features(trajectories_and_df):
    """ラベル期間 (2024-06) のレビューは特徴量側に絶対入らない."""
    trajs, _ = trajectories_and_df
    traj = _alice_nova_compute(trajs)
    label_dates = {pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-15")}
    for history in traj["monthly_activity_histories"]:
        for act in history:
            assert pd.Timestamp(act["timestamp"]) not in label_dates


def test_future_acceptance_uses_label_period(trajectories_and_df):
    """ラベル期間 (2024-05-01 〜 2024-08-01) のうち alice の nova/compute
    レビュー 2 件 (06-01 承諾, 06-15 拒否) → 1 件以上承諾あり → 正例."""
    trajs, _ = trajectories_and_df
    traj = _alice_nova_compute(trajs)
    assert traj["future_acceptance"] is True
    assert traj["sample_weight"] == pytest.approx(1.0)
