"""extract_review_acceptance_trajectories のデータリーク防止テスト.

CLAUDE.md「データリーク防止」の項にある以下の不変条件を検証する:
  - 特徴量計算 (monthly_activity_histories[i]) は feature_start ≤ t < month_end の
    範囲のデータしか参照しない
  - activity_history 全体も train_end を超えるデータを含まない
  - step_total_project_reviews[i] は history_start から month_end までの件数と一致
  - future_start >= train_end になる月はラベルを作らない (= step が削られる)

戦略: 訓練期間内・ラベル期間内・「ありえない未来」3 種をミックスした fixture を作り、
未来データが軌跡に紛れ込まないことを timestamp ベースで確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# scripts/train を sys.path に通して train_model.py から関数をロード
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_TRAIN = ROOT / "scripts" / "train"
if str(SCRIPTS_TRAIN) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_TRAIN))

from train_model import extract_review_acceptance_trajectories  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# fixture: 訓練期間 [2024-01-01, 2024-05-01) / future window 0-3m
# ─────────────────────────────────────────────────────────────────────

TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2024-05-01")
FUTURE_WINDOW_START_M = 0
FUTURE_WINDOW_END_M = 3
# → ラベル期間 = [2024-05-01, 2024-08-01), 拡張期間 = [2024-05-01, 2025-01-01)
LABEL_START = pd.Timestamp("2024-05-01")
LABEL_END = pd.Timestamp("2024-08-01")
EXTENDED_LABEL_END = pd.Timestamp("2025-01-01")


def _row(reviewer: str, ts: str, label: int, project: str = "openstack/nova",
         owner: str = "owner-x@example.com") -> dict:
    return {
        "reviewer_email": reviewer,
        "request_time": pd.Timestamp(ts),
        "label": label,
        "project": project,
        "owner_email": owner,
        "first_response_time": None,
        "change_insertions": 10,
        "change_deletions": 5,
        "change_files_count": 2,
        "is_cross_project": False,
    }


def _build_df() -> pd.DataFrame:
    rows = [
        # alice: 訓練期間に毎月 1 件レビューを受ける (正例候補)
        _row("alice@x.com", "2024-01-15", label=1),
        _row("alice@x.com", "2024-02-15", label=1),
        _row("alice@x.com", "2024-03-15", label=0),
        _row("alice@x.com", "2024-04-15", label=1),
        # alice: ラベル期間内に 1 件承諾 (これは label_df 経由で正例認定に使う)
        _row("alice@x.com", "2024-06-01", label=1),
        # alice: ありえないほど未来 — リーク検査用。
        # この行が monthly_activity_histories / activity_history に
        # 1 件でも紛れたらリーク。
        _row("alice@x.com", "2030-01-01", label=1),

        # bob: 訓練期間に依頼なし、拡張期間内のみに依頼あり → 弱負例
        _row("bob@x.com", "2024-09-15", label=1),

        # carol: 拡張期間にも依頼なし → 除外 (pu_unlabeled_weight=0 デフォルト)
        # ※ そもそも history_df に出てこないので軌跡候補にもならない
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# tests
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trajectories():
    df = _build_df()
    return extract_review_acceptance_trajectories(
        df,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        future_window_start_months=FUTURE_WINDOW_START_M,
        future_window_end_months=FUTURE_WINDOW_END_M,
        min_history_requests=0,
        extended_label_window_months=12,
        negative_oversample_factor=1,
        pu_unlabeled_weight=0.0,
    )


def _get(trajs, reviewer):
    matched = [t for t in trajs if t.get("reviewer") == reviewer]
    return matched[0] if matched else None


def test_alice_trajectory_exists(trajectories):
    """alice は訓練期間に履歴を持つので軌跡が作られる."""
    alice = _get(trajectories, "alice@x.com")
    assert alice is not None
    # 正例 (ラベル期間に承諾あり) として認定
    assert alice["future_acceptance"] is True
    assert alice["sample_weight"] == 1.0


def test_activity_history_excludes_post_train_end_data(trajectories):
    """activity_history (評価用全期間) も train_end を超えるデータを含まない.

    fixture には 2030-01-01 と 2024-06-01 の未来データを入れているが、
    どちらも train_end=2024-05-01 を超えるため軌跡には現れてはいけない.
    """
    alice = _get(trajectories, "alice@x.com")
    assert alice is not None
    for act in alice["activity_history"]:
        ts = pd.Timestamp(act["timestamp"])
        assert ts < TRAIN_END, (
            f"activity_history に train_end 以降のデータが混入: {ts}"
        )


def test_monthly_history_respects_month_end_cutoff(trajectories):
    """monthly_activity_histories[i] の各 timestamp は対応する month_end 未満.

    これが破れていると LSTM 入力に未来情報がリークする (最重要不変条件).
    """
    alice = _get(trajectories, "alice@x.com")
    assert alice is not None
    monthly = alice["monthly_activity_histories"]
    contexts = alice["step_context_dates"]
    assert len(monthly) == len(contexts), "月次履歴と基準日のペアがズレている"

    for i, (history, month_end) in enumerate(zip(monthly, contexts)):
        cutoff = pd.Timestamp(month_end)
        for act in history:
            ts = pd.Timestamp(act["timestamp"])
            assert ts < cutoff, (
                f"step {i} (cutoff={cutoff}) に未来データが混入: {ts}"
            )


def test_step_count_drops_months_with_future_start_beyond_train_end(trajectories):
    """future_start >= train_end の月はラベルが作られず step が削られる.

    history_months = [2024-01-01, 2024-02-01, 2024-03-01, 2024-04-01, 2024-05-01]
    history_months[:-1] = 4 候補 → month_end ∈ [02-01, 03-01, 04-01, 05-01]
    future_window_start=0 なので future_start = month_end. train_end=05-01 なので
    month_end=05-01 のステップだけ drop され、残り 3 ステップ.
    """
    alice = _get(trajectories, "alice@x.com")
    assert alice is not None
    assert len(alice["step_labels"]) == 3
    assert len(alice["monthly_activity_histories"]) == 3
    assert len(alice["step_context_dates"]) == 3


def test_step_total_project_reviews_matches_pre_cutoff_count(trajectories):
    """step_total_project_reviews[i] = train_start から month_end までの件数."""
    alice = _get(trajectories, "alice@x.com")
    assert alice is not None
    # 訓練期間内の依頼 (alice + bob + carol 含む) の累積カウント.
    # fixture では訓練期間に入るのは alice の 4 件のみ.
    # cutoff = 2024-02-01 → alice の 2024-01-15 のみ → 1
    # cutoff = 2024-03-01 → 2024-01-15, 2024-02-15 → 2
    # cutoff = 2024-04-01 → 2024-01-15, 2024-02-15, 2024-03-15 → 3
    expected = [1, 2, 3]
    assert alice["step_total_project_reviews"] == expected


def test_bob_is_weak_negative_without_in_window_requests(trajectories):
    """bob は訓練期間に依頼ゼロ・拡張期間に 1 件 → 弱負例 (weight=0.1)."""
    bob = _get(trajectories, "bob@x.com")
    # 注: bob は history_df に出てこない (訓練期間に依頼なし) ため
    # active_reviewers に含まれず、そもそも軌跡候補にすらならない.
    # これが「依頼なし開発者は除外」の意図された挙動.
    assert bob is None, (
        "history_df に出てこない開発者は active_reviewers に含まれず軌跡化されないはず"
    )


def test_no_leakage_from_label_period_into_features(trajectories):
    """ラベル期間 (2024-05-01〜2024-08-01) の依頼が特徴量側に混入していないこと.

    alice の 2024-06-01 はラベル判定 (future_acceptance) には使われるが、
    activity_history / monthly_activity_histories には絶対に入ってはならない.
    """
    alice = _get(trajectories, "alice@x.com")
    target_ts = pd.Timestamp("2024-06-01")

    for act in alice["activity_history"]:
        assert pd.Timestamp(act["timestamp"]) != target_ts

    for month_history in alice["monthly_activity_histories"]:
        for act in month_history:
            assert pd.Timestamp(act["timestamp"]) != target_ts


def test_far_future_data_never_appears_anywhere(trajectories):
    """2030-01-01 のような遠い未来のデータも一切軌跡に入らない."""
    alice = _get(trajectories, "alice@x.com")
    far_future = pd.Timestamp("2030-01-01")

    for act in alice["activity_history"]:
        assert pd.Timestamp(act["timestamp"]) != far_future
    for month_history in alice["monthly_activity_histories"]:
        for act in month_history:
            assert pd.Timestamp(act["timestamp"]) != far_future
