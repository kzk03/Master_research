"""extract_common_features 残り 18 特徴量の境界値テスト.

既存の test_new_features.py は 2026-05-14 追加の 3 特徴量 (n_projects,
cross_project_review_share, same_domain_share) と path features のみカバー.
本ファイルでは残りの state 18 + action 5 を境界値で押さえる:

  ▸ 空 df → _get_default_features() が返る
  ▸ dev_data 空 (reviewer 経験ゼロ、owner 経験のみ) → 早期 return ブロック
  ▸ 単一レコード → avg_activity_gap=60.0, activity_trend=0.0 など
  ▸ 全件同時刻 → activity_trend=0.0
  ▸ 全 label=1 / 全 label=0 → overall_acceptance_rate=1.0/0.0
  ▸ 直近30日が空 → recent_acceptance_rate=0.5 (中立値)
  ▸ reviewer_owner_ratio の 0/0 → 0.5
  ▸ acceptance_rate_last10 が末尾10件で計算される
  ▸ response_time_trend が正→「最近速くなった」を表す
  ▸ normalize_features の clip と log1p
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from review_predictor.IRL.features.common_features import (
    ACTION_FEATURES,
    FEATURE_NAMES,
    STATE_FEATURES,
    _get_default_features,
    extract_common_features,
    normalize_features,
)


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

EMAIL = "alice@example.com"
WIN_START = datetime(2024, 1, 1)
WIN_END = datetime(2024, 7, 1)   # 6ヶ月窓


def _row(ts, label=1, owner="owner@example.com",
         insertions=10, deletions=5, files=2,
         project="openstack/nova", first_response=None,
         email=EMAIL):
    return {
        "email": email,
        "owner_email": owner,
        "timestamp": pd.Timestamp(ts),
        "label": int(label),
        "project": project,
        "change_insertions": insertions,
        "change_deletions": deletions,
        "change_files_count": files,
        "first_response_time": first_response,
    }


def _df(rows):
    return pd.DataFrame(rows)


def _extract(df, normalize=False, total_project_reviews=0):
    return extract_common_features(
        df, EMAIL, WIN_START, WIN_END,
        normalize=normalize, total_project_reviews=total_project_reviews,
    )


# ─────────────────────────────────────────────────────────────────────
# 全空 / dev_data ゼロ
# ─────────────────────────────────────────────────────────────────────


def test_empty_df_returns_full_default_features():
    df = pd.DataFrame(
        columns=["email", "owner_email", "timestamp", "label",
                 "project", "change_insertions", "change_deletions",
                 "change_files_count", "first_response_time"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = _extract(df, normalize=False)

    default = _get_default_features()
    assert set(feats.keys()) == set(FEATURE_NAMES)
    for k, v in default.items():
        assert feats[k] == pytest.approx(v), f"{k} mismatch: {feats[k]} vs {v}"


def test_owner_only_history_uses_early_return_block():
    """alice が PR 作成者 (owner) のみで reviewer 経験ゼロのケース.

    dev_data 空ブロック (common_features L234) に入り、reviewer_owner_ratio=0,
    avg_activity_gap=120.0 (cap), days_since_last_activity=730.0 (cap) になる.
    """
    df = _df([
        # alice は owner として PR を作っている (= owner_data に入る)
        _row("2024-02-15", owner=EMAIL, email="someone@x.com"),
        _row("2024-03-15", owner=EMAIL, email="someone@x.com"),
    ])
    feats = _extract(df, normalize=False)

    assert feats["total_reviews"] == 0
    assert feats["total_changes"] == 2
    assert feats["reviewer_owner_ratio"] == pytest.approx(0.0)
    assert feats["avg_activity_gap"] == pytest.approx(120.0)
    assert feats["days_since_last_activity"] == pytest.approx(730.0)
    assert feats["overall_acceptance_rate"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────
# 単一レコード / 全件同時刻
# ─────────────────────────────────────────────────────────────────────


def test_single_review_gives_neutral_activity_trend_and_gap_cap():
    df = _df([_row("2024-03-15", label=1)])
    feats = _extract(df, normalize=False)

    assert feats["total_reviews"] == 1
    assert feats["activity_trend"] == pytest.approx(0.0)   # < 2 件で計算不能
    assert feats["avg_activity_gap"] == pytest.approx(60.0)  # 単一 → cap 値
    assert feats["overall_acceptance_rate"] == pytest.approx(1.0)


def test_all_reviews_same_timestamp_zero_trend():
    same_ts = "2024-03-15 10:00:00"
    df = _df([_row(same_ts, label=1) for _ in range(5)])
    feats = _extract(df, normalize=False)

    # t_min == t_max → activity_trend=0 (L663)
    assert feats["activity_trend"] == pytest.approx(0.0)
    # gap は同時刻なので 0
    assert feats["avg_activity_gap"] == pytest.approx(0.0)


def test_activity_trend_increasing_when_back_loaded():
    """期間中点より後ろに偏ったら trend > 0."""
    df = _df([
        _row("2024-02-01", label=1),
        _row("2024-06-01", label=1),
        _row("2024-06-15", label=1),
        _row("2024-06-25", label=1),
    ])
    feats = _extract(df, normalize=False)
    assert feats["activity_trend"] > 0.0


# ─────────────────────────────────────────────────────────────────────
# acceptance rate (overall / recent / last10)
# ─────────────────────────────────────────────────────────────────────


def test_all_accepted_gives_full_acceptance_rate():
    df = _df([
        _row("2024-03-15", label=1),
        _row("2024-04-15", label=1),
        _row("2024-05-15", label=1),
    ])
    feats = _extract(df, normalize=False)
    assert feats["overall_acceptance_rate"] == pytest.approx(1.0)


def test_all_rejected_gives_zero_acceptance_rate():
    df = _df([
        _row("2024-03-15", label=0),
        _row("2024-04-15", label=0),
        _row("2024-05-15", label=0),
    ])
    feats = _extract(df, normalize=False)
    assert feats["overall_acceptance_rate"] == pytest.approx(0.0)


def test_recent_acceptance_rate_defaults_to_neutral_when_no_recent_data():
    """直近30日 (WIN_END - 30days = 2024-06-01) 以前のレビューだけだと中立 0.5."""
    df = _df([
        _row("2024-02-01", label=1),  # 古い
        _row("2024-03-01", label=1),  # 古い
    ])
    feats = _extract(df, normalize=False)
    assert feats["recent_acceptance_rate"] == pytest.approx(0.5)


def test_acceptance_rate_last10_uses_tail_only():
    """20 件のうち最新 10 件で計算される.

    最初 10 件は全 label=0、最新 10 件は全 label=1 → last10 平均 = 1.0.
    """
    rows = []
    for i in range(10):
        rows.append(_row(f"2024-01-{i+1:02d}", label=0))
    for i in range(10):
        rows.append(_row(f"2024-06-{i+10:02d}", label=1))
    df = _df(rows)
    feats = _extract(df, normalize=False)
    assert feats["acceptance_rate_last10"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────
# reviewer_owner_ratio
# ─────────────────────────────────────────────────────────────────────


def test_reviewer_owner_ratio_pure_reviewer_is_one():
    """PR を作らず review しかしない人は ratio = 1.0."""
    df = _df([_row("2024-03-15", label=1)])
    feats = _extract(df, normalize=False)
    assert feats["reviewer_owner_ratio"] == pytest.approx(1.0)


def test_reviewer_owner_ratio_balanced():
    """review 3 + own 3 → 0.5."""
    rows = [_row(f"2024-0{i+1}-15", label=1) for i in range(3)]
    # 自分が owner の PR を 3 件 (= owner_data に入る)
    rows += [_row(f"2024-04-{i+1:02d}", owner=EMAIL, email="other@x.com")
             for i in range(3)]
    df = _df(rows)
    feats = _extract(df, normalize=False)
    assert feats["reviewer_owner_ratio"] == pytest.approx(3 / (3 + 3))


# ─────────────────────────────────────────────────────────────────────
# core_reviewer_ratio / days_since_last_activity
# ─────────────────────────────────────────────────────────────────────


def test_core_reviewer_ratio_with_explicit_project_total():
    """total_project_reviews を引数で渡せばそれが分母になる."""
    df = _df([_row("2024-03-15", label=1) for _ in range(5)])
    feats = _extract(df, normalize=False, total_project_reviews=100)
    assert feats["core_reviewer_ratio"] == pytest.approx(5 / 100)


def test_core_reviewer_ratio_uses_in_window_count_when_zero():
    """total_project_reviews=0 のときは window 内の全件数を分母にする."""
    df = _df([_row("2024-03-15", label=1) for _ in range(5)])
    feats = _extract(df, normalize=False, total_project_reviews=0)
    # alice の 5 件しか df に無いので分母 = 5、ratio = 1.0
    assert feats["core_reviewer_ratio"] == pytest.approx(1.0)


def test_days_since_last_activity_uses_max_timestamp():
    """feature_end から最新レビューまでの差日数."""
    last_review = pd.Timestamp("2024-06-15")
    df = _df([_row("2024-01-15"), _row(last_review)])
    feats = _extract(df, normalize=False)
    expected = (WIN_END - last_review).days
    assert feats["days_since_last_activity"] == expected


# ─────────────────────────────────────────────────────────────────────
# unique_collaborator_count / repeat_collaboration_rate
# ─────────────────────────────────────────────────────────────────────


def test_unique_collaborator_count_caps_at_ten():
    """ユニーク owner 10 人で 1.0 にキャップ."""
    rows = [_row("2024-03-15", owner=f"u{i}@x.com") for i in range(15)]
    df = _df(rows)
    feats = _extract(df, normalize=False)
    assert feats["unique_collaborator_count"] == pytest.approx(1.0)


def test_repeat_collaboration_rate_counts_repeat_owners():
    """2 回以上依頼してきた owner の割合 (ユニーク owner ベース).

    owner X: 3回, Y: 1回, Z: 1回 → unique=3, repeat>1 = 1 (X のみ) → 1/3.
    """
    df = _df([
        _row("2024-03-01", owner="X@x.com"),
        _row("2024-03-15", owner="X@x.com"),
        _row("2024-04-01", owner="X@x.com"),
        _row("2024-04-15", owner="Y@x.com"),
        _row("2024-05-01", owner="Z@x.com"),
    ])
    feats = _extract(df, normalize=False)
    assert feats["repeat_collaboration_rate"] == pytest.approx(1 / 3)


# ─────────────────────────────────────────────────────────────────────
# response_time / response_time_trend
# ─────────────────────────────────────────────────────────────────────


def test_avg_response_time_zero_days_gives_one():
    """応答 0 日 (即応答) → 1.0."""
    rows = [
        _row("2024-03-15", first_response=pd.Timestamp("2024-03-15")),
        _row("2024-04-15", first_response=pd.Timestamp("2024-04-15")),
    ]
    df = _df(rows)
    feats = _extract(df, normalize=False)
    assert feats["avg_response_time"] == pytest.approx(1.0)


def test_avg_response_time_three_days_gives_half():
    """応答 3 日 → 1 / (1 + 3/3) = 0.5."""
    rows = [
        _row("2024-03-15", first_response=pd.Timestamp("2024-03-18")),
        _row("2024-04-15", first_response=pd.Timestamp("2024-04-18")),
    ]
    df = _df(rows)
    feats = _extract(df, normalize=False)
    assert feats["avg_response_time"] == pytest.approx(0.5)


def test_response_time_trend_positive_when_recent_is_faster():
    """直近の応答が速い → trend > 0.

    pd.Timestamp を直接渡して datetime64 dtype を揃える (混在 format を避ける).
    """
    rows = [
        # 古い: 応答 10 日
        _row("2024-02-01", first_response=pd.Timestamp("2024-02-11")),
        _row("2024-02-15", first_response=pd.Timestamp("2024-02-25")),
        # 直近 (WIN_END=2024-07-01 から 30 日以内): 応答 0.5 日
        _row("2024-06-15", first_response=pd.Timestamp("2024-06-15 12:00:00")),
        _row("2024-06-20", first_response=pd.Timestamp("2024-06-20 12:00:00")),
    ]
    df = _df(rows)
    feats = _extract(df, normalize=False)
    assert feats["response_time_trend"] > 0.0


# ─────────────────────────────────────────────────────────────────────
# avg_change_lines / avg_review_size / avg_action_intensity
# ─────────────────────────────────────────────────────────────────────


def test_avg_change_lines_sums_ins_del():
    df = _df([
        _row("2024-03-15", insertions=50, deletions=50),
        _row("2024-04-15", insertions=100, deletions=100),
    ])
    feats = _extract(df, normalize=False)
    # mean((50+50), (100+100)) = mean(100, 200) = 150
    assert feats["avg_change_lines"] == pytest.approx(150.0)


def test_avg_review_size_caps_at_one():
    """files=20 (cap=10) → 2.0 だが min(_, 1.0) で 1.0."""
    df = _df([_row("2024-03-15", files=20)])
    feats = _extract(df, normalize=False)
    assert feats["avg_review_size"] == pytest.approx(1.0)


def test_avg_action_intensity_is_geometric_mean():
    """幾何平均: sqrt(files_score * lines_score)."""
    # files=10 → files_score=10/20=0.5, lines=250 → lines_score=250/500=0.5
    df = _df([_row("2024-03-15", files=10, insertions=125, deletions=125)])
    feats = _extract(df, normalize=False)
    assert feats["avg_action_intensity"] == pytest.approx(math.sqrt(0.5 * 0.5))


# ─────────────────────────────────────────────────────────────────────
# normalize_features
# ─────────────────────────────────────────────────────────────────────


def test_normalize_clips_at_one():
    raw = {"core_reviewer_ratio": 5.0}  # cap=0.2 → 5/0.2 = 25 だが 1.0 でクリップ
    result = normalize_features(raw)
    assert result["core_reviewer_ratio"] == pytest.approx(1.0)


def test_normalize_log1p_for_count_features():
    """total_reviews は log1p / log1p(cap) で正規化される."""
    raw = {"total_reviews": 50.0}
    result = normalize_features(raw)
    expected = math.log1p(50.0) / math.log1p(500.0)
    assert result["total_reviews"] == pytest.approx(expected)


def test_normalize_preserves_trend_features():
    """activity_trend / response_time_trend は -1〜1 のため正規化対象外."""
    raw = {"activity_trend": -0.7, "response_time_trend": 0.3}
    result = normalize_features(raw)
    assert result["activity_trend"] == pytest.approx(-0.7)
    assert result["response_time_trend"] == pytest.approx(0.3)


def test_normalized_features_in_unit_range_except_trends():
    """normalize=True で返ったすべての値は [-1,1] (trend) or [0,1] (others)."""
    df = _df([
        _row("2024-02-01", label=1, owner="A@x.com", insertions=50, files=3),
        _row("2024-03-01", label=0, owner="B@x.com", insertions=100, files=5),
        _row("2024-04-01", label=1, owner="A@x.com", insertions=200, files=10),
        _row("2024-05-01", label=1, owner="C@x.com", insertions=80, files=4),
        _row("2024-06-15", label=1, owner="A@x.com", insertions=120, files=6),
    ])
    feats = _extract(df, normalize=True)

    trend_keys = {"activity_trend", "response_time_trend"}
    for k, v in feats.items():
        assert math.isfinite(v), f"{k}={v} is not finite"
        if k in trend_keys:
            assert -1.0 <= v <= 1.0, f"{k}={v} outside [-1,1]"
        else:
            assert 0.0 <= v <= 1.0, f"{k}={v} outside [0,1]"


def test_all_feature_names_present():
    """state 21 + action 5 = 26 個すべて返る."""
    df = _df([_row("2024-03-15", label=1)])
    feats = _extract(df, normalize=False)
    assert set(feats.keys()) == set(STATE_FEATURES) | set(ACTION_FEATURES)
    assert len(STATE_FEATURES) == 21
    assert len(ACTION_FEATURES) == 5
