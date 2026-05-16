"""raw_statistics.py / directory_contributors.py の挙動テスト.

raw_statistics は「正規化なし生統計量」がモットーなので、
  ▸ 空 df → default (NaN 中心) が返る
  ▸ 単一レコード → 標準偏差や gap は NaN
  ▸ 値そのものが clip / normalize されない (= 大きい値はそのまま返る)
を確認.

directory_contributors は集合・カウント集約の挙動を確認.
"""
from __future__ import annotations

from datetime import datetime

import math
import numpy as np
import pandas as pd
import pytest

from review_predictor.IRL.features.path_features import attach_dirs_to_df
from review_predictor.IRL.features.directory_contributors import (
    count_actual_contributors,
    get_all_directories,
    get_directory_developers,
)
from review_predictor.IRL.features.raw_statistics import (
    RAW_STAT_NAMES,
    extract_raw_statistics,
    get_default_raw_statistics,
)


# ─────────────────────────────────────────────────────────────────────
# raw_statistics: fixture
# ─────────────────────────────────────────────────────────────────────

EMAIL = "alice@example.com"
WIN_START = datetime(2024, 1, 1)
WIN_END = datetime(2024, 7, 1)


def _row(ts, label=1, owner="owner@example.com",
         insertions=10, deletions=5, files=2,
         first_response=None, email=EMAIL):
    return {
        "email": email,
        "owner_email": owner,
        "timestamp": pd.Timestamp(ts),
        "label": int(label),
        "change_insertions": insertions,
        "change_deletions": deletions,
        "change_files_count": files,
        "first_response_time": first_response,
    }


def _extract(rows):
    df = pd.DataFrame(rows)
    return extract_raw_statistics(df, EMAIL, WIN_START, WIN_END)


# ─────────────────────────────────────────────────────────────────────
# raw_statistics: empty / default
# ─────────────────────────────────────────────────────────────────────


def test_empty_df_returns_default_raw_statistics():
    df = pd.DataFrame(columns=[
        "email", "owner_email", "timestamp", "label",
        "change_insertions", "change_deletions", "change_files_count",
        "first_response_time",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    stats = extract_raw_statistics(df, EMAIL, WIN_START, WIN_END)
    default = get_default_raw_statistics()
    assert set(stats.keys()) == set(RAW_STAT_NAMES)
    for k in default:
        if isinstance(default[k], float) and math.isnan(default[k]):
            assert math.isnan(stats[k]), f"{k} should be NaN"
        else:
            assert stats[k] == pytest.approx(default[k]), f"{k}: {stats[k]} vs {default[k]}"


def test_default_keys_match_raw_stat_names():
    default = get_default_raw_statistics()
    assert set(default.keys()) == set(RAW_STAT_NAMES)


# ─────────────────────────────────────────────────────────────────────
# raw_statistics: 値が clip/normalize されないこと
# ─────────────────────────────────────────────────────────────────────


def test_raw_values_not_clipped_or_normalized():
    """raw 統計量は「生値」がモットーなので大きな数値もそのまま返る."""
    rows = [
        _row("2024-03-01", insertions=10_000, deletions=10_000, files=500),
        _row("2024-04-01", insertions=10_000, deletions=10_000, files=500),
    ]
    stats = _extract(rows)
    # change_lines = 20000 を 2 件、mean=20000
    assert stats["raw_mean_change_lines"] == pytest.approx(20_000.0)
    assert stats["raw_max_change_lines"] == pytest.approx(20_000.0)
    # files も clip されない
    assert stats["raw_max_files"] == pytest.approx(500.0)


def test_total_reviews_and_changes_count_distinct_roles():
    """reviewer 行と owner 行を別カウント."""
    rows = [
        # alice が reviewer
        _row("2024-02-01"),
        _row("2024-03-01"),
        _row("2024-04-01"),
        # alice が owner (他人が reviewer)
        _row("2024-05-01", email="other@x.com", owner=EMAIL),
        _row("2024-05-15", email="other@x.com", owner=EMAIL),
    ]
    stats = _extract(rows)
    assert stats["total_reviews"] == 3
    assert stats["total_changes"] == 2
    assert stats["total_activity"] == 5


# ─────────────────────────────────────────────────────────────────────
# raw_statistics: 単一レコード / gap = NaN
# ─────────────────────────────────────────────────────────────────────


def test_single_review_gives_nan_gap_stats():
    rows = [_row("2024-03-15")]
    stats = _extract(rows)
    assert math.isnan(stats["mean_activity_gap_days"])
    assert math.isnan(stats["median_activity_gap_days"])
    assert math.isnan(stats["std_activity_gap_days"])
    # accepted_count は計算可能 (1)
    assert stats["accepted_count"] == 1
    assert stats["rejected_count"] == 0
    assert stats["acceptance_rate_raw"] == pytest.approx(1.0)


def test_two_reviews_gives_finite_gap():
    rows = [
        _row("2024-03-01"),
        _row("2024-03-15"),  # 14 日後
    ]
    stats = _extract(rows)
    assert stats["mean_activity_gap_days"] == pytest.approx(14.0)
    assert stats["median_activity_gap_days"] == pytest.approx(14.0)
    # std は 1 サンプル (gap 1 個) なので NaN
    assert math.isnan(stats["std_activity_gap_days"])


# ─────────────────────────────────────────────────────────────────────
# raw_statistics: acceptance / active_months
# ─────────────────────────────────────────────────────────────────────


def test_acceptance_counts_match_labels():
    rows = [
        _row("2024-02-01", label=1),
        _row("2024-03-01", label=1),
        _row("2024-04-01", label=0),
        _row("2024-05-01", label=0),
        _row("2024-06-01", label=0),
    ]
    stats = _extract(rows)
    assert stats["accepted_count"] == 2
    assert stats["rejected_count"] == 3
    assert stats["acceptance_rate_raw"] == pytest.approx(2 / 5)


def test_active_months_counts_unique_yearmonth():
    """同じ月 (3 月) に複数件あっても 1 ヶ月としてカウント."""
    rows = [
        _row("2024-03-01"),
        _row("2024-03-15"),
        _row("2024-04-01"),
        _row("2024-05-15"),
    ]
    stats = _extract(rows)
    # 3 月, 4 月, 5 月 = 3 active months
    assert stats["active_months"] == 3


def test_days_since_last_activity_uses_max_timestamp():
    last_ts = pd.Timestamp("2024-06-15")
    rows = [_row("2024-01-15"), _row(last_ts)]
    stats = _extract(rows)
    # WIN_END = 2024-07-01
    expected = (WIN_END - last_ts).days
    assert stats["days_since_last_activity"] == expected


# ─────────────────────────────────────────────────────────────────────
# raw_statistics: collaboration
# ─────────────────────────────────────────────────────────────────────


def test_unique_and_repeat_collaborators():
    """owner X: 3回, Y: 1回, Z: 1回 → unique=3, repeated=1, rate=1/3."""
    rows = [
        _row("2024-02-01", owner="X@x.com"),
        _row("2024-03-01", owner="X@x.com"),
        _row("2024-04-01", owner="X@x.com"),
        _row("2024-05-01", owner="Y@x.com"),
        _row("2024-06-01", owner="Z@x.com"),
    ]
    stats = _extract(rows)
    assert stats["unique_collaborators"] == 3
    assert stats["repeated_collaborators"] == 1
    assert stats["repeat_collaboration_rate_raw"] == pytest.approx(1 / 3)


# ─────────────────────────────────────────────────────────────────────
# raw_statistics: response time
# ─────────────────────────────────────────────────────────────────────


def test_response_time_uses_first_response_minus_timestamp():
    rows = [
        _row("2024-03-01", first_response=pd.Timestamp("2024-03-02")),  # 1 日
        _row("2024-04-01", first_response=pd.Timestamp("2024-04-04")),  # 3 日
    ]
    stats = _extract(rows)
    assert stats["raw_mean_response_days"] == pytest.approx(2.0)
    assert stats["raw_max_response_days"] == pytest.approx(3.0)
    assert stats["raw_median_response_days"] == pytest.approx(2.0)


def test_response_time_nan_when_no_response_data():
    """first_response_time が全部 NaN なら nan."""
    rows = [_row("2024-03-01", first_response=None)]
    stats = _extract(rows)
    assert math.isnan(stats["raw_mean_response_days"])


# ─────────────────────────────────────────────────────────────────────
# directory_contributors
# ─────────────────────────────────────────────────────────────────────


def _build_dir_df():
    rows = []
    base = pd.Timestamp("2024-01-01")
    # alice × nova/compute
    rows.append({"email": "alice@x.com", "timestamp": base, "change_id": "c1"})
    rows.append({"email": "alice@x.com", "timestamp": base + pd.Timedelta(days=10),
                 "change_id": "c2"})
    # bob × nova/api
    rows.append({"email": "bob@x.com", "timestamp": base + pd.Timedelta(days=5),
                 "change_id": "c3"})
    # carol × nova/compute
    rows.append({"email": "carol@x.com", "timestamp": base + pd.Timedelta(days=15),
                 "change_id": "c4"})
    # 期間外
    rows.append({"email": "dave@x.com", "timestamp": pd.Timestamp("2025-01-01"),
                 "change_id": "c5"})
    df = pd.DataFrame(rows)
    cdm = {
        "c1": frozenset({"nova/compute"}),
        "c2": frozenset({"nova/compute"}),
        "c3": frozenset({"nova/api"}),
        "c4": frozenset({"nova/compute"}),
        "c5": frozenset({"nova/network"}),
    }
    df = attach_dirs_to_df(df, cdm, column="dirs")
    return df


def test_get_directory_developers_groups_by_dir():
    df = _build_dir_df()
    result = get_directory_developers(
        df, datetime(2024, 1, 1), datetime(2024, 6, 1),
    )
    assert result["nova/compute"] == {"alice@x.com", "carol@x.com"}
    assert result["nova/api"] == {"bob@x.com"}
    # dave は期間外なので nova/network エントリは存在しない
    assert "nova/network" not in result


def test_count_actual_contributors_returns_unique_counts():
    df = _build_dir_df()
    counts = count_actual_contributors(
        df, datetime(2024, 1, 1), datetime(2024, 6, 1),
    )
    # alice は 2 回 nova/compute をタッチ → unique count 1
    assert counts["nova/compute"] == 2  # alice + carol
    assert counts["nova/api"] == 1


def test_get_all_directories_excludes_dot_directory():
    """'.' は除外される (root-level files)."""
    rows = [
        {"email": "x@x.com", "timestamp": pd.Timestamp("2024-03-01"), "change_id": "c1"},
        {"email": "x@x.com", "timestamp": pd.Timestamp("2024-03-02"), "change_id": "c2"},
    ]
    cdm = {
        "c1": frozenset({"nova/compute"}),
        "c2": frozenset({".", "nova/api"}),
    }
    df = attach_dirs_to_df(pd.DataFrame(rows), cdm, column="dirs")
    dirs = get_all_directories(df)
    assert dirs == {"nova/compute", "nova/api"}
    assert "." not in dirs


def test_get_all_directories_without_range_uses_whole_df():
    df = _build_dir_df()
    dirs = get_all_directories(df)
    assert dirs == {"nova/compute", "nova/api", "nova/network"}


def test_get_directory_developers_skips_empty_dirs():
    """dirs が空の行はスキップされる."""
    rows = [
        {"email": "a@x.com", "timestamp": pd.Timestamp("2024-03-01"), "change_id": "c1"},
        {"email": "b@x.com", "timestamp": pd.Timestamp("2024-03-15"), "change_id": "c2"},
    ]
    cdm = {
        "c1": frozenset({"nova/compute"}),
        # c2 は cdm に無い → attach で空 frozenset
    }
    df = attach_dirs_to_df(pd.DataFrame(rows), cdm, column="dirs")
    result = get_directory_developers(
        df, datetime(2024, 1, 1), datetime(2024, 6, 1),
    )
    # b@x.com は dirs が空なので結果に含まれない
    assert result == {"nova/compute": {"a@x.com"}}
