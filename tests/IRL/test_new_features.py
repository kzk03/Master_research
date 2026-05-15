"""
2026-05-14 追加の 5 特徴量（state 3 + path 2）の動作確認テスト。

state side:
  - n_projects
  - cross_project_review_share
  - same_domain_share

path side:
  - path_lcp_similarity
  - path_owner_overlap

注: WRC (weighted_review_count_wrc) も当初検討したが、smoke test で任意の
half-life で既存 count 系特徴量 (total_reviews / recent_30d_count) と
r ≥ 0.89 の高相関を確認したため、LSTM の表現力と冗長との判断で除外。
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from review_predictor.IRL.features.common_features import (
    FEATURE_NAMES,
    STATE_FEATURES,
    extract_common_features,
    normalize_features,
)
from review_predictor.IRL.features.path_features import (
    PATH_FEATURE_DIM,
    PATH_FEATURE_NAMES,
    PathFeatureExtractor,
    _lcp_score,
    attach_dirs_to_df,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# state-side fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_state_df() -> pd.DataFrame:
    """
    alice@redhat.com が
      - project=nova で 5件 (owner=ham@redhat.com, jane@gmail.com)
      - project=neutron で 3件 (owner=ham@redhat.com)
    の合計 8 件のレビューを受けたデータ。
    """
    base = datetime(2024, 6, 1)
    rows = []
    for i in range(5):
        rows.append(dict(
            change_id=f"openstack%2Fnova~{i}",
            project="openstack/nova",
            owner_email=("ham@redhat.com" if i < 3 else "jane@gmail.com"),
            email="alice@redhat.com",
            timestamp=base + timedelta(days=i),
            label=1,
        ))
    for i in range(3):
        rows.append(dict(
            change_id=f"openstack%2Fneutron~{i}",
            project="openstack/neutron",
            owner_email="ham@redhat.com",
            email="alice@redhat.com",
            timestamp=base + timedelta(days=10 + i),
            label=0,
        ))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# n_projects / cross_project_review_share
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_n_projects_counts_unique_projects():
    df = _build_state_df()
    feats = extract_common_features(
        df,
        email="alice@redhat.com",
        feature_start=datetime(2024, 5, 1),
        feature_end=datetime(2024, 7, 1),
        normalize=False,
    )
    assert feats["n_projects"] == 2.0


def test_cross_project_review_share_uses_home_project():
    df = _build_state_df()
    feats = extract_common_features(
        df,
        email="alice@redhat.com",
        feature_start=datetime(2024, 5, 1),
        feature_end=datetime(2024, 7, 1),
        normalize=False,
    )
    # home=nova (5件) なので neutron 3件が cross = 3/8 = 0.375
    assert math.isclose(feats["cross_project_review_share"], 3 / 8, abs_tol=1e-6)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# same_domain_share
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_same_domain_share():
    """alice@redhat.com の owner:
       nova (5件):    ham@redhat × 3 + jane@gmail × 2
       neutron (3件): ham@redhat × 3
    合計 8 件のうち ham@redhat = 6 件 → same_domain_share = 6/8 = 0.75"""
    df = _build_state_df()
    feats = extract_common_features(
        df,
        email="alice@redhat.com",
        feature_start=datetime(2024, 5, 1),
        feature_end=datetime(2024, 7, 1),
        normalize=False,
    )
    assert math.isclose(feats["same_domain_share"], 6 / 8, abs_tol=1e-6)


def test_same_domain_share_default_when_no_history():
    df = _build_state_df()
    feats = extract_common_features(
        df,
        email="nonexistent@redhat.com",
        feature_start=datetime(2024, 5, 1),
        feature_end=datetime(2024, 7, 1),
        normalize=False,
    )
    # 履歴なし = 中立値 0.5
    assert feats["same_domain_share"] == 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 整合性: STATE_FEATURES と feature dict のキー集合一致
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_extract_returns_all_feature_names():
    df = _build_state_df()
    feats = extract_common_features(
        df, "alice@redhat.com",
        datetime(2024, 5, 1), datetime(2024, 7, 1),
        normalize=False,
    )
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert len(STATE_FEATURES) == 21  # 18 + 3 (新規)


def test_normalize_keeps_new_features_in_range():
    df = _build_state_df()
    feats = extract_common_features(
        df, "alice@redhat.com",
        datetime(2024, 5, 1), datetime(2024, 7, 1),
        normalize=True,
    )
    for k in ("n_projects", "cross_project_review_share", "same_domain_share"):
        v = feats[k]
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"
        assert math.isfinite(v), f"{k}={v} is not finite"


def test_default_features_complete():
    df = pd.DataFrame(columns=["change_id","project","owner_email","email",
                                "timestamp","label"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # 完全に空の df でも default が返る
    feats = extract_common_features(
        df, "anyone@x", datetime(2024, 5, 1), datetime(2024, 7, 1),
        normalize=True,
    )
    assert set(feats.keys()) == set(FEATURE_NAMES)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# path side: _lcp_score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.parametrize("a,b,expected", [
    ("nova/compute", "nova/compute",     1.0),
    ("nova/compute", "nova/network",     0.5),
    ("nova/compute", "neutron/agent",    0.0),
    ("nova/compute/api", "nova/compute", 2 / 3),
    ("", "x", 0.0),
    ("x", "", 0.0),
])
def test_lcp_score(a, b, expected):
    assert math.isclose(_lcp_score(a, b), expected, abs_tol=1e-6)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# path side: PathFeatureExtractor.compute returns 5 dims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_path_df():
    base = datetime(2024, 6, 1)
    rows = [
        # alice が nova/compute を 3 件レビュー (label=1,1,0)
        dict(change_id=f"nova~{i}", project="openstack/nova",
             owner_email="ham@x.com", email="alice@x.com",
             timestamp=base + timedelta(days=i),
             label=1 if i < 2 else 0)
        for i in range(3)
    ]
    rows += [
        # alice が nova/network も 2 件 → reviewer_dirs に出てくる
        dict(change_id=f"net~{i}", project="openstack/nova",
             owner_email="jane@y.com", email="alice@x.com",
             timestamp=base + timedelta(days=5 + i), label=1)
        for i in range(2)
    ]
    # ham と jane 以外の owner が nova/compute に PR を出していて、bob がそれをレビュー
    # → bob と alice の owner overlap が ham 経由で発生する
    rows.append(dict(
        change_id="nova~99", project="openstack/nova",
        owner_email="ham@x.com", email="bob@x.com",
        timestamp=base + timedelta(days=2), label=1,
    ))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # change_id -> dirs マップ (手動構築)
    cdm = {
        **{f"nova~{i}": frozenset({"nova/compute"}) for i in range(3)},
        **{f"net~{i}":  frozenset({"nova/network"}) for i in range(2)},
        "nova~99": frozenset({"nova/compute"}),
    }
    df = attach_dirs_to_df(df, cdm)
    return df


def test_compute_returns_five_dims():
    df = _build_path_df()
    ext = PathFeatureExtractor(df, window_days=30)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),
        current_time=datetime(2024, 6, 30),
    )
    assert vec.shape == (5,)
    assert vec.dtype == np.float32
    assert PATH_FEATURE_DIM == 5
    assert np.all(np.isfinite(vec))


def test_lcp_similarity_recognizes_related_dir():
    """task_dir=nova/db (alice は触っていない) でも nova/compute, nova/network があるので
    LCP は 1/2 = 0.5 程度."""
    df = _build_path_df()
    ext = PathFeatureExtractor(df, window_days=30)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/db"}),
        current_time=datetime(2024, 6, 30),
    )
    # vec[3] = path_lcp_similarity
    assert vec[3] == pytest.approx(0.5, abs=1e-3)
    # path_review_count は 0
    assert vec[0] == 0.0


def test_owner_overlap_jaccard():
    """alice の owner = {ham, jane}, task=nova/compute の owner = {ham} (window内)
    → Jaccard = 1/2 = 0.5"""
    df = _build_path_df()
    ext = PathFeatureExtractor(df, window_days=30)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),
        current_time=datetime(2024, 6, 30),
    )
    # vec[4] = path_owner_overlap (rev_owners={ham,jane}, td_owners={ham}, union=2, inter=1)
    assert vec[4] == pytest.approx(0.5, abs=1e-3)


def test_empty_task_dirs_returns_zero():
    df = _build_path_df()
    ext = PathFeatureExtractor(df, window_days=30)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=None,
        current_time=datetime(2024, 6, 30),
    )
    assert vec.shape == (5,)
    assert np.all(vec == 0.0)


def test_path_feature_names_count():
    assert len(PATH_FEATURE_NAMES) == 5
    assert PATH_FEATURE_NAMES[-2:] == ["path_lcp_similarity", "path_owner_overlap"]
