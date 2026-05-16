"""
2026-05-14〜15 追加の特徴量の動作確認テスト。

state side (Phase 1, 3 features):
  - n_projects
  - cross_project_review_share
  - same_domain_share

path side (Phase 1, 1 feature):
  - path_owner_overlap

path side (Phase 2, 2 features):
  - path_hub_score
  - path_neighbor_coverage

注:
- WRC (weighted_review_count_wrc) も当初検討したが、smoke test で任意の
  half-life で既存 count 系特徴量 (total_reviews / recent_30d_count) と
  r ≥ 0.89 の高相関を確認したため、LSTM の表現力と冗長との判断で除外。
- path_lcp_similarity (REVFINDER 式) は v2_phase1 で RF importance=0.000
  と完全 dead だったため 2026-05-15 に削除。
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
# path side: PathFeatureExtractor.compute returns 6 dims
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


def test_compute_returns_six_dims():
    df = _build_path_df()
    ext = PathFeatureExtractor(df, window_days=30)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),
        current_time=datetime(2024, 6, 30),
    )
    assert vec.shape == (6,)
    assert vec.dtype == np.float32
    assert PATH_FEATURE_DIM == 6
    assert np.all(np.isfinite(vec))
    # hub_score / coverage が未指定なので 0.0
    assert vec[4] == 0.0  # path_hub_score
    assert vec[5] == 0.0  # path_neighbor_coverage


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
    # vec[3] = path_owner_overlap (rev_owners={ham,jane}, td_owners={ham}, union=2, inter=1)
    assert vec[3] == pytest.approx(0.5, abs=1e-3)


def test_empty_task_dirs_returns_zero():
    df = _build_path_df()
    ext = PathFeatureExtractor(df, window_days=30)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=None,
        current_time=datetime(2024, 6, 30),
    )
    assert vec.shape == (6,)
    assert np.all(vec == 0.0)


def test_path_feature_names_count():
    assert len(PATH_FEATURE_NAMES) == 6
    assert PATH_FEATURE_NAMES[3] == "path_owner_overlap"
    assert PATH_FEATURE_NAMES[4:] == ["path_hub_score", "path_neighbor_coverage"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: hub_score & neighbor_coverage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _write_hub_csv(tmp_path, rows):
    """rows = [(project, directory, hub_score, change_count), ...]"""
    p = tmp_path / "hub_scores.csv"
    df = pd.DataFrame(rows, columns=["project", "directory", "hub_score", "change_count"])
    df.to_csv(p, index=False)
    return p


def _write_neighbors_csv(tmp_path, rows):
    """rows = [(project, directory, neighbor, weight), ...]
    両方向 (d1->d2, d2->d1) を呼び出し側で渡すこと。
    """
    p = tmp_path / "cochange_neighbors.csv"
    df = pd.DataFrame(rows, columns=["project", "directory", "neighbor", "weight"])
    df.to_csv(p, index=False)
    return p


def test_hub_score_loaded_and_max_normalized(tmp_path):
    """hub_scores CSV を読み込み、task_dirs の max を 64 で正規化して返す。"""
    df = _build_path_df()
    hub = _write_hub_csv(tmp_path, [
        ("openstack/nova", "nova/compute", 32, 100),
        ("openstack/nova", "nova/network",  16,  50),
        ("openstack/other", "other/x",       8,  10),
    ])
    ext = PathFeatureExtractor(df, window_days=30, hub_scores_path=hub)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute", "nova/network"}),
        current_time=datetime(2024, 6, 30),
    )
    # max(32, 16) / 64 = 0.5
    assert vec[4] == pytest.approx(0.5, abs=1e-4)


def test_hub_score_unknown_dir_returns_zero(tmp_path):
    df = _build_path_df()
    hub = _write_hub_csv(tmp_path, [("openstack/nova", "nova/compute", 32, 100)])
    ext = PathFeatureExtractor(df, window_days=30, hub_scores_path=hub)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"unknown/dir"}),
        current_time=datetime(2024, 6, 30),
    )
    assert vec[4] == 0.0


def test_neighbor_coverage_intersection(tmp_path):
    """alice の touched dirs = {nova/compute, nova/network}.
    target=nova/compute の co-change 近傍 = {nova/db, nova/network}.
    intersect = {nova/network}, denom = 2 (nova/db, nova/network).
    coverage = 1/2 = 0.5."""
    df = _build_path_df()
    neigh = _write_neighbors_csv(tmp_path, [
        ("openstack/nova", "nova/compute", "nova/db",      5),
        ("openstack/nova", "nova/db",      "nova/compute", 5),
        ("openstack/nova", "nova/compute", "nova/network", 3),
        ("openstack/nova", "nova/network", "nova/compute", 3),
    ])
    ext = PathFeatureExtractor(df, window_days=30, cochange_neighbors_path=neigh)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),
        current_time=datetime(2024, 6, 30),
    )
    # alice の reviewer_dirs = {nova/compute, nova/network}
    # target=nova/compute の近傍 = {nova/db, nova/network}
    # target_dirs を近傍から除外 (自己排除) → {nova/db, nova/network}
    # 交差 = {nova/network} → coverage = 1/2
    assert vec[5] == pytest.approx(0.5, abs=1e-4)


def test_neighbor_coverage_no_neighbors_returns_zero(tmp_path):
    df = _build_path_df()
    # target dir has no neighbors in the table
    neigh = _write_neighbors_csv(tmp_path, [
        ("openstack/nova", "nova/other", "nova/x", 5),
        ("openstack/nova", "nova/x",     "nova/other", 5),
    ])
    ext = PathFeatureExtractor(df, window_days=30, cochange_neighbors_path=neigh)
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),  # 近傍なし
        current_time=datetime(2024, 6, 30),
    )
    assert vec[5] == 0.0


def test_path_features_backward_compat_no_graph():
    """hub_scores / cochange_neighbors を指定しなければ新 2 特徴量は 0.0、
    既存呼び出しサイトは壊れない。"""
    df = _build_path_df()
    ext = PathFeatureExtractor(df, window_days=30)  # 旧シグネチャ
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),
        current_time=datetime(2024, 6, 30),
    )
    assert vec.shape == (6,)
    assert vec[4] == 0.0  # hub_score 未ロード
    assert vec[5] == 0.0  # neighbor_coverage 未ロード


def test_hub_and_coverage_combined(tmp_path):
    """両方読み込み + 両方の値が独立に取れることを確認。"""
    df = _build_path_df()
    hub = _write_hub_csv(tmp_path, [
        ("openstack/nova", "nova/compute", 16, 100),
        ("openstack/nova", "nova/db",       8,  50),
        ("openstack/nova", "nova/network",  4,  20),
    ])
    neigh = _write_neighbors_csv(tmp_path, [
        ("openstack/nova", "nova/compute", "nova/db",      5),
        ("openstack/nova", "nova/db",      "nova/compute", 5),
        ("openstack/nova", "nova/compute", "nova/network", 3),
        ("openstack/nova", "nova/network", "nova/compute", 3),
    ])
    ext = PathFeatureExtractor(
        df, window_days=30,
        hub_scores_path=hub, cochange_neighbors_path=neigh,
    )
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),
        current_time=datetime(2024, 6, 30),
    )
    # hub: 16 / 64 = 0.25
    assert vec[4] == pytest.approx(0.25, abs=1e-4)
    # coverage: 1/2 (alice の nova/network ⊂ 近傍 {nova/db, nova/network})
    assert vec[5] == pytest.approx(0.5, abs=1e-4)


def test_missing_csv_path_is_safe(tmp_path):
    """指定した CSV が存在しない場合、警告を出して 0.0 を返す (例外を投げない)."""
    df = _build_path_df()
    fake = tmp_path / "nonexistent.csv"
    ext = PathFeatureExtractor(
        df, window_days=30,
        hub_scores_path=fake, cochange_neighbors_path=fake,
    )
    vec = ext.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),
        current_time=datetime(2024, 6, 30),
    )
    assert vec[4] == 0.0
    assert vec[5] == 0.0
