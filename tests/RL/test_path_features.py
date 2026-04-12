"""Tests for IRL/features/path_features.py and StateBuilder integration."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from IRL.features.path_features import (
    PATH_FEATURE_DIM,
    PathFeatureExtractor,
    attach_dirs_to_df,
    extract_dirs,
    file_to_dir,
)
from RL.state.state_builder import StateBuilder


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

def _make_df():
    """alice が nova/compute を 5 件レビュー、bob は nova/api だけ."""
    rows = []
    base = datetime(2024, 1, 1)
    for i in range(5):
        rows.append(
            dict(
                change_id=f"openstack%2Fnova~{1000 + i}",
                project="openstack/nova",
                owner_email="owner@x.com",
                email="alice@x.com",
                timestamp=base + timedelta(days=i),
                label=1 if i < 4 else 0,  # 4/5 accepted
                first_response_time=1.0,
                response_latency_days=1.0,
                change_insertions=10,
                change_deletions=5,
                change_files_count=2,
                is_cross_project=False,
                extraction_date="2024-12-01",
            )
        )
    rows.append(
        dict(
            change_id="openstack%2Fnova~2000",
            project="openstack/nova",
            owner_email="owner@x.com",
            email="bob@x.com",
            timestamp=datetime(2024, 1, 6),
            label=1,
            first_response_time=1.0,
            response_latency_days=1.0,
            change_insertions=10,
            change_deletions=5,
            change_files_count=1,
            is_cross_project=False,
            extraction_date="2024-12-01",
        )
    )
    return pd.DataFrame(rows)


def _fake_change_dir_map():
    m = {}
    for i in range(5):
        m[f"openstack%2Fnova~{1000 + i}"] = frozenset({"nova/compute"})
    m["openstack%2Fnova~2000"] = frozenset({"nova/api"})
    return m


# ─────────────────────────────────────────────────────────────────────
# tests
# ─────────────────────────────────────────────────────────────────────

def test_file_to_dir():
    assert file_to_dir("nova/compute/manager.py") == "nova/compute"
    assert file_to_dir("nova/tests/functional/x.py") == "nova/tests"
    assert file_to_dir("README.rst") == "."
    assert file_to_dir("a/b/c/d.py", depth=3) == "a/b/c"


def test_extract_dirs():
    dirs = extract_dirs(["nova/compute/x.py", "nova/api/y.py", "nova/compute/z.py"])
    assert dirs == frozenset({"nova/compute", "nova/api"})


def test_attach_dirs_to_df():
    df = _make_df()
    df2 = attach_dirs_to_df(df, _fake_change_dir_map())
    assert "dirs" in df2.columns
    # alice rows must have nova/compute
    alice_dirs = df2[df2["email"] == "alice@x.com"]["dirs"].iloc[0]
    assert "nova/compute" in alice_dirs


def test_path_feature_extractor_hits():
    df = attach_dirs_to_df(_make_df(), _fake_change_dir_map())
    ex = PathFeatureExtractor(df, window_days=30)
    vec = ex.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/compute"}),
        current_time=datetime(2024, 1, 10),
    )
    assert vec.shape == (PATH_FEATURE_DIM,)
    assert vec[0] > 0.0   # review count > 0
    assert vec[1] > 0.0   # recency > 0
    assert 0.0 <= vec[2] <= 1.0  # acceptance rate
    assert vec[2] == pytest.approx(4.0 / 5.0)


def test_path_feature_extractor_no_match():
    df = attach_dirs_to_df(_make_df(), _fake_change_dir_map())
    ex = PathFeatureExtractor(df, window_days=30)
    # alice has never touched nova/api
    vec = ex.compute(
        developer_id="alice@x.com",
        task_dirs=frozenset({"nova/api"}),
        current_time=datetime(2024, 1, 10),
    )
    assert np.allclose(vec, 0.0)


def test_path_feature_extractor_empty_task_dirs():
    df = attach_dirs_to_df(_make_df(), _fake_change_dir_map())
    ex = PathFeatureExtractor(df, window_days=30)
    vec = ex.compute("alice@x.com", frozenset(), datetime(2024, 1, 10))
    assert np.allclose(vec, 0.0)
    vec2 = ex.compute("alice@x.com", None, datetime(2024, 1, 10))
    assert np.allclose(vec2, 0.0)


def test_state_builder_includes_path_features():
    df = attach_dirs_to_df(_make_df(), _fake_change_dir_map())
    ex = PathFeatureExtractor(df, window_days=30)

    sb_base = StateBuilder(window_days=30)
    sb_path = StateBuilder(window_days=30, path_extractor=ex)

    assert sb_path.obs_dim == sb_base.obs_dim + PATH_FEATURE_DIM
    assert sb_path.feature_names[-PATH_FEATURE_DIM:] == [
        "path_review_count",
        "path_recency",
        "path_acceptance_rate",
    ]

    vec = sb_path.build(
        df=df,
        developer_id="alice@x.com",
        current_time=datetime(2024, 1, 10),
        task_dirs=frozenset({"nova/compute"}),
    )
    assert vec.shape == (sb_path.obs_dim,)
    # tail = path features, must be non-zero for alice@nova/compute
    tail = vec[-PATH_FEATURE_DIM:]
    assert tail[0] > 0.0
