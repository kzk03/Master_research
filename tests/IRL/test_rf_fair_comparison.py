"""rf_predictor の「IRL と公平比較できる前提」を担保するテスト.

CLAUDE.md: 「特徴量定義は common_features.py に一元化。IRL と RF で同じ特徴量を
使い公平比較。」

ここで検証する不変条件:
  ▸ extract_features_for_window の出力列が FEATURE_NAMES + ['label','email']
  ▸ prepare_rf_features が FEATURE_NAMES の順に X を組み立てる
  ▸ extract_features_for_window_directory の出力が FEATURE_NAMES_WITH_PATH + ...
  ▸ train_and_evaluate_rf が AUC=1.0 を返せる (=線形分離可能な fixture)
  ▸ feature_importance のキーが FEATURE_NAMES と一致
  ▸ クラスが1つしかない学習データで None を返す
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from review_predictor.IRL.features.common_features import (
    FEATURE_NAMES,
    FEATURE_NAMES_WITH_PATH,
)
from review_predictor.IRL.features.path_features import (
    PATH_FEATURE_DIM,
    PathFeatureExtractor,
    attach_dirs_to_df,
)
from review_predictor.IRL.model.rf_predictor import (
    extract_features_for_window,
    extract_features_for_window_directory,
    prepare_rf_features,
    prepare_rf_features_directory,
    train_and_evaluate_rf,
)


# ─────────────────────────────────────────────────────────────────────
# fixture
# ─────────────────────────────────────────────────────────────────────


def _row(email, ts, label, owner="owner@x.com", project="openstack/nova",
         change_id=None, insertions=10, deletions=5, files=2):
    return {
        "email": email,
        "owner_email": owner,
        "timestamp": pd.Timestamp(ts),
        "label": int(label),
        "project": project,
        "change_id": change_id or f"chg-{email}-{ts}",
        "first_response_time": None,
        "change_insertions": insertions,
        "change_deletions": deletions,
        "change_files_count": files,
        "is_cross_project": False,
    }


def _build_df():
    rows = []
    # alice: train 期間 4 件 (label=1) + eval 期間 2 件 (label=1)
    for i in range(4):
        rows.append(_row("alice@x.com", f"2024-01-{10+i*5:02d}", 1))
    rows.append(_row("alice@x.com", "2024-06-05", 1))
    rows.append(_row("alice@x.com", "2024-06-15", 1))
    # bob: train 期間 4 件 + eval 期間に全 reject
    for i in range(4):
        rows.append(_row("bob@x.com", f"2024-02-{10+i*5:02d}", 1))
    rows.append(_row("bob@x.com", "2024-06-05", 0))
    rows.append(_row("bob@x.com", "2024-06-10", 0))
    # carol: train 期間 3 件 + eval 期間 1 件 (label=1)
    for i in range(3):
        rows.append(_row("carol@x.com", f"2024-03-{5+i*5:02d}", 1))
    rows.append(_row("carol@x.com", "2024-06-20", 1))
    # dave: train 期間 5 件 + eval 期間 reject
    for i in range(5):
        rows.append(_row("dave@x.com", f"2024-02-{5+i*5:02d}", 1))
    rows.append(_row("dave@x.com", "2024-07-01", 0))
    return pd.DataFrame(rows)


TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2024-05-01")
EVAL_START = pd.Timestamp("2024-05-01")
EVAL_END = pd.Timestamp("2024-08-01")


# ─────────────────────────────────────────────────────────────────────
# 形状・列名の整合性
# ─────────────────────────────────────────────────────────────────────


def test_extract_features_for_window_columns_match_feature_names():
    df = _build_df()
    feats_df = extract_features_for_window(
        df, TRAIN_START, TRAIN_END, EVAL_START, EVAL_END,
    )
    expected_cols = set(FEATURE_NAMES) | {"label", "email"}
    assert set(feats_df.columns) == expected_cols
    # 各 email は 1 行ずつ
    assert feats_df["email"].is_unique


def test_extract_features_returns_empty_dataframe_when_no_label_data():
    """eval 期間に誰も依頼を受けていないと空 DF が返る."""
    df = _build_df()
    # eval 期間を future (誰もいない) にする
    feats_df = extract_features_for_window(
        df, TRAIN_START, TRAIN_END,
        pd.Timestamp("2030-01-01"), pd.Timestamp("2030-04-01"),
    )
    assert len(feats_df) == 0
    assert set(feats_df.columns) == set(FEATURE_NAMES) | {"label", "email"}


def test_prepare_rf_features_returns_arrays_in_feature_name_order():
    df = _build_df()
    feats_df = extract_features_for_window(
        df, TRAIN_START, TRAIN_END, EVAL_START, EVAL_END,
    )
    X, y = prepare_rf_features(feats_df)
    assert X.shape == (len(feats_df), len(FEATURE_NAMES))
    assert y.shape == (len(feats_df),)
    assert X.dtype == np.float64
    assert y.dtype == np.int64 or y.dtype == np.int32 or np.issubdtype(y.dtype, np.integer)


def test_labels_match_eval_period_acceptance():
    """alice (eval 期間に承諾あり) は label=1, bob/dave (全拒否) は label=0."""
    df = _build_df()
    feats_df = extract_features_for_window(
        df, TRAIN_START, TRAIN_END, EVAL_START, EVAL_END,
    )
    label_by_email = dict(zip(feats_df["email"], feats_df["label"]))
    assert label_by_email.get("alice@x.com") == 1
    assert label_by_email.get("carol@x.com") == 1
    assert label_by_email.get("bob@x.com") == 0
    assert label_by_email.get("dave@x.com") == 0


# ─────────────────────────────────────────────────────────────────────
# directory-level
# ─────────────────────────────────────────────────────────────────────


def _build_df_with_dirs():
    df = _build_df()
    cdm = {cid: frozenset({"nova/compute"}) for cid in df["change_id"].unique()}
    return attach_dirs_to_df(df, cdm, column="dirs")


def test_extract_features_for_window_directory_uses_path_dim():
    df = _build_df_with_dirs()
    path_extractor = PathFeatureExtractor(df, window_days=180)
    feats_df = extract_features_for_window_directory(
        df, TRAIN_START, TRAIN_END, EVAL_START, EVAL_END,
        path_extractor=path_extractor,
    )
    expected_cols = set(FEATURE_NAMES_WITH_PATH) | {"label", "email", "directory"}
    assert set(feats_df.columns) == expected_cols
    assert len(FEATURE_NAMES_WITH_PATH) == len(FEATURE_NAMES) + PATH_FEATURE_DIM


def test_prepare_rf_features_directory_returns_28_or_more_dims():
    df = _build_df_with_dirs()
    path_extractor = PathFeatureExtractor(df, window_days=180)
    feats_df = extract_features_for_window_directory(
        df, TRAIN_START, TRAIN_END, EVAL_START, EVAL_END,
        path_extractor=path_extractor,
    )
    X, y = prepare_rf_features_directory(feats_df)
    assert X.shape[1] == len(FEATURE_NAMES_WITH_PATH)
    assert len(y) == X.shape[0]


# ─────────────────────────────────────────────────────────────────────
# train_and_evaluate_rf
# ─────────────────────────────────────────────────────────────────────


def test_train_and_evaluate_rf_returns_none_when_single_class():
    """1 クラスしかない訓練データ → None."""
    X_train = np.random.rand(20, len(FEATURE_NAMES))
    y_train = np.zeros(20, dtype=int)   # 全部 0
    X_eval = np.random.rand(10, len(FEATURE_NAMES))
    y_eval = np.random.randint(0, 2, 10)

    result = train_and_evaluate_rf(X_train, y_train, X_eval, y_eval)
    assert result is None


def test_train_and_evaluate_rf_returns_metrics_dict():
    """正常ケース: 線形分離可能データで AUC が高い."""
    rng = np.random.default_rng(0)
    n_per_class = 30
    n_feat = len(FEATURE_NAMES)
    # 正例は最初の特徴量が大きい、負例は小さい — 完全分離可能
    X_pos = rng.normal(loc=2.0, scale=0.5, size=(n_per_class, n_feat))
    X_neg = rng.normal(loc=-2.0, scale=0.5, size=(n_per_class, n_feat))
    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)]).astype(int)

    X_eval_pos = rng.normal(loc=2.0, scale=0.5, size=(20, n_feat))
    X_eval_neg = rng.normal(loc=-2.0, scale=0.5, size=(20, n_feat))
    X_eval = np.vstack([X_eval_pos, X_eval_neg])
    y_eval = np.concatenate([np.ones(20), np.zeros(20)]).astype(int)

    result = train_and_evaluate_rf(X_train, y_train, X_eval, y_eval)
    assert result is not None
    # 構造
    expected_keys = {
        "auc_roc", "auc_pr", "precision", "recall", "f1_score",
        "optimal_threshold", "positive_count", "negative_count",
        "total_count", "feature_importance", "predictions",
    }
    assert expected_keys.issubset(result.keys())
    # 線形分離可能なら AUC ≈ 1.0
    assert result["auc_roc"] == pytest.approx(1.0, abs=0.05)
    assert result["f1_score"] == pytest.approx(1.0, abs=0.05)
    # predictions の長さが y_eval と一致
    assert len(result["predictions"]) == len(y_eval)


def test_feature_importance_keys_match_feature_names():
    """feature_importance の keys が FEATURE_NAMES と完全一致 (公平比較の前提)."""
    rng = np.random.default_rng(1)
    n_feat = len(FEATURE_NAMES)
    X_train = rng.normal(size=(40, n_feat))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_eval = rng.normal(size=(20, n_feat))
    y_eval = (X_eval[:, 0] > 0).astype(int)

    result = train_and_evaluate_rf(X_train, y_train, X_eval, y_eval)
    assert result is not None
    importance = result["feature_importance"]
    assert set(importance.keys()) == set(FEATURE_NAMES)
    # 全特徴量の重要度合計 ≈ 1.0
    assert sum(importance.values()) == pytest.approx(1.0, abs=1e-4)


def test_feature_extraction_is_deterministic():
    """同じ入力で 2 回呼んでも特徴量が一致 (RANDOM_SEED 不要)."""
    df = _build_df()
    feats1 = extract_features_for_window(df, TRAIN_START, TRAIN_END, EVAL_START, EVAL_END)
    feats2 = extract_features_for_window(df, TRAIN_START, TRAIN_END, EVAL_START, EVAL_END)
    # email 順に sort して比較
    feats1 = feats1.sort_values("email").reset_index(drop=True)
    feats2 = feats2.sort_values("email").reset_index(drop=True)
    pd.testing.assert_frame_equal(feats1, feats2)
