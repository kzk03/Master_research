"""MCEBatchContinuationPredictor の推論動作テスト.

戦略: 実 .pt 成果物に依存せず、テスト内で
  1. ランダム重みの MCEIRLSystem を作成
  2. state_dict を tmp_path に保存
  3. MCEBatchContinuationPredictor で読み込み
  4. predict_* メソッドを呼んで出力契約 (形状・[0,1] 範囲) を検証
する.

カバー観点:
  - state_dim の auto-detect (state_encoder.0.weight.shape[1])
  - model_metadata.json による model_type の取り回し
  - predict_developer / predict_batch / predict_developer_directory /
    predict_developer_directories の戻り値型 + 範囲
  - 履歴ゼロ → 0.5 を返すフォールバック
  - dir-level 推論で path_extractor を渡しても落ちないこと
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import torch

from review_predictor.IRL.features.common_features import (
    ACTION_FEATURES,
    STATE_FEATURES,
    STATE_FEATURES_WITH_PATH,
)
from review_predictor.IRL.features.path_features import (
    PATH_FEATURE_DIM,
    PathFeatureExtractor,
    attach_dirs_to_df,
)
from review_predictor.IRL.model.mce_irl_batch_predictor import (
    MCEBatchContinuationPredictor,
)
from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem


# ─────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────


def _save_dummy_mce_model(
    tmp_path: Path, state_dim: int, model_type: int = 0
) -> Path:
    """ランダム重みの MCE-IRL モデルを state_dim 指定で保存し、パスを返す."""
    config = {
        "state_dim": state_dim,
        "action_dim": len(ACTION_FEATURES),
        "hidden_dim": 128,   # predictor 側がハードコードで使うため合わせる
        "dropout": 0.1,
        "model_type": model_type,
    }
    system = MCEIRLSystem(config)
    model_path = tmp_path / "mce_irl_model.pt"
    torch.save(system.network.state_dict(), model_path)

    metadata = {
        "model_class": "mce_irl",
        "model_type": model_type,
        "state_dim": state_dim,
        "action_dim": len(ACTION_FEATURES),
    }
    with open(tmp_path / "model_metadata.json", "w") as f:
        json.dump(metadata, f)
    return model_path


def _build_review_df() -> pd.DataFrame:
    """alice/bob が複数ヶ月にわたって履歴を持つ最小 DF.

    MCEBatchContinuationPredictor は reviewer_col='email',
    date_col='timestamp' をデフォルトで使う.
    """
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(10):
        rows.append({
            "email": "alice@x.com",
            "owner_email": "owner-x@example.com",
            "timestamp": base + pd.Timedelta(days=i * 15),
            "label": 1 if i % 2 == 0 else 0,
            "project": "openstack/nova",
            "change_id": f"openstack%2Fnova~{1000+i}",
            "first_response_time": None,
            "change_insertions": 10,
            "change_deletions": 5,
            "change_files_count": 2,
            "is_cross_project": False,
        })
    for i in range(8):
        rows.append({
            "email": "bob@x.com",
            "owner_email": "owner-y@example.com",
            "timestamp": base + pd.Timedelta(days=i * 20),
            "label": 1,
            "project": "openstack/nova",
            "change_id": f"openstack%2Fnova~{2000+i}",
            "first_response_time": None,
            "change_insertions": 20,
            "change_deletions": 8,
            "change_files_count": 3,
            "is_cross_project": False,
        })
    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────
# global model (state_dim = len(STATE_FEATURES))
# ─────────────────────────────────────────────────────────────────────


def test_predict_developer_returns_probability(tmp_path):
    """predict_developer は [0,1] 範囲の float を返す."""
    model_path = _save_dummy_mce_model(tmp_path, state_dim=len(STATE_FEATURES))
    df = _build_review_df()

    predictor = MCEBatchContinuationPredictor(
        model_path=model_path,
        df=df,
        history_start=datetime(2024, 1, 1),
    )
    prob = predictor.predict_developer(
        email="alice@x.com",
        prediction_time=datetime(2024, 6, 1),
    )
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_predict_developer_falls_back_when_no_history(tmp_path):
    """履歴ゼロの開発者は 0.5 を返すフォールバック."""
    model_path = _save_dummy_mce_model(tmp_path, state_dim=len(STATE_FEATURES))
    df = _build_review_df()

    predictor = MCEBatchContinuationPredictor(
        model_path=model_path,
        df=df,
        history_start=datetime(2024, 1, 1),
    )
    prob = predictor.predict_developer(
        email="unknown@nowhere.com",
        prediction_time=datetime(2024, 6, 1),
    )
    assert prob == pytest.approx(0.5)


def test_predict_batch_returns_dict_with_valid_scores(tmp_path):
    """predict_batch は emails と同じキーセットの辞書を返す."""
    model_path = _save_dummy_mce_model(tmp_path, state_dim=len(STATE_FEATURES))
    df = _build_review_df()

    predictor = MCEBatchContinuationPredictor(
        model_path=model_path,
        df=df,
        history_start=datetime(2024, 1, 1),
    )
    emails = ["alice@x.com", "bob@x.com", "unknown@nowhere.com"]
    result = predictor.predict_batch(
        emails=emails,
        prediction_time=datetime(2024, 6, 1),
    )
    assert set(result.keys()) == set(emails)
    for email, prob in result.items():
        assert 0.0 <= prob <= 1.0, f"{email}: {prob}"
    assert result["unknown@nowhere.com"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────
# directory-level model (state_dim = STATE + path)
# ─────────────────────────────────────────────────────────────────────


def _build_review_df_with_dirs() -> pd.DataFrame:
    df = _build_review_df()
    # change_id → dirs マッピング (テスト用に固定)
    cdm = {}
    for cid in df["change_id"].unique():
        if "1000" <= cid.split("~")[-1] < "2000":
            cdm[cid] = frozenset({"nova/compute"})
        else:
            cdm[cid] = frozenset({"nova/api"})
    df = attach_dirs_to_df(df, cdm, column="dirs")
    return df


def test_predict_developer_directory_with_path_extractor(tmp_path):
    """dir-level モデル (state_dim=STATE+path) + path_extractor で推論できる."""
    state_dim = len(STATE_FEATURES_WITH_PATH)
    assert state_dim == len(STATE_FEATURES) + PATH_FEATURE_DIM
    model_path = _save_dummy_mce_model(tmp_path, state_dim=state_dim)

    df = _build_review_df_with_dirs()
    path_extractor = PathFeatureExtractor(df, window_days=180)

    predictor = MCEBatchContinuationPredictor(
        model_path=model_path,
        df=df,
        history_start=datetime(2024, 1, 1),
    )
    prob = predictor.predict_developer_directory(
        email="alice@x.com",
        directory="nova/compute",
        prediction_time=datetime(2024, 6, 1),
        path_extractor=path_extractor,
    )
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_predict_developer_directories_returns_per_dir_scores(tmp_path):
    """複数 dir 一括推論で各 dir の確率が返る."""
    state_dim = len(STATE_FEATURES_WITH_PATH)
    model_path = _save_dummy_mce_model(tmp_path, state_dim=state_dim)
    df = _build_review_df_with_dirs()
    path_extractor = PathFeatureExtractor(df, window_days=180)

    predictor = MCEBatchContinuationPredictor(
        model_path=model_path,
        df=df,
        history_start=datetime(2024, 1, 1),
    )
    dirs = ["nova/compute", "nova/api", "nova/network"]
    result = predictor.predict_developer_directories(
        email="alice@x.com",
        directories=dirs,
        prediction_time=datetime(2024, 6, 1),
        path_extractor=path_extractor,
    )
    assert set(result.keys()) == set(dirs)
    for d, p in result.items():
        assert 0.0 <= p <= 1.0, f"{d}: {p}"


def test_predict_developer_directories_falls_back_for_unknown_email(tmp_path):
    """履歴ゼロでは全 dir に 0.5."""
    state_dim = len(STATE_FEATURES_WITH_PATH)
    model_path = _save_dummy_mce_model(tmp_path, state_dim=state_dim)
    df = _build_review_df_with_dirs()
    path_extractor = PathFeatureExtractor(df, window_days=180)

    predictor = MCEBatchContinuationPredictor(
        model_path=model_path,
        df=df,
        history_start=datetime(2024, 1, 1),
    )
    result = predictor.predict_developer_directories(
        email="unknown@nowhere.com",
        directories=["nova/compute", "nova/api"],
        prediction_time=datetime(2024, 6, 1),
        path_extractor=path_extractor,
    )
    assert result == {"nova/compute": 0.5, "nova/api": 0.5}


# ─────────────────────────────────────────────────────────────────────
# state_dim auto-detect
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "state_dim",
    [len(STATE_FEATURES), len(STATE_FEATURES_WITH_PATH)],
    ids=["global", "directory"],
)
def test_state_dim_is_inferred_from_state_dict(tmp_path, state_dim):
    """_load_model は state_encoder.0.weight.shape[1] から state_dim を復元する."""
    model_path = _save_dummy_mce_model(tmp_path, state_dim=state_dim)
    df = _build_review_df()

    predictor = MCEBatchContinuationPredictor(
        model_path=model_path,
        df=df,
        history_start=datetime(2024, 1, 1),
    )
    # 直接 _load_model を叩いて検証 (lazy load)
    predictor._load_model()
    assert predictor._irl_system is not None
    assert predictor._irl_system.state_dim == state_dim


# ─────────────────────────────────────────────────────────────────────
# 推論時に未来のレビューを参照しないか
# ─────────────────────────────────────────────────────────────────────


def test_prediction_does_not_use_future_data(tmp_path):
    """予測時刻より未来のレビューを混ぜても出力に影響しない (= 履歴に含めない)."""
    model_path = _save_dummy_mce_model(tmp_path, state_dim=len(STATE_FEATURES))
    df_clean = _build_review_df()

    # 同じ DF + 未来のレビューを 1 件追加した DF を作る
    future_row = {
        "email": "alice@x.com",
        "owner_email": "owner-x@example.com",
        "timestamp": pd.Timestamp("2030-01-01"),
        "label": 1,
        "project": "openstack/nova",
        "change_id": "openstack%2Fnova~9999",
        "first_response_time": None,
        "change_insertions": 999,
        "change_deletions": 999,
        "change_files_count": 99,
        "is_cross_project": False,
    }
    df_polluted = pd.concat(
        [df_clean, pd.DataFrame([future_row])], ignore_index=True
    )

    # シード固定で同じネットワーク重みを使う
    torch.manual_seed(42)
    pred_clean = MCEBatchContinuationPredictor(
        model_path=model_path, df=df_clean,
        history_start=datetime(2024, 1, 1),
    )
    prob_clean = pred_clean.predict_developer(
        "alice@x.com", datetime(2024, 6, 1)
    )

    torch.manual_seed(42)
    pred_polluted = MCEBatchContinuationPredictor(
        model_path=model_path, df=df_polluted,
        history_start=datetime(2024, 1, 1),
    )
    prob_polluted = pred_polluted.predict_developer(
        "alice@x.com", datetime(2024, 6, 1)
    )

    # _build_monthly_data は prediction_time 未満で切るため未来データは無視される
    assert prob_clean == pytest.approx(prob_polluted, abs=1e-6)
