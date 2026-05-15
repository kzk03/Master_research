"""extract_review_acceptance_trajectories のラベル付与ロジック (4 分岐 + PU モード) のテスト.

train_model.py L266-313 の分岐を網羅:

  1. 正例 (label=1, weight=1.0):
       訓練期間に依頼あり & ラベル期間に依頼あり & 1 件以上承諾
  2. 確実な負例 (label=0, weight=1.0):
       訓練期間に依頼あり & ラベル期間に依頼あり & 全拒否
  3. 弱負例 (label=0, weight=0.1):
       訓練期間に依頼あり & ラベル期間に依頼なし & 拡張期間 (0-12m) に依頼あり
  4a. 除外:
       訓練期間に依頼あり & ラベル期間にも拡張期間にも依頼なし
       (pu_unlabeled_weight=0, デフォルト)
  4b. PU 弱負例 (sample_weight=pu_unlabeled_weight):
       上と同じ条件で pu_unlabeled_weight > 0 のとき (PU 学習モード)

「訓練期間に依頼なし」の開発者は active_reviewers に入らないため軌跡化されない.
これは別途 test_trajectory_leak.py で検証済み.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_TRAIN = ROOT / "scripts" / "train"
if str(SCRIPTS_TRAIN) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_TRAIN))

from train_model import extract_review_acceptance_trajectories  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# fixture
# ─────────────────────────────────────────────────────────────────────

TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2024-05-01")
# future window 0-3m → ラベル期間 = [2024-05-01, 2024-08-01)
# 拡張期間 12m → [2024-05-01, 2025-01-01)


def _row(reviewer: str, ts: str, label: int,
         project: str = "openstack/nova",
         owner: str = "owner@example.com") -> dict:
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
    rows = []

    # alice: 訓練期間 4 件 + ラベル期間 (2024-06) に承諾 → 正例
    for d in ("2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"):
        rows.append(_row("alice@x.com", d, label=1))
    rows.append(_row("alice@x.com", "2024-06-01", label=1))

    # bob: 訓練期間 4 件 + ラベル期間 (2024-06, 2024-07) に依頼ありだが全部拒否 → 確実な負例
    for d in ("2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"):
        rows.append(_row("bob@x.com", d, label=1))
    rows.append(_row("bob@x.com", "2024-06-01", label=0))
    rows.append(_row("bob@x.com", "2024-07-01", label=0))

    # carol: 訓練期間 4 件 + ラベル期間 (0-3m) に依頼なし + 拡張期間後半 (2024-10) に依頼あり
    #        → 弱負例 (weight=0.1)
    for d in ("2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"):
        rows.append(_row("carol@x.com", d, label=1))
    rows.append(_row("carol@x.com", "2024-10-15", label=1))

    # dave: 訓練期間 4 件 + ラベル期間にも拡張期間 (0-12m) にも依頼なし
    #        → pu_unlabeled_weight=0 で除外 / >0 で PU 弱負例
    for d in ("2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"):
        rows.append(_row("dave@x.com", d, label=1))

    return pd.DataFrame(rows)


def _run(pu_weight: float = 0.0):
    return extract_review_acceptance_trajectories(
        _build_df(),
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        future_window_start_months=0,
        future_window_end_months=3,
        min_history_requests=0,
        extended_label_window_months=12,
        negative_oversample_factor=1,
        pu_unlabeled_weight=pu_weight,
    )


def _get(trajs, reviewer):
    matched = [t for t in trajs if t.get("reviewer") == reviewer]
    return matched[0] if matched else None


# ─────────────────────────────────────────────────────────────────────
# 1. 正例
# ─────────────────────────────────────────────────────────────────────


def test_alice_is_positive_with_full_weight():
    """ラベル期間に 1 件でも承諾があれば正例 (weight=1.0)."""
    alice = _get(_run(), "alice@x.com")
    assert alice is not None
    assert alice["future_acceptance"] is True
    assert alice["sample_weight"] == pytest.approx(1.0)
    assert alice["had_requests"] is True
    assert alice["label_accepted_count"] >= 1


# ─────────────────────────────────────────────────────────────────────
# 2. 確実な負例
# ─────────────────────────────────────────────────────────────────────


def test_bob_is_reliable_negative_with_full_weight():
    """ラベル期間に依頼あり + 全拒否 → 確実な負例 (weight=1.0)."""
    bob = _get(_run(), "bob@x.com")
    assert bob is not None
    assert bob["future_acceptance"] is False
    assert bob["sample_weight"] == pytest.approx(1.0)
    assert bob["had_requests"] is True
    assert bob["label_request_count"] >= 1
    assert bob["label_accepted_count"] == 0
    assert bob["label_rejected_count"] >= 1


# ─────────────────────────────────────────────────────────────────────
# 3. 弱負例 (依頼なし + 拡張期間に依頼あり)
# ─────────────────────────────────────────────────────────────────────


def test_carol_is_weak_negative_with_low_weight():
    """ラベル期間に依頼なしだが拡張期間に依頼あり → 弱負例 (weight=0.1)."""
    carol = _get(_run(), "carol@x.com")
    assert carol is not None
    assert carol["future_acceptance"] is False
    assert carol["sample_weight"] == pytest.approx(0.1)
    assert carol["had_requests"] is False
    assert carol["label_request_count"] == 0


# ─────────────────────────────────────────────────────────────────────
# 4a. 除外 (pu_unlabeled_weight=0)
# ─────────────────────────────────────────────────────────────────────


def test_dave_is_excluded_when_pu_weight_is_zero():
    """ラベル期間にも拡張期間にも依頼なし + PU 無効 → 軌跡化されない."""
    trajs = _run(pu_weight=0.0)
    dave = _get(trajs, "dave@x.com")
    assert dave is None, "PU 無効時は完全離脱した開発者は除外されるはず"


# ─────────────────────────────────────────────────────────────────────
# 4b. PU 弱負例 (pu_unlabeled_weight>0)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pu_w", [0.01, 0.05, 0.2])
def test_dave_becomes_pu_negative_when_pu_weight_is_positive(pu_w):
    """PU 学習モードでは完全離脱者も低重みの負例として扱う.

    sample_weight は pu_unlabeled_weight の値そのものになる
    (train_model.py L285).
    """
    trajs = _run(pu_weight=pu_w)
    dave = _get(trajs, "dave@x.com")
    assert dave is not None, "PU 有効時は完全離脱者も軌跡化されるはず"
    assert dave["future_acceptance"] is False
    assert dave["sample_weight"] == pytest.approx(pu_w)
    assert dave["had_requests"] is False


# ─────────────────────────────────────────────────────────────────────
# 整合性: 4 ケースの合計
# ─────────────────────────────────────────────────────────────────────


def test_all_four_categories_appear_with_pu_enabled():
    """PU 有効時は 4 名全員が軌跡化される (alice/bob/carol/dave)."""
    trajs = _run(pu_weight=0.05)
    names = sorted(t["reviewer"] for t in trajs)
    assert names == ["alice@x.com", "bob@x.com", "carol@x.com", "dave@x.com"]


def test_only_three_categories_appear_with_pu_disabled():
    """PU 無効時は dave が除外され 3 名のみ."""
    trajs = _run(pu_weight=0.0)
    names = sorted(t["reviewer"] for t in trajs)
    assert names == ["alice@x.com", "bob@x.com", "carol@x.com"]


def test_sample_weight_ordering():
    """重みの大小関係: 正例/確実な負例 (1.0) > 弱負例 (0.1) > PU 弱負例 (0.05)."""
    trajs = _run(pu_weight=0.05)
    by_name = {t["reviewer"]: t for t in trajs}
    assert by_name["alice@x.com"]["sample_weight"] == pytest.approx(1.0)
    assert by_name["bob@x.com"]["sample_weight"] == pytest.approx(1.0)
    assert by_name["carol@x.com"]["sample_weight"] == pytest.approx(0.1)
    assert by_name["dave@x.com"]["sample_weight"] == pytest.approx(0.05)
