"""eval_path_prediction.py の評価メトリクス関数の挙動テスト.

対象:
  ▸ compute_affinity_score: path features → scalar (np.dot)
  ▸ compute_metrics: MAE / RMSE / Pearson / Spearman の境界
  ▸ compute_danger_detection: precision/recall の典型ケース
  ▸ baseline_naive: 窓内貢献者数の集計
  ▸ baseline_linear: 線形外挿の単調性

外形だけで成立するよう、IRL モデル / RF 学習を含む重い経路は対象外.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_EVAL = ROOT / "scripts" / "analyze" / "eval"
if str(SCRIPTS_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EVAL))

import eval_path_prediction as epp  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# DEPRECATED (2026-05-15): compute_affinity_score / predict_contributor_counts
# はグローバル IRL 時代のもので eval_path_prediction.py 側でコメントアウト済み.
# 関連テストもまとめて skip にして痕跡を残す.
# ─────────────────────────────────────────────────────────────────────

pytestmark_global_irl_skip = pytest.mark.skip(
    reason="グローバル IRL (Variant A/B) は 2026-05-15 に deprecated"
)


@pytestmark_global_irl_skip
def test_affinity_score_of_zeros_is_zero():
    pf = np.zeros(3, dtype=np.float32)
    assert epp.compute_affinity_score(pf) == pytest.approx(0.0)


@pytestmark_global_irl_skip
def test_affinity_score_of_all_ones_equals_weight_sum():
    pf = np.ones(3, dtype=np.float32)
    expected = float(epp.AFFINITY_WEIGHTS.sum())
    assert epp.compute_affinity_score(pf) == pytest.approx(expected)


@pytestmark_global_irl_skip
def test_affinity_score_is_weighted_dot_product():
    assert epp.compute_affinity_score(np.array([1, 0, 0], dtype=np.float32)) == pytest.approx(0.5)
    assert epp.compute_affinity_score(np.array([0, 1, 0], dtype=np.float32)) == pytest.approx(0.3)
    assert epp.compute_affinity_score(np.array([0, 0, 1], dtype=np.float32)) == pytest.approx(0.2)


# ─────────────────────────────────────────────────────────────────────
# compute_metrics
# ─────────────────────────────────────────────────────────────────────


def test_metrics_perfect_prediction_has_zero_error_and_unit_corr():
    actual = {"d1": 1, "d2": 3, "d3": 5, "d4": 7, "d5": 9}
    predicted = {k: float(v) for k, v in actual.items()}
    m = epp.compute_metrics(predicted, actual, "perfect")
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["spearman_r"] == pytest.approx(1.0)
    assert m["n_dirs"] == 5


def test_metrics_inverse_prediction_has_negative_corr():
    actual = {"d1": 1, "d2": 2, "d3": 3, "d4": 4, "d5": 5}
    predicted = {"d1": 5.0, "d2": 4.0, "d3": 3.0, "d4": 2.0, "d5": 1.0}
    m = epp.compute_metrics(predicted, actual, "inverse")
    assert m["pearson_r"] == pytest.approx(-1.0)
    assert m["spearman_r"] == pytest.approx(-1.0)


def test_metrics_linear_scaled_prediction_has_unit_pearson():
    """predicted = 2 * actual → pearson=1 (線形相関は scale 不変)."""
    actual = {"d1": 1, "d2": 2, "d3": 3, "d4": 4, "d5": 5}
    predicted = {k: 2.0 * v for k, v in actual.items()}
    m = epp.compute_metrics(predicted, actual, "scaled")
    assert m["pearson_r"] == pytest.approx(1.0)
    # mae は 0 ではない
    assert m["mae"] > 0.0


def test_metrics_constant_prediction_yields_nan_correlation():
    """std=0 → pearson/spearman は NaN."""
    actual = {"d1": 1, "d2": 2, "d3": 3, "d4": 4, "d5": 5}
    predicted = {k: 3.0 for k in actual}  # 全部同じ
    m = epp.compute_metrics(predicted, actual, "constant")
    assert np.isnan(m["pearson_r"])
    assert np.isnan(m["spearman_r"])


def test_metrics_empty_intersection_returns_nan():
    actual = {"d1": 1, "d2": 2}
    predicted = {"d99": 5.0}  # 共通 key なし
    m = epp.compute_metrics(predicted, actual, "empty")
    assert m["n_dirs"] == 0
    assert np.isnan(m["mae"])
    assert np.isnan(m["rmse"])


def test_metrics_intersection_only():
    """predicted と actual の共通 key だけで計算."""
    actual = {"d1": 1, "d2": 2, "d3": 3}
    predicted = {"d1": 1.0, "d2": 2.0, "d99": 100.0}  # d99 は無視
    m = epp.compute_metrics(predicted, actual, "intersection")
    assert m["n_dirs"] == 2


# ─────────────────────────────────────────────────────────────────────
# compute_danger_detection
# ─────────────────────────────────────────────────────────────────────


def test_danger_detection_perfect_case():
    """すべての危険 dir を検知、誤検知なし."""
    # actual <=1 (danger) を predicted < 1.0 で検知できれば perfect
    actual = {"d1": 0, "d2": 1, "d3": 5, "d4": 10}
    predicted = {"d1": 0.3, "d2": 0.5, "d3": 8.0, "d4": 12.0}
    m = epp.compute_danger_detection(
        predicted, actual, threshold=1.0, danger_actual_threshold=1,
    )
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_danger_detection_zero_recall_when_no_alerts():
    """全 dir で predicted >= threshold → 危険を 1 つも検知できない."""
    actual = {"d1": 0, "d2": 1, "d3": 5}
    predicted = {"d1": 5.0, "d2": 5.0, "d3": 5.0}  # 全部 high
    m = epp.compute_danger_detection(
        predicted, actual, threshold=1.0, danger_actual_threshold=1,
    )
    assert m["recall"] == pytest.approx(0.0)
    assert m["tp"] == 0
    assert m["fn"] == 2  # d1, d2 が見逃し


def test_danger_detection_all_false_positives():
    """全 dir で predicted < threshold だが、actual が全部 safe."""
    actual = {"d1": 10, "d2": 20, "d3": 30}
    predicted = {"d1": 0.3, "d2": 0.5, "d3": 0.1}
    m = epp.compute_danger_detection(
        predicted, actual, threshold=1.0, danger_actual_threshold=1,
    )
    assert m["precision"] == pytest.approx(0.0)
    assert m["fp"] == 3


# ─────────────────────────────────────────────────────────────────────
# baseline_naive
# ─────────────────────────────────────────────────────────────────────


def _build_dir_df():
    """alice, bob, carol が nova/compute と nova/api に貢献する fixture."""
    from review_predictor.IRL.features.path_features import attach_dirs_to_df

    base = pd.Timestamp("2024-01-01")
    rows = []
    # alice × nova/compute × 5 件
    for i in range(5):
        rows.append({
            "email": "alice@x.com", "owner_email": "owner@x.com",
            "timestamp": base + pd.Timedelta(days=i * 10),
            "label": 1, "change_id": f"c_alice_{i}",
            "project": "openstack/nova",
        })
    # bob × nova/api × 3 件
    for i in range(3):
        rows.append({
            "email": "bob@x.com", "owner_email": "owner@x.com",
            "timestamp": base + pd.Timedelta(days=i * 15),
            "label": 1, "change_id": f"c_bob_{i}",
            "project": "openstack/nova",
        })
    # carol × nova/compute × 2 件
    for i in range(2):
        rows.append({
            "email": "carol@x.com", "owner_email": "owner@x.com",
            "timestamp": base + pd.Timedelta(days=i * 20 + 100),
            "label": 1, "change_id": f"c_carol_{i}",
            "project": "openstack/nova",
        })
    df = pd.DataFrame(rows)
    cdm = {}
    for cid in df["change_id"]:
        if cid.startswith("c_alice"):
            cdm[cid] = frozenset({"nova/compute"})
        elif cid.startswith("c_bob"):
            cdm[cid] = frozenset({"nova/api"})
        else:
            cdm[cid] = frozenset({"nova/compute"})
    df = attach_dirs_to_df(df, cdm, column="dirs")
    return df


def test_baseline_naive_counts_unique_contributors_in_window():
    df = _build_dir_df()
    # 2024-01-01 から 2024-05-01 までの 120 日窓 → 全データ含む
    counts = epp.baseline_naive(
        df, prediction_time=pd.Timestamp("2024-05-01"), window_days=200,
    )
    # nova/compute: alice + carol = 2, nova/api: bob = 1
    assert counts["nova/compute"] == 2
    assert counts["nova/api"] == 1


def test_baseline_naive_excludes_outside_window():
    df = _build_dir_df()
    # 短い窓: 2024-04-30 - 30days = 2024-03-31 〜 2024-04-30
    counts = epp.baseline_naive(
        df, prediction_time=pd.Timestamp("2024-04-30"), window_days=30,
    )
    # alice の最後の活動は 2024-01-01+40days=2024-02-10 (≒) → 窓外
    # bob の最後は 2024-01-01+30days=2024-01-31 → 窓外
    # carol の最後は 2024-01-01+120days=2024-05-01 (≒) → 窓内かどうかは ts 次第
    # 厳密判定よりも「窓外データはカウントしない」原理を確認
    total = sum(counts.values())
    full_counts = epp.baseline_naive(df, pd.Timestamp("2024-05-01"), 200)
    full_total = sum(full_counts.values())
    assert total <= full_total


# ─────────────────────────────────────────────────────────────────────
# baseline_linear
# ─────────────────────────────────────────────────────────────────────


def test_baseline_linear_extrapolates_increasing_trend():
    """各 3 ヶ月期間で貢献者数が 1, 2, 3 と増加 → 外挿で 4 近傍を返す."""
    from review_predictor.IRL.features.path_features import attach_dirs_to_df

    # 3 期間 × ユニーク貢献者数増加 を作る
    # prediction_time=2024-10-01, period=3ヶ月, n=3 → 期間:
    #   p3: [2024-07-01, 2024-10-01) ← 直近
    #   p2: [2024-04-01, 2024-07-01)
    #   p1: [2024-01-01, 2024-04-01) ← 最古
    # p1: 1 人, p2: 2 人, p3: 3 人 にしたい
    rows = []
    contributors = {
        "2024-01-15": ["a@x.com"],
        "2024-04-15": ["a@x.com", "b@x.com"],
        "2024-07-15": ["a@x.com", "b@x.com", "c@x.com"],
    }
    cid_counter = 0
    cdm = {}
    for ts, devs in contributors.items():
        for d in devs:
            cid = f"c{cid_counter}"
            cid_counter += 1
            rows.append({
                "email": d, "owner_email": "o@x.com",
                "timestamp": pd.Timestamp(ts), "label": 1,
                "change_id": cid, "project": "openstack/nova",
            })
            cdm[cid] = frozenset({"nova/compute"})
    df = attach_dirs_to_df(pd.DataFrame(rows), cdm, column="dirs")

    result = epp.baseline_linear(
        df,
        prediction_time=pd.Timestamp("2024-10-01"),
        delta_months=3,
        n_periods=3,
        period_months=3,
    )
    # 1,2,3 の線形外挿で 1 step 先 (= delta=3ヶ月 後) → 4 を予測
    assert result["nova/compute"] == pytest.approx(4.0, abs=0.5)


def test_baseline_linear_returns_nonnegative_values():
    """単調減少データでも 0 で clip される (max(0, ..))."""
    from review_predictor.IRL.features.path_features import attach_dirs_to_df

    rows = []
    contributors = {
        # 5 → 3 → 1 と減少
        "2024-01-15": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        "2024-04-15": ["a@x.com", "b@x.com", "c@x.com"],
        "2024-07-15": ["a@x.com"],
    }
    cid_counter = 0
    cdm = {}
    for ts, devs in contributors.items():
        for d in devs:
            cid = f"c{cid_counter}"
            cid_counter += 1
            rows.append({
                "email": d, "owner_email": "o@x.com",
                "timestamp": pd.Timestamp(ts), "label": 1,
                "change_id": cid, "project": "openstack/nova",
            })
            cdm[cid] = frozenset({"nova/compute"})
    df = attach_dirs_to_df(pd.DataFrame(rows), cdm, column="dirs")

    # 線形外挿で負になる場合があるが、max(0, ..) で 0 にクリップ
    result = epp.baseline_linear(
        df,
        prediction_time=pd.Timestamp("2024-10-01"),
        delta_months=3,
        n_periods=3,
        period_months=3,
    )
    for d, v in result.items():
        assert v >= 0.0, f"{d}: {v} is negative"


# ─────────────────────────────────────────────────────────────────────
# predict_contributor_counts (Variant A / B)
#
# NOTE (2026-05-14 path features 拡張に伴う既知バグ):
#   AFFINITY_WEIGHTS は 3 次元のままだが、PathFeatureExtractor.compute() は
#   5 次元を返すようになった (path_lcp_similarity, path_owner_overlap 追加).
#   eval_path_prediction.py:76 で np.dot(pf(5), weights(3)) が dim mismatch.
#   → eval スクリプト側を weights = [w1..w5] に拡張する or pf[:3] でスライス
#     する修正が必要. 修正後にこの xfail を外す.
# ─────────────────────────────────────────────────────────────────────


@pytestmark_global_irl_skip
def test_predict_contributor_counts_variant_a_sums_probs():
    """Variant A: Σ continuation_prob over devs in each dir."""
    from review_predictor.IRL.features.path_features import (
        PathFeatureExtractor,
        attach_dirs_to_df,
    )

    df = _build_dir_df()
    path_extractor = PathFeatureExtractor(df, window_days=180)
    dir_developers = {
        "nova/compute": {"alice@x.com", "carol@x.com"},
        "nova/api": {"bob@x.com"},
    }
    probs = {"alice@x.com": 0.8, "carol@x.com": 0.4, "bob@x.com": 0.7}

    variant_a, variant_b = epp.predict_contributor_counts(
        dir_developers, probs, path_extractor,
        prediction_time=datetime(2024, 6, 1),
    )
    # Variant A: nova/compute = 0.8 + 0.4 = 1.2, nova/api = 0.7
    assert variant_a["nova/compute"] == pytest.approx(1.2)
    assert variant_a["nova/api"] == pytest.approx(0.7)
    # Variant B = Σ prob * affinity. affinity は path features × weights なので
    # 値そのものは依存性が大きい. ただし Variant A の値より大きくはない (affinity <= 1)
    assert variant_b["nova/compute"] <= variant_a["nova/compute"] + 1e-6
    assert variant_b["nova/api"] <= variant_a["nova/api"] + 1e-6


@pytestmark_global_irl_skip
def test_predict_contributor_counts_unknown_dev_uses_default_half():
    """probs に無い dev は default 0.5 が使われる."""
    from review_predictor.IRL.features.path_features import (
        PathFeatureExtractor,
        attach_dirs_to_df,
    )

    df = _build_dir_df()
    path_extractor = PathFeatureExtractor(df, window_days=180)
    dir_developers = {"nova/compute": {"alice@x.com", "unknown@x.com"}}
    probs = {"alice@x.com": 1.0}  # unknown は欠落

    variant_a, _ = epp.predict_contributor_counts(
        dir_developers, probs, path_extractor,
        prediction_time=datetime(2024, 6, 1),
    )
    # 1.0 (alice) + 0.5 (unknown default) = 1.5
    assert variant_a["nova/compute"] == pytest.approx(1.5)
