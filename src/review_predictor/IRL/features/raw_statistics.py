"""
raw_statistics.py

特徴量変換前の「純粋な生統計量」を抽出するモジュール。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 目的
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

このモジュールは、

    生ログ → raw統計量

のみを担当する。

ここでは:

- clip しない
- normalize しない
- min/max しない
- 0〜1圧縮しない
- heuristic scaling しない

ことを厳守する。

モデル入力用特徴量は別モジュール
(feature_transforms.py など) で生成する。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd


RAW_STAT_NAMES = [

    # activity
    "window_tenure_days",
    "total_reviews",
    "total_changes",
    "total_activity",

    # recent activity
    "recent_review_count_30d",

    # activity gap
    "mean_activity_gap_days",
    "median_activity_gap_days",
    "std_activity_gap_days",

    # last activity
    "days_since_last_activity",

    # collaboration
    "unique_collaborators",
    "repeated_collaborators",
    "repeat_collaboration_rate_raw",

    # acceptance
    "accepted_count",
    "rejected_count",
    "acceptance_rate_raw",
    "recent_acceptance_rate_raw",

    # change lines
    "raw_mean_change_lines",
    "raw_median_change_lines",
    "raw_std_change_lines",
    "raw_max_change_lines",

    # files
    "raw_mean_files",
    "raw_median_files",
    "raw_std_files",
    "raw_max_files",

    # response
    "raw_mean_response_days",
    "raw_median_response_days",
    "raw_std_response_days",
    "raw_max_response_days",

    # active months
    "active_months",
]


def extract_raw_statistics(
    df: pd.DataFrame,
    email: str,
    feature_start: datetime,
    feature_end: datetime,
) -> Dict[str, float]:
    """
    開発者の raw統計量を抽出する。

    IMPORTANT:
        この関数では特徴量変換を行わない。

    Args:
        df:
            review dataset

        email:
            対象開発者

        feature_start:
            観測開始時刻

        feature_end:
            観測終了時刻（予測時点）

    Returns:
        raw statistics dict
    """

    # reviewer履歴
    reviewer_mask = (
        (df["email"] == email)
        & (df["timestamp"] >= feature_start)
        & (df["timestamp"] < feature_end)
    )

    reviewer_data = df[reviewer_mask].copy()

    # owner履歴
    if "owner_email" in df.columns:

        owner_mask = (
            (df["owner_email"] == email)
            & (df["timestamp"] >= feature_start)
            & (df["timestamp"] < feature_end)
        )

        owner_data = df[owner_mask].copy()

    else:
        owner_data = pd.DataFrame()

    # データ完全なし
    if len(reviewer_data) == 0 and len(owner_data) == 0:
        return get_default_raw_statistics()

    reviewer_dates = reviewer_data["timestamp"].sort_values()

    # =====================================================
    # activity
    # =====================================================

    total_reviews = len(reviewer_data)
    total_changes = len(owner_data)
    total_activity = total_reviews + total_changes

    all_dates = pd.concat([
        reviewer_data["timestamp"],
        owner_data["timestamp"]
        if len(owner_data) > 0
        else pd.Series(dtype="datetime64[ns]")
    ]).dropna()

    if len(all_dates) > 0:

        first_seen = all_dates.min()

        window_tenure_days = max(
            (feature_end - first_seen).days,
            0
        )

    else:
        window_tenure_days = 0

    # =====================================================
    # recent activity
    # =====================================================

    recent_cutoff = feature_end - timedelta(days=30)

    recent_data = reviewer_data[
        reviewer_data["timestamp"] >= recent_cutoff
    ]

    recent_review_count_30d = len(recent_data)

    # =====================================================
    # activity gap
    # =====================================================

    if len(reviewer_dates) > 1:

        gaps = (
            reviewer_dates.diff()
            .dt.total_seconds()
            .dropna()
            / 86400.0
        )

        mean_activity_gap_days = float(gaps.mean())
        median_activity_gap_days = float(gaps.median())
        std_activity_gap_days = float(gaps.std())

    else:

        mean_activity_gap_days = np.nan
        median_activity_gap_days = np.nan
        std_activity_gap_days = np.nan

    # =====================================================
    # last activity
    # =====================================================

    if len(reviewer_dates) > 0:

        days_since_last_activity = (
            feature_end - reviewer_dates.max()
        ).days

    else:
        days_since_last_activity = np.nan

    # =====================================================
    # collaboration
    # =====================================================

    if "owner_email" in reviewer_data.columns:

        unique_collaborators = (
            reviewer_data["owner_email"]
            .nunique()
        )

        owner_counts = (
            reviewer_data["owner_email"]
            .value_counts()
        )

        repeated_collaborators = int(
            (owner_counts > 1).sum()
        )

        repeat_collaboration_rate_raw = (
            repeated_collaborators / len(owner_counts)
            if len(owner_counts) > 0
            else np.nan
        )

    else:

        unique_collaborators = np.nan
        repeated_collaborators = np.nan
        repeat_collaboration_rate_raw = np.nan

    # =====================================================
    # acceptance
    # =====================================================

    if (
        "label" in reviewer_data.columns
        and total_reviews > 0
    ):

        accepted_count = int(
            (reviewer_data["label"] == 1).sum()
        )

        rejected_count = int(
            (reviewer_data["label"] == 0).sum()
        )

        acceptance_rate_raw = (
            accepted_count / total_reviews
        )

        if len(recent_data) > 0:

            recent_acceptance_rate_raw = float(
                (recent_data["label"] == 1).mean()
            )

        else:
            recent_acceptance_rate_raw = np.nan

    else:

        accepted_count = np.nan
        rejected_count = np.nan
        acceptance_rate_raw = np.nan
        recent_acceptance_rate_raw = np.nan

    # =====================================================
    # change lines
    # =====================================================

    if (
        "change_insertions" in reviewer_data.columns
        and "change_deletions" in reviewer_data.columns
    ):

        change_lines = (
            reviewer_data["change_insertions"].fillna(0)
            + reviewer_data["change_deletions"].fillna(0)
        )

        raw_mean_change_lines = float(
            change_lines.mean()
        )

        raw_median_change_lines = float(
            change_lines.median()
        )

        raw_std_change_lines = float(
            change_lines.std()
        )

        raw_max_change_lines = float(
            change_lines.max()
        )

    else:

        raw_mean_change_lines = np.nan
        raw_median_change_lines = np.nan
        raw_std_change_lines = np.nan
        raw_max_change_lines = np.nan

    # =====================================================
    # files
    # =====================================================

    if "change_files_count" in reviewer_data.columns:

        files = reviewer_data["change_files_count"]

        raw_mean_files = float(files.mean())
        raw_median_files = float(files.median())
        raw_std_files = float(files.std())
        raw_max_files = float(files.max())

    else:

        raw_mean_files = np.nan
        raw_median_files = np.nan
        raw_std_files = np.nan
        raw_max_files = np.nan

    # =====================================================
    # response time
    # =====================================================

    if (
        "first_response_time" in reviewer_data.columns
        and len(reviewer_data) > 0
    ):

        response_df = reviewer_data.dropna(
            subset=["first_response_time"]
        )

        if len(response_df) > 0:

            response_days = (
                pd.to_datetime(
                    response_df["first_response_time"]
                )
                - pd.to_datetime(
                    response_df["timestamp"]
                )
            ).dt.total_seconds() / 86400.0

            raw_mean_response_days = float(
                response_days.mean()
            )

            raw_median_response_days = float(
                response_days.median()
            )

            raw_std_response_days = float(
                response_days.std()
            )

            raw_max_response_days = float(
                response_days.max()
            )

        else:

            raw_mean_response_days = np.nan
            raw_median_response_days = np.nan
            raw_std_response_days = np.nan
            raw_max_response_days = np.nan

    else:

        raw_mean_response_days = np.nan
        raw_median_response_days = np.nan
        raw_std_response_days = np.nan
        raw_max_response_days = np.nan

    # =====================================================
    # active months
    # =====================================================

    if len(reviewer_data) > 0:

        active_months = (
            reviewer_data["timestamp"]
            .dt.to_period("M")
            .nunique()
        )

    else:
        active_months = 0

    # =====================================================
    # return
    # =====================================================

    return {

        # activity
        "window_tenure_days": window_tenure_days,
        "total_reviews": total_reviews,
        "total_changes": total_changes,
        "total_activity": total_activity,

        # recent
        "recent_review_count_30d":
            recent_review_count_30d,

        # gaps
        "mean_activity_gap_days":
            mean_activity_gap_days,

        "median_activity_gap_days":
            median_activity_gap_days,

        "std_activity_gap_days":
            std_activity_gap_days,

        # last activity
        "days_since_last_activity":
            days_since_last_activity,

        # collaboration
        "unique_collaborators":
            unique_collaborators,

        "repeated_collaborators":
            repeated_collaborators,

        "repeat_collaboration_rate_raw":
            repeat_collaboration_rate_raw,

        # acceptance
        "accepted_count":
            accepted_count,

        "rejected_count":
            rejected_count,

        "acceptance_rate_raw":
            acceptance_rate_raw,

        "recent_acceptance_rate_raw":
            recent_acceptance_rate_raw,

        # change lines
        "raw_mean_change_lines":
            raw_mean_change_lines,

        "raw_median_change_lines":
            raw_median_change_lines,

        "raw_std_change_lines":
            raw_std_change_lines,

        "raw_max_change_lines":
            raw_max_change_lines,

        # files
        "raw_mean_files":
            raw_mean_files,

        "raw_median_files":
            raw_median_files,

        "raw_std_files":
            raw_std_files,

        "raw_max_files":
            raw_max_files,

        # response
        "raw_mean_response_days":
            raw_mean_response_days,

        "raw_median_response_days":
            raw_median_response_days,

        "raw_std_response_days":
            raw_std_response_days,

        "raw_max_response_days":
            raw_max_response_days,

        # months
        "active_months":
            active_months,
    }


def get_default_raw_statistics() -> Dict[str, float]:
    """
    raw統計量のデフォルト値。

    IMPORTANT:
        raw統計量なので、
        極力 0 or NaN を返す。
    """

    return {

        # activity
        "window_tenure_days": 0.0,
        "total_reviews": 0.0,
        "total_changes": 0.0,
        "total_activity": 0.0,

        # recent
        "recent_review_count_30d": 0.0,

        # gaps
        "mean_activity_gap_days": np.nan,
        "median_activity_gap_days": np.nan,
        "std_activity_gap_days": np.nan,

        # last
        "days_since_last_activity": np.nan,

        # collaboration
        "unique_collaborators": np.nan,
        "repeated_collaborators": np.nan,
        "repeat_collaboration_rate_raw": np.nan,

        # acceptance
        "accepted_count": np.nan,
        "rejected_count": np.nan,
        "acceptance_rate_raw": np.nan,
        "recent_acceptance_rate_raw": np.nan,

        # lines
        "raw_mean_change_lines": np.nan,
        "raw_median_change_lines": np.nan,
        "raw_std_change_lines": np.nan,
        "raw_max_change_lines": np.nan,

        # files
        "raw_mean_files": np.nan,
        "raw_median_files": np.nan,
        "raw_std_files": np.nan,
        "raw_max_files": np.nan,

        # response
        "raw_mean_response_days": np.nan,
        "raw_median_response_days": np.nan,
        "raw_std_response_days": np.nan,
        "raw_max_response_days": np.nan,

        # months
        "active_months": 0.0,
    }