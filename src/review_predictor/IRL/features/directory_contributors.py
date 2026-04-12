"""
ディレクトリ × 開発者のマッピングと集計ユーティリティ

Step 1（パスごとの貢献者数予測）で使用する。
各ディレクトリについて「過去の貢献者集合」や「実際の貢献者数（Ground Truth）」を算出する。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, FrozenSet, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def get_directory_developers(
    df: pd.DataFrame,
    start_time: datetime,
    end_time: datetime,
    dirs_column: str = "dirs",
) -> Dict[str, Set[str]]:
    """
    期間内に各ディレクトリに関わった開発者（レビュアー）の集合を返す。

    Args:
        df: attach_dirs_to_df() で 'dirs' 列が付いた DataFrame
        start_time: 集計開始日
        end_time: 集計終了日
        dirs_column: ディレクトリ列名

    Returns:
        {directory: {email1, email2, ...}}
    """
    mask = (df["timestamp"] >= start_time) & (df["timestamp"] < end_time)
    sub = df.loc[mask]

    result: Dict[str, Set[str]] = {}
    for _, row in sub.iterrows():
        email = row.get("email")
        dirs = row.get(dirs_column)
        if not email or not dirs:
            continue
        for d in dirs:
            if d == ".":
                continue
            result.setdefault(d, set()).add(email)

    logger.info(
        f"get_directory_developers: {len(result)} dirs, "
        f"period {start_time} ~ {end_time}"
    )
    return result


def count_actual_contributors(
    df: pd.DataFrame,
    start_time: datetime,
    end_time: datetime,
    dirs_column: str = "dirs",
) -> Dict[str, int]:
    """
    Ground truth: 各ディレクトリの実際の貢献者数（ユニーク開発者数）。

    Args:
        df: attach_dirs_to_df() で 'dirs' 列が付いた DataFrame
        start_time: 集計開始日
        end_time: 集計終了日

    Returns:
        {directory: count}
    """
    dir_devs = get_directory_developers(df, start_time, end_time, dirs_column)
    return {d: len(devs) for d, devs in dir_devs.items()}


def get_all_directories(
    df: pd.DataFrame,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    dirs_column: str = "dirs",
) -> Set[str]:
    """
    期間内にアクティビティのあったディレクトリ集合を返す。
    start_time / end_time が None の場合は全期間。
    """
    if start_time is not None and end_time is not None:
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] < end_time)
        sub = df.loc[mask]
    else:
        sub = df

    dirs: Set[str] = set()
    for ds in sub[dirs_column]:
        if ds:
            dirs.update(d for d in ds if d != ".")
    return dirs
