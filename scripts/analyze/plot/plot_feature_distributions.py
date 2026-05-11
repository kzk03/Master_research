"""特徴量の「正真正銘の生分布」を可視化するスクリプト。

本スクリプトは raw_statistics.py の extract_raw_statistics() を使用する。
common_features.py の extract_common_features() は内部で
  - min(x, 1.0) クリップ（avg_action_intensity / avg_review_size / active_months_ratio など）
  - 1/(1+days/3) 変換（avg_response_time）
  - 60.0 / 730.0 等のフォールバック置換
  - normalize=True の場合 _NORM_CAPS による上限クリップ
を行うため、可視化結果が「加工後の値の分布」になってしまう。

ここでは clip も normalize も min/max も heuristic scaling も一切かけない
raw_statistics.py の出力を、そのままヒストグラムにする。

使い方:
    uv run python scripts/analyze/plot/plot_feature_distributions.py \\
        --reviews data/combined_raw.csv \\
        --train-start 2019-01-01 --train-end 2022-01-01 \\
        --output-dir outputs/feature_dist_raw

オプション:
    --snapshot-interval-months N : 何ヶ月ごとにスナップショットを取るか (default 1)
    --window-days N              : feature_start = feature_end - N 日 (default 180)
    --max-developers N           : デバッグ用、対象開発者を上位 N 人に絞る
    --label-split                : 各スナップショットで「今後 --future-months 月以内に
                                   レビュー承諾あり」かをラベル付けし、正負を重ね描き
    --clip-percentile P          : 表示時の上位クリップ百分位 (default 99.5)。
                                   集計値（mean/std/percentile）は常に生データから計算する。
    --log-scale                  : x軸を log スケールにする（強く右に裾を引く分布向け）
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from review_predictor.IRL.features.raw_statistics import (  # noqa: E402
    RAW_STAT_NAMES,
    extract_raw_statistics,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def _snapshot_one(df: pd.DataFrame, email: str, feature_end: pd.Timestamp, window_days: int):
    feature_start = feature_end - pd.Timedelta(days=window_days)
    feats = extract_raw_statistics(df, email, feature_start, feature_end)
    feats["__email"] = email
    feats["__snapshot"] = feature_end
    return feats


def _label_one(df: pd.DataFrame, email: str, t0: pd.Timestamp, future_months: int) -> int:
    t1 = t0 + pd.DateOffset(months=future_months)
    mask = (df["email"] == email) & (df["timestamp"] >= t0) & (df["timestamp"] < t1)
    sub = df[mask]
    if len(sub) == 0:
        return -1  # 依頼なし
    return int((sub["label"] == 1).any())


def collect_feature_snapshots(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    snapshot_interval_months: int,
    window_days: int,
    max_developers: int | None,
    n_jobs: int,
    label_split: bool,
    future_months: int = 3,
) -> pd.DataFrame:
    # 訓練期間中にレビュアーとして登場した開発者のみ対象
    mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
    reviewers = df.loc[mask, "email"].dropna().unique().tolist()
    if max_developers is not None:
        reviewers = reviewers[:max_developers]
    logger.info(f"対象 reviewer: {len(reviewers)} 人")

    # 月次スナップショット時刻
    snapshots: List[pd.Timestamp] = []
    t = train_start
    while t <= train_end:
        snapshots.append(t)
        t = t + pd.DateOffset(months=snapshot_interval_months)
    logger.info(f"スナップショット時刻数: {len(snapshots)}")

    tasks = [(email, t) for email in reviewers for t in snapshots]
    logger.info(f"特徴量抽出タスク: {len(tasks)} 件 (n_jobs={n_jobs})")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_snapshot_one)(df, email, t, window_days) for email, t in tasks
    )
    rows = pd.DataFrame(results)

    if label_split:
        logger.info("ラベル付け中（future_months=%d）", future_months)
        labels = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(_label_one)(df, email, t, future_months) for email, t in tasks
        )
        rows["__label"] = labels

    return rows


def compute_percentiles(values: pd.Series) -> dict:
    v = values.dropna().to_numpy()
    if len(v) == 0:
        return {
            k: float("nan")
            for k in ("mean", "std", "min", "p50", "p75", "p90", "p95", "p99", "max")
        } | {"n": 0}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "p50": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
        "n": int(len(v)),
    }


def plot_grid(
    df: pd.DataFrame,
    output_pdf: Path,
    label_split: bool,
    clip_percentile: float,
    log_scale: bool,
) -> None:
    feats = [f for f in RAW_STAT_NAMES if f in df.columns]
    n = len(feats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(feats):
        ax = axes[i]
        v = df[feat].dropna().to_numpy()
        if len(v) == 0:
            ax.set_title(f"{feat}\n(no data)")
            ax.axis("off")
            continue

        # 表示時のみ右側を clip_percentile で抑える（集計値は生のまま）
        if len(v) > 100:
            upper = np.percentile(v, clip_percentile)
        else:
            upper = v.max()
        show_mask = v <= upper

        # log スケール時は正値のみ
        def _filt(arr: np.ndarray) -> np.ndarray:
            arr = arr[arr <= upper]
            if log_scale:
                arr = arr[arr > 0]
            return arr

        if label_split and "__label" in df.columns:
            for lab, color, alpha in [(1, "C2", 0.5), (0, "C3", 0.5), (-1, "C7", 0.3)]:
                sub = df.loc[df["__label"] == lab, feat].dropna().to_numpy()
                sub = _filt(sub)
                if len(sub) == 0:
                    continue
                ax.hist(
                    sub,
                    bins=40,
                    alpha=alpha,
                    color=color,
                    label={1: "pos", 0: "neg", -1: "no-req"}[lab],
                    density=True,
                )
            ax.legend(fontsize=7)
        else:
            v_show = _filt(v)
            if len(v_show) > 0:
                ax.hist(v_show, bins=40, alpha=0.7, color="C0")

        # percentile lines (生データから計算)
        for p, ls in [(50, ":"), (90, "--"), (99, "-.")]:
            val = np.percentile(v, p)
            if log_scale and val <= 0:
                continue
            ax.axvline(val, color="k", ls=ls, lw=0.8, label=f"p{p}={val:.2f}")

        if log_scale:
            ax.set_xscale("log")

        n_total = len(v)
        n_nan = int(df[feat].isna().sum())
        ax.set_title(f"{feat}\nn={n_total} (nan={n_nan})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Raw feature distributions (no clip / no normalize, display cap=p{clip_percentile})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)
    logger.info(f"PDF 保存: {output_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reviews", required=True)
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--window-days", type=int, default=180)
    p.add_argument("--snapshot-interval-months", type=int, default=1)
    p.add_argument("--max-developers", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--label-split", action="store_true")
    p.add_argument("--future-months", type=int, default=3)
    p.add_argument("--clip-percentile", type=float, default=99.5)
    p.add_argument("--log-scale", action="store_true")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"読み込み: {args.reviews}")
    df = pd.read_csv(args.reviews)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"  rows={len(df)}, devs={df['email'].nunique()}")

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)

    rows = collect_feature_snapshots(
        df,
        train_start,
        train_end,
        args.snapshot_interval_months,
        args.window_days,
        args.max_developers,
        args.n_jobs,
        args.label_split,
        args.future_months,
    )
    csv_path = out / "raw_feature_values.csv"
    rows.to_csv(csv_path, index=False)
    logger.info(f"生データ保存: {csv_path} ({len(rows)} 行)")

    # percentile table
    pct_rows = []
    for feat in RAW_STAT_NAMES:
        if feat not in rows.columns:
            continue
        stats = compute_percentiles(rows[feat])
        stats["feature"] = feat
        pct_rows.append(stats)
    pct_df = pd.DataFrame(pct_rows).set_index("feature")
    pct_csv = out / "raw_feature_percentiles.csv"
    pct_df.to_csv(pct_csv)
    logger.info(f"percentile 保存: {pct_csv}")
    print(pct_df.to_string())

    plot_grid(
        rows,
        out / "raw_feature_distributions.pdf",
        args.label_split,
        args.clip_percentile,
        args.log_scale,
    )


if __name__ == "__main__":
    main()
