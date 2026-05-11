"""特徴量の生分布を可視化するスクリプト（クリップ値検討用）。

訓練期間内の月次スナップショットで extract_common_features を回し、
特徴量ごとに p50/p75/p90/p95/p99/max を集計し、ヒストグラム＋現行 _NORM_CAPS
ラインを 1 枚の PDF にまとめる。

使い方:
    uv run python scripts/analyze/plot/plot_feature_distributions.py \\
        --reviews data/combined_raw_main32.csv \\
        --train-start 2019-01-01 --train-end 2022-01-01 \\
        --output-dir outputs/feature_dist_main32

オプション:
    --snapshot-interval-months N : 何ヶ月ごとにスナップショットを取るか (default 1)
    --window-days N              : feature_start = feature_end - N 日 (default 180)
    --max-developers N           : デバッグ用、対象開発者を上位 N 人に絞る
    --label-split                : 各スナップショットで「今後3ヶ月以内にレビュー承諾あり」
                                   かどうかをラベル付けし、正負を重ね描き
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

from review_predictor.IRL.features.common_features import (  # noqa: E402
    FEATURE_NAMES,
    _NORM_CAPS,
    extract_common_features,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def _snapshot_one(df: pd.DataFrame, email: str, feature_end: pd.Timestamp, window_days: int):
    feature_start = feature_end - pd.Timedelta(days=window_days)
    feats = extract_common_features(df, email, feature_start, feature_end, normalize=False)
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
        return {k: float("nan") for k in ("mean", "std", "p50", "p75", "p90", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p50": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
        "n": int(len(v)),
    }


def plot_grid(df: pd.DataFrame, output_pdf: Path, label_split: bool) -> None:
    feats = [f for f in FEATURE_NAMES if f in df.columns]
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
            continue

        # クリッピング: p99.5 までで表示（外れ値で潰れないように）
        v_show = v[v <= np.percentile(v, 99.5)] if len(v) > 100 else v

        if label_split and "__label" in df.columns:
            for lab, color, alpha in [(1, "C2", 0.5), (0, "C3", 0.5), (-1, "C7", 0.3)]:
                sub = df.loc[df["__label"] == lab, feat].dropna().to_numpy()
                if len(sub) == 0:
                    continue
                sub_show = sub[sub <= np.percentile(v, 99.5)] if len(v) > 100 else sub
                ax.hist(sub_show, bins=40, alpha=alpha, color=color,
                        label={1: "pos", 0: "neg", -1: "no-req"}[lab], density=True)
            ax.legend(fontsize=7)
        else:
            ax.hist(v_show, bins=40, alpha=0.7, color="C0")

        # percentile lines
        for p, ls in [(90, ":"), (95, "--"), (99, "-.")]:
            ax.axvline(np.percentile(v, p), color="k", ls=ls, lw=0.8,
                       label=f"p{p}={np.percentile(v, p):.1f}")
        # 現行キャップ
        if feat in _NORM_CAPS:
            ax.axvline(_NORM_CAPS[feat], color="red", ls="-", lw=1.2,
                       label=f"cap={_NORM_CAPS[feat]}")
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature distributions (raw, pre-cap)", fontsize=12)
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
        df, train_start, train_end,
        args.snapshot_interval_months, args.window_days,
        args.max_developers, args.n_jobs,
        args.label_split, args.future_months,
    )
    csv_path = out / "feature_values.csv"
    rows.to_csv(csv_path, index=False)
    logger.info(f"生データ保存: {csv_path} ({len(rows)} 行)")

    # percentile table
    pct_rows = []
    for feat in FEATURE_NAMES:
        if feat not in rows.columns:
            continue
        stats = compute_percentiles(rows[feat])
        stats["feature"] = feat
        stats["current_cap"] = _NORM_CAPS.get(feat, None)
        pct_rows.append(stats)
    pct_df = pd.DataFrame(pct_rows).set_index("feature")
    pct_csv = out / "feature_percentiles.csv"
    pct_df.to_csv(pct_csv)
    logger.info(f"percentile 保存: {pct_csv}")
    print(pct_df.to_string())

    plot_grid(rows, out / "feature_distributions.pdf", args.label_split)


if __name__ == "__main__":
    main()
