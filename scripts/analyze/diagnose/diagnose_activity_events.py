"""
活動イベントテーブル (data/events/<proj>_events.csv) の診断スクリプト

確認内容:
  (a) per-dev event 系列長の分布 (median, mean, percentile)
  (b) (dev, dir) ペア軌跡の系列長分布（既存 B-16 と比較するため）
  (c) "今後 N ヶ月以内に何らかのイベント発生" のエンゲージメント正例率
  (d) "今後 N ヶ月以内に accept request" の従来正例率
  (e) イベント種別ごとの貢献度（系列をどれだけ伸ばすか）
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def percentile_summary(s: pd.Series, name: str) -> None:
    if s.empty:
        print(f"  [{name}] empty")
        return
    print(
        f"  [{name}] n={len(s):,} mean={s.mean():.1f} median={int(s.median())} "
        f"p25={int(s.quantile(0.25))} p75={int(s.quantile(0.75))} "
        f"p90={int(s.quantile(0.90))} p95={int(s.quantile(0.95))} "
        f"p99={int(s.quantile(0.99))} max={int(s.max())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--train-start", type=str, default="2019-01-01")
    parser.add_argument("--train-end", type=str, default="2022-01-01")
    parser.add_argument("--future-months", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    print(f"loading {args.events}")
    df = pd.read_csv(args.events)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df = df.dropna(subset=["timestamp"]).sort_values(["email", "timestamp"]).reset_index(drop=True)

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    print(f"\n=== filter to train period [{train_start.date()}, {train_end.date()}) ===")
    n_all = len(df)
    df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)].copy()
    print(f"events in train period: {len(df_train):,} / {n_all:,}")

    print(f"\n=== event_type distribution (train period) ===")
    et_counts = df_train["event_type"].value_counts()
    for et, c in et_counts.items():
        print(f"  {et}: {c:,} ({c/len(df_train)*100:.1f}%)")

    # (a) per-dev 系列長
    print(f"\n=== (a) per-developer 系列長分布 (train period) ===")
    per_dev = df_train.groupby("email").size()
    percentile_summary(per_dev, "per-dev events")
    bins = [(1, 5), (6, 20), (21, 50), (51, 100), (101, 200), (201, 10**9)]
    print("  系列長ビン:")
    for lo, hi in bins:
        n = int(((per_dev >= lo) & (per_dev <= hi)).sum())
        pct = n / len(per_dev) * 100 if len(per_dev) else 0.0
        label = f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"
        print(f"    {label}: {n:,} devs ({pct:.1f}%)")

    # 比較: request イベントだけだった場合の per-dev 系列長
    only_req = df_train[df_train["event_type"] == "request"].groupby("email").size()
    if len(only_req):
        print()
        percentile_summary(only_req, "per-dev request-only (B-16 baseline)")

    # (b) (dev, dir) ペア軌跡
    print(f"\n=== (b) (dev, dir) ペア軌跡の系列長分布 ===")
    expanded = []
    for _, row in df_train.iterrows():
        try:
            dirs = json.loads(row["dirs"])
        except (json.JSONDecodeError, TypeError):
            dirs = []
        for d in dirs:
            expanded.append((row["email"], d))
    pair_counts = pd.Series(Counter(expanded))
    percentile_summary(pair_counts, "(dev, dir) pair events")
    print("  系列長ビン:")
    for lo, hi in bins:
        n = int(((pair_counts >= lo) & (pair_counts <= hi)).sum())
        pct = n / len(pair_counts) * 100 if len(pair_counts) else 0.0
        label = f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"
        print(f"    {label}: {n:,} pairs ({pct:.1f}%)")

    # (c) (d) 未来窓ラベルの正例率
    print(
        f"\n=== (c)(d) 未来 {args.future_months}ヶ月以内のラベル正例率 (cutoff={train_end.date()}) ==="
    )
    future_end = train_end + pd.DateOffset(months=args.future_months)
    df_future = df[(df["timestamp"] >= train_end) & (df["timestamp"] < future_end)]
    print(f"future events in [{train_end.date()}, {future_end.date()}): {len(df_future):,}")

    # 評価対象は train 期間に何らか活動した dev
    eval_devs = set(per_dev.index)
    print(f"  eval devs (train 期間に活動あり): {len(eval_devs):,}")

    # (c) 何らかのイベント発生
    devs_any_future = set(df_future["email"].unique())
    pos_engagement = sum(1 for d in eval_devs if d in devs_any_future)
    rate_eng = pos_engagement / len(eval_devs) if eval_devs else 0.0
    print(f"  (c) engagement (future 何らかのイベント): {pos_engagement:,}/{len(eval_devs):,} = {rate_eng:.3f}")

    # (d) accept request
    df_future_req_pos = df_future[(df_future["event_type"] == "request") & (df_future["label"] == 1)]
    devs_accept_future = set(df_future_req_pos["email"].unique())
    pos_accept = sum(1 for d in eval_devs if d in devs_accept_future)
    rate_acc = pos_accept / len(eval_devs) if eval_devs else 0.0
    print(f"  (d) accept (future request accept):     {pos_accept:,}/{len(eval_devs):,} = {rate_acc:.3f}")

    # (c') (dev, dir) レベル engagement
    print(f"\n  (dev, dir) レベル正例率:")
    eval_pairs = set(pair_counts.index)
    expanded_future = []
    for _, row in df_future.iterrows():
        try:
            dirs = json.loads(row["dirs"])
        except (json.JSONDecodeError, TypeError):
            dirs = []
        for d in dirs:
            expanded_future.append((row["email"], d))
    pairs_any_future = set(expanded_future)
    pos_pair_eng = sum(1 for p in eval_pairs if p in pairs_any_future)
    rate_pair_eng = pos_pair_eng / len(eval_pairs) if eval_pairs else 0.0
    print(
        f"    pair engagement (future dir-touch): "
        f"{pos_pair_eng:,}/{len(eval_pairs):,} = {rate_pair_eng:.3f}"
    )

    expanded_future_acc = []
    for _, row in df_future_req_pos.iterrows():
        try:
            dirs = json.loads(row["dirs"])
        except (json.JSONDecodeError, TypeError):
            dirs = []
        for d in dirs:
            expanded_future_acc.append((row["email"], d))
    pairs_acc_future = set(expanded_future_acc)
    pos_pair_acc = sum(1 for p in eval_pairs if p in pairs_acc_future)
    rate_pair_acc = pos_pair_acc / len(eval_pairs) if eval_pairs else 0.0
    print(
        f"    pair accept     (future request accept): "
        f"{pos_pair_acc:,}/{len(eval_pairs):,} = {rate_pair_acc:.3f}"
    )

    # (e) 各イベント種別の貢献度
    print(f"\n=== (e) イベント種別ごとの per-dev 累積貢献 ===")
    for et in ["request", "comment", "vote", "patchset", "authored"]:
        sub = df_train[df_train["event_type"] == et]
        per_dev_et = sub.groupby("email").size()
        if per_dev_et.empty:
            continue
        n_devs = (per_dev_et > 0).sum()
        print(
            f"  {et:9s}: total={len(sub):,} devs_with_event={n_devs:,} "
            f"per_dev_median={int(per_dev_et.median())} mean={per_dev_et.mean():.1f}"
        )


if __name__ == "__main__":
    main()
