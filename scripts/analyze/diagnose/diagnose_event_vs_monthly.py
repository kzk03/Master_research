#!/usr/bin/env python3
"""
イベント版 vs 月次版の (reviewer, directory) ペア集合差を診断する。

両者の trajectory 抽出ロジックを単純化して、ペア集合と (reviewer, directory) ごとの
レビュー件数（= イベント版 seq_len の上限）を集計する。

使い方:
    uv run python scripts/analyze/diagnose/diagnose_event_vs_monthly.py \
        --reviews data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --periods "2019-01-01,2022-01-01" "2021-01-01,2023-01-01"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_df(reviews_path: str, raw_json_paths: list[str]) -> pd.DataFrame:
    from review_predictor.IRL.features.path_features import (
        attach_dirs_to_df,
        load_change_dir_map,
        load_change_dir_map_multi,
    )

    df = pd.read_csv(reviews_path)
    if "email" in df.columns and "reviewer_email" not in df.columns:
        df = df.rename(columns={"email": "reviewer_email"})
    if "timestamp" in df.columns and "request_time" not in df.columns:
        df = df.rename(columns={"timestamp": "request_time"})
    df["request_time"] = pd.to_datetime(df["request_time"])

    if len(raw_json_paths) == 1:
        cdm = load_change_dir_map(raw_json_paths[0], depth=2)
    else:
        cdm = load_change_dir_map_multi(raw_json_paths, depth=2)
    df = attach_dirs_to_df(df, cdm, column="dirs")
    return df


def collect_pairs(df: pd.DataFrame, train_start: pd.Timestamp, train_end: pd.Timestamp,
                  max_events: int = 256) -> dict:
    history = df[(df["request_time"] >= train_start) & (df["request_time"] < train_end)]
    pair_counts: dict[tuple[str, str], int] = {}
    for _, row in history.iterrows():
        ds = row["dirs"]
        if not ds:
            continue
        rev = row["reviewer_email"]
        for d in ds:
            if d == ".":
                continue
            key = (rev, d)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    counts = np.array(list(pair_counts.values()))
    n_pairs = len(counts)
    truncated = int((counts > max_events).sum())
    return {
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "duration_months": int(round((train_end - train_start).days / 30.4375)),
        "n_reviews_in_window": int(len(history)),
        "n_active_reviewers": int(history["reviewer_email"].nunique()),
        "n_pairs": n_pairs,
        "events_per_pair_mean": float(counts.mean()) if n_pairs else 0.0,
        "events_per_pair_median": float(np.median(counts)) if n_pairs else 0.0,
        "events_per_pair_p95": float(np.percentile(counts, 95)) if n_pairs else 0.0,
        "events_per_pair_p99": float(np.percentile(counts, 99)) if n_pairs else 0.0,
        "events_per_pair_max": int(counts.max()) if n_pairs else 0,
        "n_pairs_truncated_at_256": truncated,
        "pct_pairs_truncated": float(truncated / n_pairs * 100) if n_pairs else 0.0,
        "total_events_capped": int(np.minimum(counts, max_events).sum()) if n_pairs else 0,
        "total_events_uncapped": int(counts.sum()) if n_pairs else 0,
        "pair_counts": pair_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default="data/combined_raw.csv")
    ap.add_argument("--raw-json", nargs="+", default=[
        f"data/raw_json/openstack__{p}.json"
        for p in ["nova", "cinder", "neutron", "ironic", "glance",
                  "keystone", "horizon", "swift", "heat", "octavia"]
    ])
    ap.add_argument("--periods", nargs="+", required=True,
                    help='"YYYY-MM-DD,YYYY-MM-DD" pairs')
    ap.add_argument("--max-events", type=int, default=256)
    args = ap.parse_args()

    print("Loading data...", flush=True)
    df = load_df(args.reviews, args.raw_json)
    print(f"  {len(df)} review rows, "
          f"{df['reviewer_email'].nunique()} unique reviewers", flush=True)

    results = []
    pairs_per_period = []
    for p in args.periods:
        s, e = p.split(",")
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        print(f"\n=== {ts.date()} ~ {te.date()} ===", flush=True)
        r = collect_pairs(df, ts, te, max_events=args.max_events)
        pairs_per_period.append(set(r.pop("pair_counts").keys()))
        results.append(r)
        for k, v in r.items():
            print(f"  {k}: {v}")

    if len(results) >= 2:
        a, b = pairs_per_period[0], pairs_per_period[1]
        print(f"\n=== Pair set overlap ===")
        print(f"  Period A only: {len(a - b)}")
        print(f"  Period B only: {len(b - a)}")
        print(f"  Both:          {len(a & b)}")
        print(f"  Jaccard:       {len(a & b) / len(a | b):.4f}")

    out_df = pd.DataFrame(results)
    out_path = Path("outputs/event_vs_monthly_diagnosis.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
