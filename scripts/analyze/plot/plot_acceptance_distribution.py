# -*- coding: utf-8 -*-
"""
Acceptance rate distribution and per-period statistics (Fig 6.1 + 6.2).

Fig 6.1: Stacked bar chart of acceptance rate class composition over periods.
Fig 6.2: Boxplots of (a) request count, (b) acceptance rate, (c) response time per period.

Usage:
    uv run python scripts/analyze/plot/plot_acceptance_distribution.py \
        --input-dir outputs/cross_eval_results \
        --reviews data/review_requests_openstack_multi_5y_detail.csv \
        --output-dir outputs/cross_eval_results/discussion
"""

import argparse
import pathlib
from typing import List, Optional

import japanize_matplotlib  # noqa: F401  # 日本語フォント登録（副作用import）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

ACCEPTANCE_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.01]
ACCEPTANCE_LABELS = ["0-10%", "10-30%", "30-50%", "50-70%", "70-100%"]


def load_diagonal_predictions(input_dir: pathlib.Path) -> pd.DataFrame:
    """Load predictions.csv from diagonal patterns (train_X/eval_X)."""
    rows: List[pd.DataFrame] = []
    for period in PERIODS:
        csv_path = input_dir / f"train_{period}" / f"eval_{period}" / "predictions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["period"] = period
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No predictions.csv found in diagonal patterns under {input_dir}"
        )
    return pd.concat(rows, ignore_index=True)


def compute_acceptance_rate(df: pd.DataFrame) -> pd.Series:
    """Compute per-reviewer acceptance rate from eval columns.

    Reviewers with eval_request_count=0 are treated as 0% acceptance.
    """
    req = df["eval_request_count"]
    acc = df["eval_accepted_count"]
    return acc.div(req).fillna(0.0)


# Eval period base date (matches the cross-temporal experiment)
EVAL_BASE = pd.Timestamp("2023-01-01")


def _build_reviewer_period_df(
    reviews_df: pd.DataFrame,
    eval_base: pd.Timestamp = EVAL_BASE,
) -> pd.DataFrame:
    """Build per-reviewer acceptance rate DataFrame directly from the raw review CSV.

    This produces the correct figures for Fig 6.1 and 6.2 because it counts
    *all* reviewers present in each eval period, not only those who appear in
    a specific train/eval predictions.csv.

    Parameters
    ----------
    reviews_df : DataFrame with columns containing at least
        ``reviewer_email``, ``request_time``, ``label`` (1=accepted, 0=rejected).
    eval_base : start of the first eval window (default 2023-01-01).

    Returns
    -------
    DataFrame with columns: period, reviewer_email, eval_accepted_count,
    eval_request_count.  Compatible with ``compute_acceptance_rate()``.
    """
    df = reviews_df.copy()

    # --- normalise required columns ---
    # Find reviewer identity column
    email_col = next(
        (c for c in ["reviewer_email", "email", "developer_email"] if c in df.columns),
        None,
    )
    if email_col is None:
        raise KeyError("Raw CSV must have a reviewer identity column (reviewer_email / email / developer_email).")
    if email_col != "reviewer_email":
        df = df.rename(columns={email_col: "reviewer_email"})

    df["request_time"] = pd.to_datetime(df["request_time"], errors="coerce")
    df = df.dropna(subset=["request_time", "reviewer_email", "label"])
    df["label"] = df["label"].astype(int)

    rows: List[pd.DataFrame] = []
    for i, period in enumerate(PERIODS):
        start = eval_base + pd.DateOffset(months=i * 3)
        end = eval_base + pd.DateOffset(months=(i + 1) * 3)
        mask = (df["request_time"] >= start) & (df["request_time"] < end)
        period_df = df[mask]
        grp = (
            period_df.groupby("reviewer_email")["label"]
            .agg(["sum", "count"])
            .reset_index()
        )
        grp.columns = ["reviewer_email", "eval_accepted_count", "eval_request_count"]
        grp["period"] = period
        rows.append(grp)

    return pd.concat(rows, ignore_index=True)


def plot_acceptance_distribution(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Fig 6.1: Stacked bar chart of acceptance rate class composition."""
    df = df.copy()
    df["acceptance_rate"] = compute_acceptance_rate(df)
    df["rate_class"] = pd.cut(
        df["acceptance_rate"],
        bins=ACCEPTANCE_BINS,
        labels=ACCEPTANCE_LABELS,
        right=False,
    )

    # Count per period x class
    counts = (
        df.groupby(["period", "rate_class"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    # Reindex to ensure correct period order
    counts = counts.reindex(
        [p for p in PERIODS if p in counts.index]
    )

    # Convert to proportions
    proportions = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728", "#ff7f0e", "#ffcc00", "#2ca02c", "#1f77b4"]
    proportions.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white")

    ax.set_xlabel("評価期間")
    ax.set_ylabel("レビュアー比率")
    ax.set_title("図6.1: 承諾率階級の構成比の推移（レビュアー単位）")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="承諾率", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.png')}")


def plot_period_statistics(
    df: pd.DataFrame,
    reviews_df: Optional[pd.DataFrame],
    out_path: pathlib.Path,
) -> None:
    """Fig 6.2: Bar charts (median) of request count, acceptance rate, and response time."""
    df = df.copy()
    df["acceptance_rate"] = compute_acceptance_rate(df)

    has_response_time = reviews_df is not None
    ncols = 3 if has_response_time else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    available_periods = [p for p in PERIODS if p in df["period"].unique()]
    x = np.arange(len(available_periods))

    # (a) Request count – median bar
    ax = axes[0]
    medians_a = [
        df[df["period"] == p]["eval_request_count"].median() for p in available_periods
    ]
    bars = ax.bar(x, medians_a, color="#aec7e8", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(available_periods)
    ax.set_xlabel("評価期間")
    ax.set_ylabel("レビュー依頼数（中央値）")
    ax.set_title("(a) レビュー依頼数")
    for bar, val in zip(bars, medians_a):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.1f}", ha="center", va="bottom", fontsize=8,
        )

    # (b) Acceptance rate – median bar
    ax = axes[1]
    medians_b = [
        df[df["period"] == p]["acceptance_rate"].median() for p in available_periods
    ]
    bars = ax.bar(x, medians_b, color="#98df8a", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(available_periods)
    ax.set_xlabel("評価期間")
    ax.set_ylabel("承諾率（中央値）")
    ax.set_ylim(0, max(medians_b) * 1.3 if medians_b else 1)
    ax.set_title("(b) 承諾率")
    for bar, val in zip(bars, medians_b):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )

    # (c) Response time (if reviews data available) – median bar
    if has_response_time:
        ax = axes[2]
        response_data = _compute_response_times(reviews_df, available_periods)
        medians_c = []
        if response_data:
            medians_c = [
                float(np.median(response_data[p])) if response_data.get(p) else 0.0
                for p in available_periods
            ]
            bars = ax.bar(x, medians_c, color="#ffbb78", edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(available_periods)
            for bar, val in zip(bars, medians_c):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                )
        ax.set_xlabel("評価期間")
        ax.set_ylabel("平均応答日数（中央値）")
        ax.set_title("(c) 応答速度")

    fig.suptitle("図6.2: 評価期間別のレビュー依頼数・承諾率・応答速度（依頼数・承諾率は中央値）", y=1.02)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.png')}")


def _compute_response_times(
    reviews_df: pd.DataFrame,
    periods: List[str],
) -> dict:
    """Compute per-reviewer mean response days for each eval period."""
    # Determine eval period date ranges (2023 base)
    eval_base = pd.Timestamp("2023-01-01")
    period_ranges = {}
    for p in periods:
        start_m, end_m = [int(x.replace("m", "")) for x in p.split("-")]
        period_ranges[p] = (
            eval_base + pd.DateOffset(months=start_m),
            eval_base + pd.DateOffset(months=end_m),
        )

    df = reviews_df.copy()

    # Find timestamp columns
    if "request_time" in df.columns:
        df["request_time"] = pd.to_datetime(df["request_time"], errors="coerce")
    else:
        return {}

    if "first_response_time" in df.columns:
        df["first_response_time"] = pd.to_datetime(
            df["first_response_time"], errors="coerce"
        )
    else:
        return {}

    # Determine reviewer email column
    email_col = None
    for col in ["reviewer_email", "email", "developer_email"]:
        if col in df.columns:
            email_col = col
            break
    if email_col is None:
        return {}

    df["response_days"] = (
        df["first_response_time"] - df["request_time"]
    ).dt.total_seconds() / 86400.0
    df = df.dropna(subset=["response_days"])
    df = df[df["response_days"] >= 0]

    result = {}
    for p, (start, end) in period_ranges.items():
        mask = (df["request_time"] >= start) & (df["request_time"] < end)
        period_df = df[mask]
        if len(period_df) == 0:
            result[p] = []
            continue
        mean_per_reviewer = period_df.groupby(email_col)["response_days"].mean()
        result[p] = mean_per_reviewer.values.tolist()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot acceptance rate distribution (Fig 6.1) and period statistics (Fig 6.2)"
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Base directory containing train_*/eval_*/ results",
    )
    parser.add_argument(
        "--reviews",
        type=pathlib.Path,
        default=None,
        help="Raw review requests CSV (for response time in Fig 6.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory for output PNG/PDF files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally load raw review CSV for Fig 6.1 / 6.2
    reviews_df = None
    if args.reviews and args.reviews.exists():
        reviews_df = pd.read_csv(args.reviews)

    # Determine data source:
    # Raw CSV counts ALL reviewers in each eval period → correct acceptance-rate
    # distribution.  When raw CSV is absent we fall back to predictions.csv which
    # only includes reviewers that appear in the specific train/eval split.
    if reviews_df is not None:
        source_df = _build_reviewer_period_df(reviews_df)
    else:
        source_df = load_diagonal_predictions(args.input_dir)

    # Fig 6.1
    plot_acceptance_distribution(source_df, args.output_dir / "fig6_1_acceptance_distribution")

    # Fig 6.2
    plot_period_statistics(source_df, reviews_df, args.output_dir / "fig6_2_period_statistics")


if __name__ == "__main__":
    main()
