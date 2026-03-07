# -*- coding: utf-8 -*-
"""
Activity pattern classification accuracy for IRL and RF (Fig 6.5).

Classifies reviewers present in all 4 diagonal periods into:
  - Consistent acceptors: true_label=1 in all periods
  - Consistent rejectors: true_label=0 in all periods
  - Variable: true_label changes across periods

Then computes IRL/RF accuracy per group and plots a grouped bar chart.

Usage:
    uv run python scripts/analyze/plot_activity_patterns.py \
        --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/discussion
"""

import argparse
import pathlib
from typing import Dict, List

import japanize_matplotlib  # noqa: F401  # 日本語フォント登録（副作用import）  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

PATTERN_CONSISTENT_ACCEPT = "一貫承諾型"
PATTERN_CONSISTENT_REJECT = "一貫非承諾型"
PATTERN_VARIABLE = "変動型"

PATTERN_ORDER = [PATTERN_CONSISTENT_ACCEPT, PATTERN_CONSISTENT_REJECT, PATTERN_VARIABLE]


def load_diagonal_predictions_with_rf(
    input_dir: pathlib.Path,
) -> Dict[str, pd.DataFrame]:
    """Load predictions.csv (with RF columns) from diagonal patterns.

    Returns:
        Dict mapping period name to DataFrame (only rows with RF predictions).
    """
    result = {}
    for period in PERIODS:
        csv_path = input_dir / f"train_{period}" / f"eval_{period}" / "predictions.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if "rf_predicted_binary" not in df.columns:
            continue

        df = df.dropna(subset=["rf_predicted_binary"])
        if len(df) == 0:
            continue

        result[period] = df

    if not result:
        raise FileNotFoundError(
            f"No predictions.csv with RF columns found in diagonal patterns under {input_dir}\n"
            "Hint: Run train_cross_temporal_multiproject.py with --run-rf first."
        )
    return result


def classify_activity_patterns(
    period_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Classify reviewers into activity pattern groups.

    Only reviewers present in ALL available periods are classified.

    Returns:
        DataFrame with columns: reviewer_email, activity_pattern,
        irl_accuracy, rf_accuracy, period_count.
    """
    available_periods = sorted(period_data.keys())

    # Find common reviewers
    email_sets = [set(df["reviewer_email"]) for df in period_data.values()]
    common_emails = set.intersection(*email_sets)

    if not common_emails:
        raise ValueError("No reviewers found in all periods")

    rows: List[Dict] = []
    for email in common_emails:
        labels = []
        irl_correct_count = 0
        rf_correct_count = 0
        total = 0

        for period in available_periods:
            df = period_data[period]
            row = df[df["reviewer_email"] == email]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            labels.append(int(row["true_label"]))

            irl_ok = int(row["predicted_binary"]) == int(row["true_label"])
            rf_ok = int(row["rf_predicted_binary"]) == int(row["true_label"])
            irl_correct_count += int(irl_ok)
            rf_correct_count += int(rf_ok)
            total += 1

        if total == 0:
            continue

        # Classify
        unique_labels = set(labels)
        if unique_labels == {1}:
            pattern = PATTERN_CONSISTENT_ACCEPT
        elif unique_labels == {0}:
            pattern = PATTERN_CONSISTENT_REJECT
        else:
            pattern = PATTERN_VARIABLE

        rows.append({
            "reviewer_email": email,
            "activity_pattern": pattern,
            "irl_accuracy": irl_correct_count / total,
            "rf_accuracy": rf_correct_count / total,
            "period_count": total,
        })

    return pd.DataFrame(rows)


def plot_activity_pattern_accuracy(
    classified: pd.DataFrame,
    out_path: pathlib.Path,
) -> None:
    """Fig 6.5: Grouped bar chart of IRL/RF accuracy by activity pattern."""
    summary = []
    for pattern in PATTERN_ORDER:
        sub = classified[classified["activity_pattern"] == pattern]
        if len(sub) == 0:
            continue
        summary.append({
            "pattern": pattern,
            "IRL": sub["irl_accuracy"].mean(),
            "RF": sub["rf_accuracy"].mean(),
            "count": len(sub),
        })

    if not summary:
        print("No data available for activity pattern plot")
        return

    summary_df = pd.DataFrame(summary)
    # Convert to percentage
    summary_df["IRL"] = summary_df["IRL"] * 100
    summary_df["RF"] = summary_df["RF"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, summary_df["IRL"], width, label="IRL", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, summary_df["RF"], width, label="RF", color="#ff7f0e")

    # Add value labels on bars (percentage)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Add count labels below pattern names
    labels = [
        f"{row['pattern']}\n(n={row['count']})" for _, row in summary_df.iterrows()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("正解率 (%)")
    ax.set_title("図6.5: 活動パターン分類別のIRL/RF正解率")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot activity pattern classification accuracy (Fig 6.5)"
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Base directory containing train_*/eval_*/ results",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory for output PNG/PDF files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load diagonal pattern data
    period_data = load_diagonal_predictions_with_rf(args.input_dir)

    # Classify reviewers
    classified = classify_activity_patterns(period_data)

    # Print summary
    print(f"Classified {len(classified)} reviewers present in all {len(period_data)} periods:")
    for pattern in PATTERN_ORDER:
        count = (classified["activity_pattern"] == pattern).sum()
        print(f"  {pattern}: {count}")

    # Fig 6.5
    plot_activity_pattern_accuracy(
        classified, args.output_dir / "fig6_5_activity_pattern_accuracy"
    )


if __name__ == "__main__":
    main()
