# -*- coding: utf-8 -*-
"""
IRL/RF prediction comparison: Venn diagram and scatter plot (Fig 6.3 + 6.4).

Fig 6.3: Venn diagram showing overlap of IRL/RF correct predictions.
Fig 6.4: Scatter plot of request count x acceptance rate, colored by prediction category.

Usage:
    uv run python scripts/analyze/plot_prediction_comparison.py \
        --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/discussion
"""

import argparse
import pathlib
from typing import List

import japanize_matplotlib  # noqa: F401  # 日本語フォント登録（副作用import）  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

# Prediction category names
CAT_BOTH_CORRECT = "両方正解"
CAT_IRL_ONLY = "IRLのみ正解"
CAT_RF_ONLY = "RFのみ正解"
CAT_BOTH_WRONG = "両方不正解"

CATEGORY_COLORS = {
    CAT_BOTH_CORRECT: "#2ca02c",
    CAT_IRL_ONLY: "#1f77b4",
    CAT_RF_ONLY: "#ff7f0e",
    CAT_BOTH_WRONG: "#d62728",
}


def load_predictions_with_rf(input_dir: pathlib.Path) -> pd.DataFrame:
    """Load predictions.csv (containing RF columns) from train<=eval patterns.

    Only upper-triangular combinations (train period ≤ eval period) are loaded,
    matching the temporal ordering used in the thesis (N=452).
    Rows where RF prediction is missing (NaN) are dropped.
    """
    rows: List[pd.DataFrame] = []
    period_order = {p: i for i, p in enumerate(PERIODS)}

    for train_dir in sorted(input_dir.glob("train_*")):
        train_name = train_dir.name.replace("train_", "")
        if train_name not in period_order:
            continue
        for eval_dir in sorted(train_dir.glob("eval_*")):
            eval_name = eval_dir.name.replace("eval_", "")
            if eval_name not in period_order:
                continue
            # Skip train > eval (lower triangular)
            if period_order[train_name] > period_order[eval_name]:
                continue
            csv_path = eval_dir / "predictions.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if "rf_predicted_binary" not in df.columns:
                continue

            df = df.dropna(subset=["rf_predicted_binary"])
            if len(df) == 0:
                continue

            df["train_period"] = train_name
            df["eval_period"] = eval_name
            rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No predictions.csv with RF columns found under {input_dir}\n"
            "Hint: Run train_cross_temporal_multiproject.py with --run-rf first."
        )
    return pd.concat(rows, ignore_index=True)


def classify_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each prediction into one of 4 categories."""
    df = df.copy()
    irl_correct = df["predicted_binary"] == df["true_label"]
    rf_correct = df["rf_predicted_binary"] == df["true_label"]

    conditions = [
        irl_correct & rf_correct,
        irl_correct & ~rf_correct,
        ~irl_correct & rf_correct,
        ~irl_correct & ~rf_correct,
    ]
    choices = [CAT_BOTH_CORRECT, CAT_IRL_ONLY, CAT_RF_ONLY, CAT_BOTH_WRONG]
    df["category"] = np.select(conditions, choices, default=CAT_BOTH_WRONG)
    return df


def plot_venn_diagram(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Fig 6.3: Venn diagram of IRL/RF correct predictions."""
    irl_only = int((df["category"] == CAT_IRL_ONLY).sum())
    rf_only = int((df["category"] == CAT_RF_ONLY).sum())
    both_correct = int((df["category"] == CAT_BOTH_CORRECT).sum())
    both_wrong = int((df["category"] == CAT_BOTH_WRONG).sum())
    total = len(df)

    def _fmt(n: int) -> str:
        return f"{n}\n({n/total:.1%})"

    # Try matplotlib_venn, fall back to manual drawing
    try:
        from matplotlib_venn import venn2

        fig, ax = plt.subplots(figsize=(7, 6))
        v = venn2(
            subsets=(irl_only, rf_only, both_correct),
            set_labels=("IRL", "RF"),
            ax=ax,
        )
        # Overwrite default count labels with count + percentage
        if v.get_label_by_id("10"):
            v.get_label_by_id("10").set_text(_fmt(irl_only))
        if v.get_label_by_id("01"):
            v.get_label_by_id("01").set_text(_fmt(rf_only))
        if v.get_label_by_id("11"):
            v.get_label_by_id("11").set_text(_fmt(both_correct))
        if v.get_patch_by_id("10"):
            v.get_patch_by_id("10").set_color(CATEGORY_COLORS[CAT_IRL_ONLY])
            v.get_patch_by_id("10").set_alpha(0.6)
        if v.get_patch_by_id("01"):
            v.get_patch_by_id("01").set_color(CATEGORY_COLORS[CAT_RF_ONLY])
            v.get_patch_by_id("01").set_alpha(0.6)
        if v.get_patch_by_id("11"):
            v.get_patch_by_id("11").set_color(CATEGORY_COLORS[CAT_BOTH_CORRECT])
            v.get_patch_by_id("11").set_alpha(0.6)
        ax.set_title("図6.3: IRL/RF 予測結果のベン図")
        ax.text(
            0.5, -0.1,
            f"両方不正解: {both_wrong} ({both_wrong/total:.1%})  |  合計: {total}",
            transform=ax.transAxes, ha="center", fontsize=10,
        )
    except ImportError:
        # Fallback: manual circles
        fig, ax = plt.subplots(figsize=(7, 6))
        circle1 = plt.Circle((-0.2, 0), 0.5, alpha=0.4, color=CATEGORY_COLORS[CAT_IRL_ONLY], label="IRL")
        circle2 = plt.Circle((0.2, 0), 0.5, alpha=0.4, color=CATEGORY_COLORS[CAT_RF_ONLY], label="RF")
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal")

        ax.text(-0.4, 0, f"IRLのみ\n{_fmt(irl_only)}", ha="center", va="center", fontsize=11)
        ax.text(0.0, 0, f"両方\n{_fmt(both_correct)}", ha="center", va="center", fontsize=11)
        ax.text(0.4, 0, f"RFのみ\n{_fmt(rf_only)}", ha="center", va="center", fontsize=11)

        ax.set_title("図6.3: IRL/RF 予測結果のベン図")
        ax.text(
            0.5, -0.15,
            f"両方不正解: {both_wrong} ({both_wrong/total:.1%})  |  合計: {total}",
            transform=ax.transAxes, ha="center", fontsize=10,
        )
        ax.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.png')}")


def plot_scatter_by_label(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Fig 6.4: Scatter plot of request count x acceptance rate, colored by category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    label_names = {1: "承諾 (true_label=1)", 0: "非承諾 (true_label=0)"}

    for ax, label_val in zip(axes, [1, 0]):
        sub = df[df["true_label"] == label_val]
        for cat in [CAT_BOTH_CORRECT, CAT_IRL_ONLY, CAT_RF_ONLY, CAT_BOTH_WRONG]:
            mask = sub["category"] == cat
            if mask.sum() == 0:
                continue
            ax.scatter(
                sub.loc[mask, "eval_request_count"],
                sub.loc[mask, "history_acceptance_rate"],
                c=CATEGORY_COLORS[cat],
                label=f"{cat} ({mask.sum()})",
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        ax.set_xscale("log")
        ax.set_xlabel("レビュー依頼数 (対数)")
        ax.set_ylabel("承諾率")
        ax.set_title(label_names[label_val])
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, linestyle=":", alpha=0.4)

        # Median lines
        if len(sub) > 0:
            median_x = sub["eval_request_count"].median()
            median_y = sub["history_acceptance_rate"].median()
            ax.axvline(median_x, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.axhline(median_y, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    fig.suptitle("図6.4: ラベル別の依頼数×承諾率 散布図", fontsize=13)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot IRL/RF prediction comparison: Venn diagram (Fig 6.3) and scatter plot (Fig 6.4)"
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

    # Load predictions with RF columns
    merged = load_predictions_with_rf(args.input_dir)
    merged = classify_predictions(merged)

    # Fig 6.3: Venn diagram
    plot_venn_diagram(merged, args.output_dir / "fig6_3_venn_diagram")

    # Fig 6.4: Scatter plot
    plot_scatter_by_label(merged, args.output_dir / "fig6_4_scatter_by_label")


if __name__ == "__main__":
    main()
