# -*- coding: utf-8 -*-
"""
Feature importance visualization across training periods for IRL and RF.

Reads per-pattern importance data produced by train_cross_temporal_multiproject.py
and generates line plots showing how each feature's importance changes over periods.

Usage:
    # IRL only
    uv run python scripts/analyze/plot_feature_importance.py \
        --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/feature_importance

    # RF only
    uv run python scripts/analyze/plot_feature_importance.py \
        --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/feature_importance \
        --model-type rf

    # Both IRL and RF
    uv run python scripts/analyze/plot_feature_importance.py \
        --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/feature_importance \
        --model-type both

Input files (per pattern directory train_<period>/eval_<period>/):
    IRL: irl_feature_importance.json  (flat dict: feature_name -> importance)
    RF:  rf_metrics.json              (contains "feature_importance" key)

Output:
    irl_feature_importance.{png,pdf}  - IRL importance line plot
    rf_feature_importance.{png,pdf}   - RF importance line plot
    comparison.{png,pdf}              - Side-by-side comparison (when --model-type both)
"""

import argparse
import json
import pathlib
from typing import Dict, List

import japanize_matplotlib  # noqa: F401  # 日本語フォント登録（副作用import）  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

# Feature name mapping (English -> Japanese for paper)
FEATURE_NAME_JA: Dict[str, str] = {
    # State features (10-dim)
    "experience_days": "経験日数",
    "total_changes": "総レビュー依頼数",
    "total_reviews": "総レビュー数",
    "recent_activity_frequency": "最近の活動頻度",
    "avg_activity_gap": "平均活動間隔",
    "activity_trend": "月次活動変化率",
    "collaboration_score": "協力スコア",
    "code_quality_score": "総承諾率",
    "recent_acceptance_rate": "最近の承諾率",
    "review_load": "レビュー負荷",
    # Action features (4-dim)
    "avg_action_intensity": "レビューファイル数",
    "avg_collaboration": "協力度",
    "avg_response_time": "応答速度",
    "avg_review_size": "レビュー規模（行数）",
    # IRL action features (gradient-based names)
    "intensity": "レビューファイル数",
    "collaboration": "協力度",
    "response_speed": "応答速度",
    "review_size": "レビュー規模（行数）",
    "is_cross_project": "クロスプロジェクト",
    # IRL state features (multi-project)
    "project_count": "プロジェクト数",
    "project_activity_distribution": "活動分散度",
    "main_project_contribution_ratio": "メイン貢献率",
    "cross_project_collaboration_score": "横断協力スコア",
}

# Fixed colors per Japanese feature name for consistency across plots
FEATURE_COLORS: Dict[str, str] = {
    "経験日数": "#1f77b4",
    "総レビュー依頼数": "#ff7f0e",
    "総レビュー数": "#2ca02c",
    "最近の活動頻度": "#d62728",
    "平均活動間隔": "#9467bd",
    "月次活動変化率": "#8c564b",
    "協力スコア": "#e377c2",
    "総承諾率": "#7f7f7f",
    "最近の承諾率": "#bcbd22",
    "レビュー負荷": "#17becf",
    "レビューファイル数": "#aec7e8",
    "協力度": "#ffbb78",
    "応答速度": "#98df8a",
    "レビュー規模（行数）": "#ff9896",
    "クロスプロジェクト": "#c5b0d5",
    "プロジェクト数": "#c49c94",
    "活動分散度": "#f7b6d2",
    "メイン貢献率": "#c7c7c7",
    "横断協力スコア": "#dbdb8d",
}


def _to_ja(name: str) -> str:
    return FEATURE_NAME_JA.get(name, name)


def _load_irl_importances(base_dir: pathlib.Path) -> pd.DataFrame:
    """Load IRL gradient importance from diagonal patterns (train==eval).

    Tries two locations in order:
    1. train_{p}/eval_{p}/irl_feature_importance.json  (new format: flat dict)
    2. train_{p}/feature_importance/gradient_importance.json  (legacy: nested dict)
    """
    rows: List[pd.DataFrame] = []
    for period in PERIODS:
        # New format: train_X/eval_X/irl_feature_importance.json
        json_path = base_dir / f"train_{period}" / f"eval_{period}" / "irl_feature_importance.json"
        legacy = False
        # Legacy fallback: train_X/feature_importance/gradient_importance.json
        if not json_path.exists():
            json_path = base_dir / f"train_{period}" / "feature_importance" / "gradient_importance.json"
            legacy = True
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        # Legacy format has nested state_importance / action_importance dicts
        if legacy and isinstance(data.get("state_importance"), dict):
            flat: dict = {}
            flat.update(data.get("state_importance", {}))
            flat.update(data.get("action_importance", {}))
            data = flat
        # Flat dict: feature_name -> importance
        df = pd.DataFrame(list(data.items()), columns=["feature", "importance"])
        df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
        df["period"] = period
        df["feature_ja"] = df["feature"].map(_to_ja)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No irl_feature_importance.json found in {base_dir}/train_*/eval_*/"
        )
    return pd.concat(rows, ignore_index=True)


def _load_rf_importances(base_dir: pathlib.Path) -> pd.DataFrame:
    """Load RF feature importance from diagonal patterns (train==eval)."""
    rows: List[pd.DataFrame] = []
    for period in PERIODS:
        json_path = base_dir / f"train_{period}" / f"eval_{period}" / "rf_metrics.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        fi = data.get("feature_importance", {})
        if not fi:
            continue
        df = pd.DataFrame(list(fi.items()), columns=["feature", "importance"])
        df["period"] = period
        df["feature_ja"] = df["feature"].map(_to_ja)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No rf_metrics.json with feature_importance found in {base_dir}/train_*/eval_*/"
        )
    return pd.concat(rows, ignore_index=True)


def _plot_lines(
    df: pd.DataFrame,
    out_path: pathlib.Path,
    title: str,
    ylabel: str = "Importance",
    top_n: int = 14,
) -> None:
    """Plot feature importance line chart across periods."""
    available_periods = [p for p in PERIODS if p in df["period"].values]
    if not available_periods:
        return

    pivot = df.pivot_table(index="feature_ja", columns="period", values="importance")
    # Sort by mean importance
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")
    if top_n:
        pivot = pivot.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    for feature, row in pivot.iterrows():
        color = FEATURE_COLORS.get(feature, "#333333")
        vals = [row.get(p, np.nan) for p in available_periods]
        ax.plot(available_periods, vals, marker="o", label=feature, color=color, linewidth=2)

    ax.set_xlabel("Training period")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_comparison(
    irl_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    out_path: pathlib.Path,
    top_n: int = 10,
) -> None:
    """Side-by-side IRL vs RF importance comparison."""
    available_periods = [
        p for p in PERIODS
        if p in irl_df["period"].values and p in rf_df["period"].values
    ]
    if not available_periods:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, df, title, ylabel in [
        (ax1, irl_df, "IRL (gradient-based)", "Gradient importance"),
        (ax2, rf_df, "RF (Gini importance)", "Gini importance"),
    ]:
        pivot = df.pivot_table(index="feature_ja", columns="period", values="importance")
        pivot["_mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")
        if top_n:
            pivot = pivot.head(top_n)

        for feature, row in pivot.iterrows():
            color = FEATURE_COLORS.get(feature, "#333333")
            vals = [row.get(p, np.nan) for p in available_periods]
            ax.plot(available_periods, vals, marker="o", label=feature, color=color, linewidth=2)

        ax.set_xlabel("Training period")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=8)

    fig.suptitle("Feature Importance: IRL vs RF", fontsize=14, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature importance across training periods (IRL / RF)"
    )
    parser.add_argument(
        "--input-dir", type=pathlib.Path, required=True,
        help="Base output directory containing train_*/eval_*/ results",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, required=True,
        help="Directory for output PNG/PDF files",
    )
    parser.add_argument(
        "--model-type", choices=["irl", "rf", "both"], default="both",
        help="Which model's importance to plot (default: both)",
    )
    parser.add_argument(
        "--top-n", type=int, default=14,
        help="Number of top features to display (default: 14 = all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    irl_df = None
    rf_df = None

    if args.model_type in ("irl", "both"):
        irl_df = _load_irl_importances(args.input_dir)
        _plot_lines(
            irl_df,
            args.output_dir / "irl_feature_importance",
            title="IRL: Feature Importance (gradient-based)",
            ylabel="Gradient importance",
            top_n=args.top_n,
        )

    if args.model_type in ("rf", "both"):
        rf_df = _load_rf_importances(args.input_dir)
        _plot_lines(
            rf_df,
            args.output_dir / "rf_feature_importance",
            title="RF: Feature Importance (Gini)",
            ylabel="Gini importance",
            top_n=args.top_n,
        )

    if args.model_type == "both" and irl_df is not None and rf_df is not None:
        _plot_comparison(
            irl_df, rf_df,
            args.output_dir / "comparison",
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
