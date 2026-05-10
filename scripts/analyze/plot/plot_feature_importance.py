# -*- coding: utf-8 -*-
"""
Feature importance visualization across training periods for IRL and RF.

Reads per-pattern importance data produced by train_cross_temporal_multiproject.py
and generates line plots showing how each feature's importance changes over periods.

Usage:
    # IRL only
    uv run python scripts/analyze/plot/plot_feature_importance.py \
        --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/feature_importance

    # RF only
    uv run python scripts/analyze/plot/plot_feature_importance.py \
        --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/feature_importance \
        --model-type rf

    # Both IRL and RF
    uv run python scripts/analyze/plot/plot_feature_importance.py \
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
from typing import Dict, List, Optional, Tuple

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

# ---------- Feature metadata ----------
# (English name, Japanese name, category)
FEATURE_META: List[Tuple[str, str, str]] = [
    # State features (20-dim)
    ("experience_days",            "経験日数",         "state"),
    ("total_changes",              "総レビュー依頼数", "state"),
    ("total_reviews",              "総レビュー数",     "state"),
    ("recent_activity_frequency",  "最近の活動頻度",   "state"),
    ("avg_activity_gap",           "平均活動間隔",     "state"),
    ("activity_trend",             "月次活動変化率",   "state"),
    ("collaboration_score",        "協力スコア",       "state"),
    ("code_quality_score",         "総承諾率",         "state"),
    ("recent_acceptance_rate",     "最近の承諾率",     "state"),
    ("review_load",                "レビュー負荷",     "state"),
    ("days_since_last_activity",   "最終活動経過日数", "state"),
    ("acceptance_trend",           "承諾率トレンド",   "state"),
    ("reciprocity_score",          "相互レビュー率",   "state"),
    ("load_trend",                 "負荷トレンド",     "state"),
    ("core_reviewer_ratio",        "コアレビュアー率", "state"),
    ("recent_rejection_streak",    "直近拒否連続数",   "state"),
    ("acceptance_rate_last10",     "直近10件承諾率",   "state"),
    ("active_months_ratio",        "活動月割合",       "state"),
    ("response_time_trend",        "応答速度トレンド", "state"),
    ("complex_pr_bias",            "複雑PRバイアス",   "state"),
    # Path features (3-dim)
    ("path_review_count",          "パスレビュー件数", "path"),
    ("path_recency",               "パス新しさ",       "path"),
    ("path_acceptance_rate",       "パス承諾率",       "path"),
    # Action features (5-dim)
    ("avg_action_intensity",       "行動強度",         "action"),
    ("avg_change_lines",           "平均変更行数",     "action"),
    ("avg_response_time",          "応答速度",         "action"),
    ("avg_review_size",            "レビュー規模",     "action"),
    ("repeat_collaboration_rate",  "リピート協力率",   "action"),
]

FEATURE_NAME_JA: Dict[str, str] = {en: ja for en, ja, _ in FEATURE_META}
FEATURE_CATEGORY: Dict[str, str] = {ja: cat for _, ja, cat in FEATURE_META}

# Legacy aliases
_LEGACY = {
    "avg_collaboration": "協力度", "intensity": "行動強度",
    "collaboration": "協力度", "response_speed": "応答速度",
    "review_size": "レビュー規模", "is_cross_project": "クロスプロジェクト",
    "project_count": "プロジェクト数",
    "project_activity_distribution": "活動分散度",
    "main_project_contribution_ratio": "メイン貢献率",
    "cross_project_collaboration_score": "横断協力スコア",
}
FEATURE_NAME_JA.update(_LEGACY)

# ---------- Color scheme by category ----------
# State: 青系 20色, Path: 緑〜紫系 3色, Action: 赤〜橙系 5色
_STATE_COLORS = [
    "#1f77b4", "#aec7e8", "#3182bd", "#6baed6", "#9ecae1",
    "#08519c", "#2171b5", "#4292c6", "#6aaed6", "#9cc8e2",
    "#084594", "#2166ac", "#4393c3", "#74add1", "#abd9e9",
    "#045a8d", "#0570b0", "#3690c0", "#74a9cf", "#a6bddb",
]
_PATH_COLORS = ["#2ca02c", "#98df8a", "#006d2c"]
_ACTION_COLORS = ["#d62728", "#ff7f0e", "#e6550d", "#fd8d3c", "#fdae6b"]

# Line styles to further differentiate features within a category
_STATE_LINESTYLES = ["-"] * 10 + ["--"] * 10
_PATH_LINESTYLES = ["-", "-", "-"]
_ACTION_LINESTYLES = ["-", "-", "-", "--", "--"]

_STATE_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P",
                  "o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]
_PATH_MARKERS = ["o", "s", "^"]
_ACTION_MARKERS = ["o", "s", "^", "D", "v"]

# Build per-feature style dicts
FEATURE_STYLE: Dict[str, dict] = {}
_si, _pi, _ai = 0, 0, 0
for _, ja, cat in FEATURE_META:
    if cat == "state":
        FEATURE_STYLE[ja] = {
            "color": _STATE_COLORS[_si], "linestyle": _STATE_LINESTYLES[_si],
            "marker": _STATE_MARKERS[_si], "linewidth": 1.5,
        }
        _si += 1
    elif cat == "path":
        FEATURE_STYLE[ja] = {
            "color": _PATH_COLORS[_pi], "linestyle": _PATH_LINESTYLES[_pi],
            "marker": _PATH_MARKERS[_pi], "linewidth": 2.5,
        }
        _pi += 1
    elif cat == "action":
        FEATURE_STYLE[ja] = {
            "color": _ACTION_COLORS[_ai], "linestyle": _ACTION_LINESTYLES[_ai],
            "marker": _ACTION_MARKERS[_ai], "linewidth": 1.5,
        }
        _ai += 1

CATEGORY_LABEL = {"state": "状態特徴量 (20)", "path": "パス特徴量 (3)", "action": "行動特徴量 (5)"}


def _to_ja(name: str) -> str:
    return FEATURE_NAME_JA.get(name, name)


# ---------- Data loaders ----------

def _load_irl_importances(base_dir: pathlib.Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for period in PERIODS:
        json_path = base_dir / f"train_{period}" / f"eval_{period}" / "irl_feature_importance.json"
        legacy = False
        if not json_path.exists():
            json_path = base_dir / f"train_{period}" / "feature_importance" / "gradient_importance.json"
            legacy = True
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        if legacy and isinstance(data.get("state_importance"), dict):
            flat: dict = {}
            flat.update(data.get("state_importance", {}))
            flat.update(data.get("action_importance", {}))
            data = flat
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


def _load_irl_signed(base_dir: pathlib.Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for period in PERIODS:
        json_path = base_dir / f"train_{period}" / f"eval_{period}" / "irl_feature_importance_signed.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        df = pd.DataFrame(list(data.items()), columns=["feature", "importance"])
        df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
        df["period"] = period
        df["feature_ja"] = df["feature"].map(_to_ja)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No irl_feature_importance_signed.json found in {base_dir}/train_*/eval_*/"
        )
    return pd.concat(rows, ignore_index=True)


# ---------- Plot functions ----------

def _plot_lines(
    df: pd.DataFrame,
    out_path: pathlib.Path,
    title: str,
    ylabel: str = "Importance",
    top_n: Optional[int] = None,
) -> None:
    """Plot feature importance line chart across periods with category-grouped legend."""
    available_periods = [p for p in PERIODS if p in df["period"].values]
    if not available_periods:
        return

    pivot = df.pivot_table(index="feature_ja", columns="period", values="importance")
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")
    if top_n:
        pivot = pivot.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot lines grouped by category
    handles_by_cat: Dict[str, list] = {"state": [], "path": [], "action": []}
    for feature, row in pivot.iterrows():
        style = FEATURE_STYLE.get(feature, {"color": "#333333", "linestyle": "-",
                                             "marker": "o", "linewidth": 1.5})
        vals = [row.get(p, np.nan) for p in available_periods]
        line, = ax.plot(available_periods, vals, label=feature,
                        color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], linewidth=style["linewidth"],
                        markersize=6)
        cat = FEATURE_CATEGORY.get(feature, "state")
        handles_by_cat[cat].append(line)

    ax.set_xlabel("Training period", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.5)

    if (pivot.values < 0).any():
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    # Category-grouped legend
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    for cat_key in ["state", "path", "action"]:
        if not handles_by_cat[cat_key]:
            continue
        # Category header (invisible line)
        header = Line2D([], [], color="none", label=CATEGORY_LABEL[cat_key])
        legend_handles.append(header)
        legend_labels.append(CATEGORY_LABEL[cat_key])
        for h in handles_by_cat[cat_key]:
            legend_handles.append(h)
            legend_labels.append(h.get_label())

    n_items = len(legend_labels)
    ncol = 2 if n_items > 16 else 1
    fontsize = 7.5 if n_items > 20 else 8.5
    ax.legend(handles=legend_handles, labels=legend_labels,
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=ncol, fontsize=fontsize, handlelength=2.5,
              columnspacing=1.0)

    fig.tight_layout(rect=(0, 0, 0.78 if ncol == 2 else 0.82, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(
    irl_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    out_path: pathlib.Path,
    top_n: Optional[int] = None,
) -> None:
    """Side-by-side IRL vs RF importance comparison."""
    available_periods = [
        p for p in PERIODS
        if p in irl_df["period"].values and p in rf_df["period"].values
    ]
    if not available_periods:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

    for ax, src_df, plot_title, ylabel in [
        (ax1, irl_df, "IRL (gradient-based)", "Gradient importance"),
        (ax2, rf_df, "RF (Gini importance)", "Gini importance"),
    ]:
        pivot = src_df.pivot_table(index="feature_ja", columns="period", values="importance")
        pivot["_mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")
        if top_n:
            pivot = pivot.head(top_n)

        handles_by_cat: Dict[str, list] = {"state": [], "path": [], "action": []}
        for feature, row in pivot.iterrows():
            style = FEATURE_STYLE.get(feature, {"color": "#333333", "linestyle": "-",
                                                 "marker": "o", "linewidth": 1.5})
            vals = [row.get(p, np.nan) for p in available_periods]
            line, = ax.plot(available_periods, vals, label=feature,
                            color=style["color"], linestyle=style["linestyle"],
                            marker=style["marker"], linewidth=style["linewidth"],
                            markersize=5)
            cat = FEATURE_CATEGORY.get(feature, "state")
            handles_by_cat[cat].append(line)

        ax.set_xlabel("Training period", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(plot_title, fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.5)

        from matplotlib.lines import Line2D
        legend_handles = []
        legend_labels = []
        for cat_key in ["state", "path", "action"]:
            if not handles_by_cat[cat_key]:
                continue
            header = Line2D([], [], color="none", label=CATEGORY_LABEL[cat_key])
            legend_handles.append(header)
            legend_labels.append(CATEGORY_LABEL[cat_key])
            for h in handles_by_cat[cat_key]:
                legend_handles.append(h)
                legend_labels.append(h.get_label())

        ax.legend(handles=legend_handles, labels=legend_labels,
                  loc="center left", bbox_to_anchor=(1.02, 0.5),
                  ncol=2, fontsize=7, handlelength=2.5, columnspacing=1.0)

    fig.suptitle("Feature Importance: IRL vs RF", fontsize=14, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.78, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------- Main ----------

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
        "--top-n", type=int, default=None,
        help="Number of top features to display (default: all 28)",
    )
    parser.add_argument(
        "--signed", action="store_true",
        help="Plot signed IRL importance (positive=continuation, negative=attrition)",
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

    if args.signed:
        try:
            signed_df = _load_irl_signed(args.input_dir)
            _plot_lines(
                signed_df,
                args.output_dir / "irl_feature_importance_signed",
                title="IRL: Signed Feature Importance (gradient-based)",
                ylabel="∂(continuation_prob)/∂(feature)",
                top_n=args.top_n,
            )
        except FileNotFoundError as e:
            print(f"符号付きデータなし: {e}"
                  "\nextract_feature_importance.py を --overwrite で再実行してください")


if __name__ == "__main__":
    main()
