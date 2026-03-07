# -*- coding: utf-8 -*-
"""
Unified heatmap script for IRL / RF cross-temporal evaluation matrices.

Usage:
    uv run python scripts/analyze/plot_heatmaps.py \
        --model-type irl --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/heatmaps

    uv run python scripts/analyze/plot_heatmaps.py \
        --model-type rf --input-dir outputs/cross_eval_results \
        --output-dir outputs/cross_eval_results/heatmaps

Input files (in --input-dir):
    matrix_AUC_ROC.csv, matrix_F1.csv, matrix_PRECISION.csv, matrix_RECALL.csv
    (for RF: rf_matrix_AUC_ROC.csv, rf_matrix_F1.csv, ...)

Output:
    heatmap_{METRIC}.{png,pdf}  (per metric)
    {MODEL}heatmap.{png,pdf}    (4-panel combined)
"""

import argparse
import pathlib
from typing import Dict

import japanize_matplotlib  # noqa: F401  # 日本語フォント登録（副作用import）  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METRIC_DISPLAY = {
    "AUC_ROC": "AUC-ROC",
    "F1": "F1",
    "PRECISION": "Precision",
    "RECALL": "Recall",
}

METRICS = ["AUC_ROC", "F1", "PRECISION", "RECALL"]

# CSV filename prefix per model type
_CSV_PREFIX = {
    "irl": "matrix",
    "rf": "rf_matrix",
}


def _csv_name(model_type: str, metric: str) -> str:
    return f"{_CSV_PREFIX[model_type]}_{metric}.csv"


def load_matrix(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = [c[5:] if isinstance(c, str) and c.startswith("eval_") else c for c in df.columns]
    df = df.replace({"": np.nan})
    return df.astype(float)


def compute_global_range(matrices: Dict[str, pd.DataFrame]) -> tuple[float, float]:
    values = []
    for df in matrices.values():
        values.append(df.values.flatten())
    arr = np.concatenate(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0, 1.0
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: pathlib.Path,
    vmin: float,
    vmax: float,
) -> None:
    plt.figure(figsize=(6, 5))
    df_t = df.T
    mask = df_t.isna()
    ax = sns.heatmap(
        df_t,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        mask=mask,
        cbar_kws={"label": title},
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Training period")
    ax.set_ylabel("Evaluation period")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_four_panel(
    matrices: Dict[str, pd.DataFrame],
    out_path: pathlib.Path,
    vmin: float,
    vmax: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    items = ["PRECISION", "RECALL", "F1", "AUC_ROC"]
    for ax, metric in zip(axes.flat, items):
        df = matrices[metric].T
        mask = df.isna()
        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            mask=mask,
            ax=ax,
            cbar_kws={"label": METRIC_DISPLAY[metric]},
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(METRIC_DISPLAY[metric])
        ax.set_xlabel("Training period")
        ax.set_ylabel("Evaluation period")
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-temporal heatmaps for IRL or RF")
    parser.add_argument("--model-type", choices=["irl", "rf"], required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True,
                        help="Directory containing matrix_*.csv files")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                        help="Directory for output PNG/PDF files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrices: Dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        csv_path = args.input_dir / _csv_name(args.model_type, metric)
        matrices[metric] = load_matrix(csv_path)

    vmin, vmax = compute_global_range(matrices)

    for metric, df in matrices.items():
        out_base = args.output_dir / f"heatmap_{metric}"
        plot_heatmap(df, METRIC_DISPLAY[metric], out_base, vmin, vmax)

    panel_name = f"{args.model_type.upper()}heatmap"
    plot_four_panel(matrices, args.output_dir / panel_name, vmin, vmax)


if __name__ == "__main__":
    main()
