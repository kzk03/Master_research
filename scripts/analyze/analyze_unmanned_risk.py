"""
無人化リスク分析スクリプト

variant_comparison の評価結果から、ディレクトリごとの将来貢献者数推移を分析し、
無人化リスク（貢献者数 ≤ 閾値）を検出・可視化する。

使い方:
    python scripts/analyze/analyze_unmanned_risk.py \
        --input-dir outputs/variant_comparison_server \
        --variant lstm_baseline \
        --data data/combined_raw.csv \
        --raw-json data/raw_json/openstack__*.json \
        --output-dir outputs/unmanned_risk_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from review_predictor.IRL.features.directory_contributors import (
    count_actual_contributors,
    get_directory_developers,
)
from review_predictor.IRL.features.path_features import (
    attach_dirs_to_df,
    load_change_dir_map,
    load_change_dir_map_multi,
)

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EVAL_CUTOFF = "2023-01-01"
WINDOWS = ["0-3", "3-6", "6-9", "9-12"]
WINDOW_OFFSETS = [(0, 3), (3, 6), (6, 9), (9, 12)]


# ── データ読み込み ──────────────────────────────────────────────


def load_eval_results(
    input_dir: Path,
    variant: str,
    use_calibrated: bool = False,
) -> pd.DataFrame:
    """対角パターン (train_W/eval_W) の pair_predictions.csv を統合読み込み。"""
    frames = []
    for win in WINDOWS:
        csv_path = input_dir / variant / f"train_{win}m" / f"eval_{win}m" / "pair_predictions.csv"
        if not csv_path.exists():
            logger.warning(f"見つからない: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["eval_window"] = win
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"pair_predictions.csv が見つからない: {input_dir / variant}")

    combined = pd.concat(frames, ignore_index=True)

    # 使用する確率列を決定
    prob_col = "irl_dir_prob"
    if use_calibrated and "irl_dir_prob_calibrated" in combined.columns:
        valid = combined["irl_dir_prob_calibrated"].notna().sum()
        if valid > 0:
            prob_col = "irl_dir_prob_calibrated"
            logger.info(f"校正済み確率を使用: {prob_col}")
        else:
            logger.warning("irl_dir_prob_calibrated が全 NaN → 生確率を使用")
    combined["prob"] = combined[prob_col]

    logger.info(
        f"読み込み完了: {len(combined)} 行, {combined['eval_window'].nunique()} 窓, "
        f"prob列={prob_col}"
    )
    return combined


def compute_ground_truth(
    df: pd.DataFrame,
    eval_cutoff: datetime,
) -> Dict[str, Dict[str, int]]:
    """各将来窓の ground truth (ディレクトリ別実貢献者数) を計算。"""
    result = {}
    for win, (start_m, end_m) in zip(WINDOWS, WINDOW_OFFSETS):
        t_start = eval_cutoff + pd.DateOffset(months=start_m)
        t_end = eval_cutoff + pd.DateOffset(months=end_m)
        actual = count_actual_contributors(df, t_start, t_end)
        result[win] = actual
        logger.info(f"Ground truth [{win}m]: {len(actual)} dirs")
    return result


# ── 分析1: 貢献者数推移 ──────────────────────────────────────────


def build_contributor_trajectories(
    eval_df: pd.DataFrame,
    ground_truth: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """ディレクトリ × 窓 ごとの予測・実績貢献者数を構築。"""
    rows = []
    for win in WINDOWS:
        win_df = eval_df[eval_df["eval_window"] == win]
        if win_df.empty:
            continue
        # 予測: ディレクトリ別Σprob
        pred_per_dir = win_df.groupby("directory")["prob"].sum()
        actual = ground_truth.get(win, {})

        all_dirs = set(pred_per_dir.index) | set(actual.keys())
        for d in all_dirs:
            rows.append({
                "directory": d,
                "eval_window": win,
                "predicted_count": pred_per_dir.get(d, 0.0),
                "actual_count": actual.get(d, 0),
                "n_past_devs": int(win_df[win_df["directory"] == d]["developer"].nunique())
                    if d in set(win_df["directory"]) else 0,
            })

    traj_df = pd.DataFrame(rows)
    traj_df["delta"] = traj_df["predicted_count"] - traj_df["actual_count"]
    return traj_df


# ── 分析2: リスクディレクトリ特定 ──────────────────────────────


def identify_at_risk_directories(
    traj_df: pd.DataFrame,
    threshold: int = 1,
) -> pd.DataFrame:
    """予測貢献者数 ≤ threshold のディレクトリを抽出し、リスクレベルを付与。"""
    # ディレクトリごとに窓を横並び
    pivot = traj_df.pivot_table(
        index="directory",
        columns="eval_window",
        values="predicted_count",
        aggfunc="first",
    ).reindex(columns=WINDOWS)

    actual_pivot = traj_df.pivot_table(
        index="directory",
        columns="eval_window",
        values="actual_count",
        aggfunc="first",
    ).reindex(columns=WINDOWS)

    rows = []
    for d in pivot.index:
        preds = pivot.loc[d].values
        actuals = actual_pivot.loc[d].values if d in actual_pivot.index else [np.nan] * 4
        n_at_risk = sum(1 for p in preds if not np.isnan(p) and p <= threshold)

        # 単調減少チェック
        valid_preds = [p for p in preds if not np.isnan(p)]
        is_decreasing = (
            len(valid_preds) >= 2
            and all(a >= b for a, b in zip(valid_preds, valid_preds[1:]))
            and valid_preds[0] > valid_preds[-1]
        )

        if n_at_risk >= 3:
            risk_level = "high"
        elif n_at_risk >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        rows.append({
            "directory": d,
            "risk_level": risk_level,
            "n_at_risk_windows": n_at_risk,
            "is_decreasing": is_decreasing,
            **{f"pred_{w}": pivot.loc[d, w] for w in WINDOWS},
            **{f"actual_{w}": actuals[i] for i, w in enumerate(WINDOWS)},
        })

    risk_df = pd.DataFrame(rows)
    logger.info(
        f"リスク分類: high={sum(risk_df['risk_level']=='high')}, "
        f"medium={sum(risk_df['risk_level']=='medium')}, "
        f"low={sum(risk_df['risk_level']=='low')}"
    )
    return risk_df


# ── 分析3: Ground truth との突合 ──────────────────────────────


def validate_unmanned_predictions(
    risk_df: pd.DataFrame,
    threshold_pred: float = 1.0,
) -> Dict:
    """無人化予測の precision/recall/F1。"""
    results = {}
    for win in WINDOWS:
        pred_col = f"pred_{win}"
        actual_col = f"actual_{win}"
        valid = risk_df.dropna(subset=[pred_col, actual_col])
        if valid.empty:
            continue

        is_danger_pred = valid[pred_col] <= threshold_pred
        is_danger_actual = valid[actual_col] <= 1

        tp = int((is_danger_pred & is_danger_actual).sum())
        fp = int((is_danger_pred & ~is_danger_actual).sum())
        fn = int((~is_danger_pred & is_danger_actual).sum())
        tn = int((~is_danger_pred & ~is_danger_actual).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[win] = {
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_dirs": len(valid),
        }
        logger.info(f"[{win}m] P={prec:.3f} R={rec:.3f} F1={f1:.3f} (tp={tp} fp={fp} fn={fn})")

    return results


# ── 分析4: プロジェクト横断比較 ──────────────────────────────


def extract_project_from_directory(directory: str, dir_to_proj: dict = None) -> str:
    """ディレクトリパスからプロジェクト名を推定。"""
    if dir_to_proj and directory in dir_to_proj:
        # returns full project name like openstack/nova or just nova
        return dir_to_proj[directory].split('/')[-1]
        
    parts = directory.strip("/").split("/")
    return parts[0] if parts else "unknown"


def cross_project_comparison(
    risk_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """プロジェクトごとの at-risk ディレクトリ割合。"""
    # eval_df からプロジェクト情報を取得
    valid_eval = eval_df.dropna(subset=['dirs']).explode('dirs')
    # 各ディレクトリが最も多く出現するプロジェクトを抽出
    if not valid_eval.empty and 'project' in valid_eval.columns:
        dir_to_proj = valid_eval.groupby('dirs')['project'].apply(
            lambda x: x.mode()[0] if not x.empty else "unknown"
        ).to_dict()
    else:
        dir_to_proj = {}
        
    risk_df = risk_df.copy()
    risk_df["project"] = risk_df["directory"].apply(lambda d: extract_project_from_directory(d, dir_to_proj))

    rows = []
    for proj, grp in risk_df.groupby("project"):
        n_total = len(grp)
        n_high = sum(grp["risk_level"] == "high")
        n_medium = sum(grp["risk_level"] == "medium")
        n_low = sum(grp["risk_level"] == "low")
        rows.append({
            "project": proj,
            "n_dirs": n_total,
            "n_high_risk": n_high,
            "n_medium_risk": n_medium,
            "n_low_risk": n_low,
            "high_risk_rate": n_high / n_total if n_total > 0 else 0.0,
            "at_risk_rate": (n_high + n_medium) / n_total if n_total > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("at_risk_rate", ascending=False)


# ── 可視化 ──────────────────────────────────────────────────


def plot_contributor_trajectory_heatmap(
    traj_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 40,
):
    """ディレクトリ × 窓 の予測貢献者数ヒートマップ。"""
    pivot = traj_df.pivot_table(
        index="directory",
        columns="eval_window",
        values="predicted_count",
        aggfunc="first",
    ).reindex(columns=WINDOWS)

    # 最大値が大きいディレクトリを上位に
    pivot["max_pred"] = pivot.max(axis=1)
    pivot = pivot.sort_values("max_pred", ascending=False).head(top_n).drop(columns="max_pred")

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd_r",
        xticklabels=[f"{w}m" for w in WINDOWS],
        ax=ax, cbar_kws={"label": "Predicted contributors"},
    )
    ax.set_title("Predicted contributor count per directory × future window")
    ax.set_ylabel("Directory")
    ax.set_xlabel("Future window")
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"保存: {output_path}")


def plot_risk_classification(
    cross_df: pd.DataFrame,
    output_path: Path,
):
    """プロジェクト別リスク分布の積み上げ棒グラフ。"""
    cross_df = cross_df.sort_values("n_dirs", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(cross_df))
    bar_width = 0.6
    ax.bar(x, cross_df["n_high_risk"], bar_width, label="High risk", color="#d62728")
    ax.bar(x, cross_df["n_medium_risk"], bar_width,
           bottom=cross_df["n_high_risk"], label="Medium risk", color="#ff7f0e")
    ax.bar(x, cross_df["n_low_risk"], bar_width,
           bottom=cross_df["n_high_risk"] + cross_df["n_medium_risk"],
           label="Low risk", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(cross_df["project"], rotation=45, ha="right")
    ax.set_ylabel("Number of directories")
    ax.set_title("Directory risk classification by project")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"保存: {output_path}")


def plot_validation_scatter(
    risk_df: pd.DataFrame,
    output_path: Path,
    window: str = "0-3",
):
    """予測 vs 実績の散布図。"""
    pred_col = f"pred_{window}"
    actual_col = f"actual_{window}"
    valid = risk_df.dropna(subset=[pred_col, actual_col])
    if valid.empty:
        logger.warning(f"散布図データなし: {window}")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"high": "#d62728", "medium": "#ff7f0e", "low": "#2ca02c"}
    for level, color in colors.items():
        sub = valid[valid["risk_level"] == level]
        ax.scatter(sub[actual_col], sub[pred_col], c=color, alpha=0.5, label=level, s=20)

    max_val = max(valid[actual_col].max(), valid[pred_col].max()) + 1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="y=x")
    ax.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="risk threshold=1")
    ax.axvline(x=1.0, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel(f"Actual contributors ({window}m)")
    ax.set_ylabel(f"Predicted contributors ({window}m)")
    ax.set_title(f"Predicted vs Actual contributor count ({window}m)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"保存: {output_path}")


def plot_cross_project_risk_rates(
    cross_df: pd.DataFrame,
    output_path: Path,
):
    """プロジェクト別の at-risk 率。"""
    cross_df = cross_df.sort_values("at_risk_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(cross_df))
    bars = ax.barh(y_pos, cross_df["at_risk_rate"], color="#d62728", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cross_df["project"])
    ax.set_xlabel("At-risk directory rate")
    ax.set_title("Proportion of at-risk directories by project")

    for bar, rate in zip(bars, cross_df["at_risk_rate"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"保存: {output_path}")


# ── メイン ──────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="無人化リスク分析")
    p.add_argument("--input-dir", type=Path,
                   default=Path("outputs/variant_comparison_server"))
    p.add_argument("--variant", type=str, default="lstm_baseline")
    p.add_argument("--data", type=Path, default=Path("data/combined_raw.csv"))
    p.add_argument("--raw-json", type=str, nargs="+",
                   default=["data/raw_json/openstack__nova.json"])
    p.add_argument("--dir-depth", type=int, default=2)
    p.add_argument("--eval-cutoff", type=str, default=EVAL_CUTOFF)
    p.add_argument("--risk-threshold", type=int, default=1)
    p.add_argument("--use-calibrated", action="store_true")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/unmanned_risk_analysis"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    logger.info(f"データ読み込み: {args.data}")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ディレクトリマッピング
    if len(args.raw_json) == 1:
        cdm = load_change_dir_map(args.raw_json[0], depth=args.dir_depth)
    else:
        cdm = load_change_dir_map_multi(args.raw_json, depth=args.dir_depth)
    df = attach_dirs_to_df(df, cdm)

    eval_cutoff = pd.Timestamp(args.eval_cutoff)

    # 1. 評価結果の読み込み
    eval_df = load_eval_results(args.input_dir, args.variant, args.use_calibrated)

    # 2. Ground truth 計算
    ground_truth = compute_ground_truth(df, eval_cutoff)

    # 3. 貢献者数推移
    traj_df = build_contributor_trajectories(eval_df, ground_truth)
    traj_df.to_csv(out / "contributor_trajectories.csv", index=False)
    logger.info(f"保存: {out / 'contributor_trajectories.csv'} ({len(traj_df)} 行)")

    # 4. リスクディレクトリ特定
    risk_df = identify_at_risk_directories(traj_df, threshold=args.risk_threshold)
    risk_df.to_csv(out / "at_risk_directories.csv", index=False)
    logger.info(f"保存: {out / 'at_risk_directories.csv'} ({len(risk_df)} 行)")

    # 5. Ground truth との突合
    validation = validate_unmanned_predictions(risk_df, threshold_pred=args.risk_threshold)

    # 6. プロジェクト横断比較
    cross_df = cross_project_comparison(risk_df, df)
    cross_df.to_csv(out / "cross_project_summary.csv", index=False)
    logger.info(f"保存: {out / 'cross_project_summary.csv'}")

    # 7. サマリ保存
    summary = {
        "variant": args.variant,
        "use_calibrated": args.use_calibrated,
        "risk_threshold": args.risk_threshold,
        "n_directories": len(risk_df),
        "n_high_risk": int(sum(risk_df["risk_level"] == "high")),
        "n_medium_risk": int(sum(risk_df["risk_level"] == "medium")),
        "n_low_risk": int(sum(risk_df["risk_level"] == "low")),
        "n_decreasing": int(risk_df["is_decreasing"].sum()),
        "validation": validation,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 8. 可視化
    figures_dir = out / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_contributor_trajectory_heatmap(traj_df, figures_dir / "contributor_trajectory_heatmap.pdf")
    plot_risk_classification(cross_df, figures_dir / "risk_classification_bar.pdf")
    plot_validation_scatter(risk_df, figures_dir / "unmanned_validation.pdf", window="0-3")
    plot_cross_project_risk_rates(cross_df, figures_dir / "cross_project_risk.pdf")

    logger.info("完了")


if __name__ == "__main__":
    main()
