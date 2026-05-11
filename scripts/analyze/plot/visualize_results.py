#!/usr/bin/env python3
"""
全バリアント結果の可視化
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from review_predictor.config import ROOT

BASE = ROOT / "outputs" / "variant_comparison_server"
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)
VARIANTS = ["lstm_baseline", "lstm_attention", "transformer"]
VARIANT_LABELS = {"lstm_baseline": "LSTM Baseline", "lstm_attention": "LSTM+Attention", "transformer": "Transformer"}
COLORS = {"lstm_baseline": "#2196F3", "lstm_attention": "#FF9800", "transformer": "#4CAF50", "RF_Dir": "#9C27B0", "RF": "#757575"}


def load_summary_metrics():
    records = []
    for variant in VARIANTS:
        for train_dir in sorted((BASE / variant).glob("train_*")):
            train_win = train_dir.name.replace("train_", "")
            for eval_dir in sorted(train_dir.glob("eval_*")):
                eval_win = eval_dir.name.replace("eval_", "")
                json_path = eval_dir / "summary_metrics.json"
                if not json_path.exists():
                    continue
                with open(json_path) as f:
                    data = json.load(f)
                for model_name, metrics in data.items():
                    rec = {"variant": variant, "train_window": train_win,
                           "eval_window": eval_win, "model": model_name}
                    rec.update(metrics)
                    records.append(rec)
    return pd.DataFrame(records)


def load_all_predictions():
    records = []
    for variant in VARIANTS:
        for train_dir in sorted((BASE / variant).glob("train_*")):
            train_win = train_dir.name.replace("train_", "")
            for eval_dir in sorted(train_dir.glob("eval_*")):
                eval_win = eval_dir.name.replace("eval_", "")
                csv_path = eval_dir / "pair_predictions.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                df["variant"] = variant
                df["train_window"] = train_win
                df["eval_window"] = eval_win
                records.append(df)
    return pd.concat(records, ignore_index=True)


def fig1_variant_comparison_bar(metrics_df):
    """バリアント別 AUC-ROC / PR-AUC / Spearman の棒グラフ"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics_list = [
        ("clf_auc_roc", "AUC-ROC (Pair Classification)"),
        ("clf_auc_pr", "PR-AUC (Pair Classification)"),
        ("spearman_r", "Spearman r (Directory Ranking)"),
    ]

    models = VARIANTS + ["RF_Dir"]
    model_labels = [VARIANT_LABELS.get(m, m) for m in models]

    for ax, (metric, title) in zip(axes, metrics_list):
        means = []
        stds = []
        colors = []
        for m in models:
            if m in VARIANTS:
                vals = metrics_df[(metrics_df["variant"] == m) & (metrics_df["model"] == "IRL_Dir")][metric].dropna()
            else:
                vals = metrics_df[metrics_df["model"] == m][metric].dropna()
                # RF_Dir values are repeated per variant, take unique
                vals = vals.drop_duplicates()
            means.append(vals.mean())
            stds.append(vals.std())
            colors.append(COLORS[m])

        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0.5, 0.9)
        ax.grid(axis="y", alpha=0.3)
        # 値を表示
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{mean:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "fig1_variant_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'fig1_variant_comparison.png'}")


def fig2_temporal_generalization(metrics_df):
    """時間的汎化: gap vs AUC-ROC"""
    fig, ax = plt.subplots(figsize=(8, 5))

    irl = metrics_df[metrics_df["model"] == "IRL_Dir"].copy()
    irl["gap"] = irl["eval_window"].str.split("-").str[0].astype(int) - \
                 irl["train_window"].str.split("-").str[0].astype(int)

    for v in VARIANTS:
        v_data = irl[irl["variant"] == v].groupby("gap")["clf_auc_roc"].mean()
        ax.plot(v_data.index, v_data.values, "o-", color=COLORS[v],
                label=VARIANT_LABELS[v], linewidth=2, markersize=8)

    # RF_Dir
    rf_dir = metrics_df[metrics_df["model"] == "RF_Dir"].copy()
    rf_dir["gap"] = rf_dir["eval_window"].str.split("-").str[0].astype(int) - \
                    rf_dir["train_window"].str.split("-").str[0].astype(int)
    rf_gap = rf_dir.drop_duplicates(subset=["train_window", "eval_window"]).groupby("gap")["clf_auc_roc"].mean()
    ax.plot(rf_gap.index, rf_gap.values, "s--", color=COLORS["RF_Dir"],
            label="RF_Dir", linewidth=2, markersize=8)

    ax.set_xlabel("Gap between train and eval window (months)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Temporal Generalization: AUC-ROC vs Prediction Horizon")
    ax.legend()
    ax.set_xticks([0, 3, 6, 9])
    ax.set_ylim(0.7, 0.86)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig2_temporal_generalization.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'fig2_temporal_generalization.png'}")


def fig3_heatmap(metrics_df):
    """train×eval のヒートマップ（baseline IRL_Dir）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    windows = ["0-3m", "3-6m", "6-9m", "9-12m"]
    irl = metrics_df[metrics_df["model"] == "IRL_Dir"]

    for ax, v in zip(axes, VARIANTS):
        v_data = irl[irl["variant"] == v]
        mat = np.full((4, 4), np.nan)
        for _, row in v_data.iterrows():
            ti = windows.index(row["train_window"])
            ei = windows.index(row["eval_window"])
            mat[ti, ei] = row["clf_auc_roc"]

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0.7, vmax=0.85, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(windows, fontsize=9)
        ax.set_yticklabels(windows, fontsize=9)
        ax.set_xlabel("Eval Window")
        ax.set_ylabel("Train Window")
        ax.set_title(VARIANT_LABELS[v])

        for i in range(4):
            for j in range(4):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=axes, shrink=0.8, label="AUC-ROC")
    plt.suptitle("Cross-Temporal Evaluation: AUC-ROC", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_heatmap_cross_temporal.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'fig3_heatmap_cross_temporal.png'}")


def fig4_directory_analysis(pred_df):
    """ディレクトリ別AUC散布図（IRL_Dir vs RF_Dir）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, v in zip(axes, VARIANTS):
        v_data = pred_df[pred_df["variant"] == v]
        dir_results = []
        for d, grp in v_data.groupby("directory"):
            if len(grp) < 50 or grp["label"].nunique() < 2:
                continue
            irl_valid = grp.dropna(subset=["irl_dir_prob"])
            rf_valid = grp.dropna(subset=["rf_dir_prob"])
            if len(irl_valid) < 20 or irl_valid["label"].nunique() < 2:
                continue
            if len(rf_valid) < 20 or rf_valid["label"].nunique() < 2:
                continue
            irl_auc = roc_auc_score(irl_valid["label"], irl_valid["irl_dir_prob"])
            rf_auc = roc_auc_score(rf_valid["label"], rf_valid["rf_dir_prob"])
            dir_results.append({"directory": d, "irl_auc": irl_auc, "rf_auc": rf_auc, "n": len(grp)})

        dir_df = pd.DataFrame(dir_results)
        ax.scatter(dir_df["rf_auc"], dir_df["irl_auc"], s=dir_df["n"] / 10,
                   alpha=0.6, color=COLORS[v], edgecolors="white", linewidth=0.5)
        ax.plot([0.3, 1], [0.3, 1], "k--", alpha=0.4)
        ax.set_xlabel("RF_Dir AUC-ROC")
        ax.set_ylabel("IRL_Dir AUC-ROC")
        ax.set_title(f"{VARIANT_LABELS[v]} (n={len(dir_df)} dirs)")
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.3)

        wins = (dir_df["irl_auc"] > dir_df["rf_auc"]).sum()
        ax.text(0.35, 0.92, f"IRL wins: {wins}/{len(dir_df)}", fontsize=10)

    plt.suptitle("Directory-Level AUC: IRL_Dir vs RF_Dir", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_directory_scatter.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'fig4_directory_scatter.png'}")


def fig5_developer_analysis(pred_df):
    """開発者別AUC: 活動量別ボックスプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # baseline のみで開発者別AUCを計算
    v_data = pred_df[pred_df["variant"] == "lstm_baseline"]
    dev_results = []
    for dev, grp in v_data.groupby("developer"):
        if len(grp) < 20 or grp["label"].nunique() < 2:
            continue
        irl_valid = grp.dropna(subset=["irl_dir_prob"])
        rf_valid = grp.dropna(subset=["rf_dir_prob"])
        if len(irl_valid) < 10 or irl_valid["label"].nunique() < 2:
            continue
        if len(rf_valid) < 10 or rf_valid["label"].nunique() < 2:
            continue
        irl_auc = roc_auc_score(irl_valid["label"], irl_valid["irl_dir_prob"])
        rf_auc = roc_auc_score(rf_valid["label"], rf_valid["rf_dir_prob"])
        dev_results.append({"developer": dev, "irl_auc": irl_auc, "rf_auc": rf_auc,
                           "n_pairs": len(grp), "advantage": irl_auc - rf_auc})

    dev_df = pd.DataFrame(dev_results)
    dev_df["activity_bin"] = pd.cut(dev_df["n_pairs"], bins=[0, 50, 100, 200, 9999],
                                     labels=["<50", "50-100", "100-200", "200+"])

    # ボックスプロット
    data_by_bin = [dev_df[dev_df["activity_bin"] == b]["advantage"].values
                   for b in ["<50", "50-100", "100-200", "200+"]]
    bp = ax.boxplot(data_by_bin, labels=["<50", "50-100", "100-200", "200+"],
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["lstm_baseline"])
        patch.set_alpha(0.5)

    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Developer Activity (number of pairs)")
    ax.set_ylabel("IRL_Dir AUC - RF_Dir AUC")
    ax.set_title("LSTM Baseline: IRL Advantage by Developer Activity Level")
    ax.grid(axis="y", alpha=0.3)

    # 各ビンのサンプル数
    for i, b in enumerate(["<50", "50-100", "100-200", "200+"]):
        n = len(dev_df[dev_df["activity_bin"] == b])
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "fig5_developer_advantage_boxplot.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'fig5_developer_advantage_boxplot.png'}")


def fig6_calibration(pred_df):
    """キャリブレーションプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # IRL_Dir (baseline) のキャリブレーション
    for ax, (col, title) in zip(axes, [("irl_dir_prob", "IRL_Dir (LSTM Baseline)"),
                                        ("rf_dir_prob", "RF_Dir")]):
        if col == "irl_dir_prob":
            data = pred_df[pred_df["variant"] == "lstm_baseline"].dropna(subset=[col])
        else:
            data = pred_df[pred_df["variant"] == "lstm_baseline"].dropna(subset=[col])

        data = data.copy()
        data["bin"] = pd.cut(data[col], bins=10)
        cal = data.groupby("bin", observed=True).agg(
            mean_pred=(col, "mean"),
            actual=("label", "mean"),
            count=("label", "count"),
        ).reset_index()

        ax.plot(cal["mean_pred"], cal["actual"], "o-", color=COLORS["lstm_baseline"],
                linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Positive Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # バーで件数表示
        ax2 = ax.twinx()
        ax2.bar(cal["mean_pred"], cal["count"], width=0.06, alpha=0.15, color="gray")
        ax2.set_ylabel("Count", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    plt.suptitle("Calibration Plot: Predicted Probability vs Actual Rate")
    plt.tight_layout()
    plt.savefig(OUT / "fig6_calibration.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'fig6_calibration.png'}")


def fig7_score_distribution(pred_df):
    """バリアント別スコア分布"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, v in zip(axes, VARIANTS):
        v_data = pred_df[pred_df["variant"] == v]
        pos = v_data[v_data["label"] == 1]["irl_dir_prob"].dropna()
        neg = v_data[v_data["label"] == 0]["irl_dir_prob"].dropna()

        ax.hist(neg, bins=50, alpha=0.6, color="blue", label=f"Negative (n={len(neg)})", density=True)
        ax.hist(pos, bins=50, alpha=0.6, color="red", label=f"Positive (n={len(pos)})", density=True)
        ax.set_xlabel("IRL_Dir Predicted Score")
        ax.set_ylabel("Density")
        ax.set_title(VARIANT_LABELS[v])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Score Distribution by Label", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig7_score_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'fig7_score_distribution.png'}")


def main():
    print("Loading data...")
    metrics_df = load_summary_metrics()
    pred_df = load_all_predictions()
    print(f"  Metrics: {len(metrics_df)} rows, Predictions: {len(pred_df)} rows")
    print("\nGenerating figures...")

    fig1_variant_comparison_bar(metrics_df)
    fig2_temporal_generalization(metrics_df)
    fig3_heatmap(metrics_df)
    fig4_directory_analysis(pred_df)
    fig5_developer_analysis(pred_df)
    fig6_calibration(pred_df)
    fig7_score_distribution(pred_df)

    print(f"\nAll figures saved to: {OUT}")


if __name__ == "__main__":
    main()
