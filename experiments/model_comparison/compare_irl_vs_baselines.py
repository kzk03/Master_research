"""IRL_Dir (LSTM baseline) vs ベースライン群の比較分析.

同期間評価・クロス時間評価の両方で、分類性能(AUC-ROC, AUC-PR, F1)と
ランキング性能(Spearman)を比較する。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "font.family": "serif",
})

BASE = Path("outputs/variant_comparison_server/lstm_baseline")
PERIODS = ["0-3", "3-6", "6-9", "9-12"]
METHODS_CLF = ["RF", "IRL_Dir", "RF_Dir"]  # clf指標を持つ手法
METHODS_ALL = ["Naive", "Linear", "RF", "IRL_Dir", "RF_Dir"]
CLF_METRICS = ["clf_auc_roc", "clf_auc_pr", "clf_f1", "clf_precision", "clf_recall"]
RANK_METRICS = ["spearman_r", "pearson_r"]
OUT_DIR = Path("experiments/model_comparison/figures")


def load_summary(train_period: str, eval_period: str) -> dict:
    p = BASE / f"train_{train_period}m" / f"eval_{eval_period}m" / "summary_metrics.json"
    with open(p) as f:
        return json.load(f)


def build_table() -> pd.DataFrame:
    """全 (train, eval) ペアの指標を収集."""
    rows = []
    for tp in PERIODS:
        train_dir = BASE / f"train_{tp}m"
        for eval_dir in sorted(train_dir.glob("eval_*m")):
            ep = eval_dir.name.replace("eval_", "").replace("m", "")
            data = load_summary(tp, ep)
            is_same = tp == ep
            for method in METHODS_ALL:
                if method not in data:
                    continue
                m = data[method]
                row = {
                    "train": tp,
                    "eval": ep,
                    "same_period": is_same,
                    "method": method,
                    "spearman_r": m.get("spearman_r"),
                    "pearson_r": m.get("pearson_r"),
                    "mae": m.get("mae"),
                }
                for ck in CLF_METRICS:
                    row[ck] = m.get(ck)
                rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    """コンソールに要約表を出力."""
    # --- 同期間評価 ---
    same = df[df["same_period"]]
    print("=" * 70)
    print("【同期間評価 (train == eval)】")
    print("=" * 70)
    for metric in CLF_METRICS + RANK_METRICS:
        tbl = same.pivot_table(index="method", columns="train", values=metric)
        tbl = tbl.reindex(METHODS_ALL)
        tbl["mean"] = tbl.mean(axis=1)
        print(f"\n--- {metric} ---")
        print(tbl.to_string(float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "   -"))

    # --- クロス時間評価 ---
    cross = df[~df["same_period"]]
    print("\n" + "=" * 70)
    print("【クロス時間評価 (train ≠ eval)】平均")
    print("=" * 70)
    for metric in CLF_METRICS + RANK_METRICS:
        mean_tbl = cross.groupby("method")[metric].mean()
        mean_tbl = mean_tbl.reindex(METHODS_ALL)
        print(f"\n--- {metric} ---")
        print(mean_tbl.to_string(float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "   -"))

    # --- 全体平均 (提案手法 vs ベースライン) ---
    print("\n" + "=" * 70)
    print("【全評価ペア 平均 ± 標準偏差】")
    print("=" * 70)
    for metric in CLF_METRICS + RANK_METRICS:
        stats = df.groupby("method")[metric].agg(["mean", "std"])
        stats = stats.reindex(METHODS_ALL)
        print(f"\n--- {metric} ---")
        for m_name, row in stats.iterrows():
            if np.isnan(row["mean"]):
                print(f"  {m_name:12s}:    -")
            else:
                print(f"  {m_name:12s}: {row['mean']:.4f} ± {row['std']:.4f}")


def plot_comparison(df: pd.DataFrame):
    """バー+エラーバー図で比較."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # clf指標 + spearman の4パネル
    metrics = ["clf_auc_roc", "clf_auc_pr", "clf_f1", "spearman_r"]
    labels = ["AUC-ROC", "AUC-PR", "F1", "Spearman ρ"]
    methods_plot = METHODS_CLF  # clf指標があるもの
    colors = {"RF": "#999999", "IRL_Dir": "#2196F3", "RF_Dir": "#FF9800"}

    # --- 図1: 同期間 vs クロス時間 ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric, label in zip(axes, metrics, labels):
        if metric == "spearman_r":
            methods_here = METHODS_ALL
        else:
            methods_here = methods_plot

        same = df[df["same_period"]]
        cross = df[~df["same_period"]]

        x = np.arange(len(methods_here))
        w = 0.35
        same_means = [same[same["method"] == m][metric].mean() for m in methods_here]
        same_stds = [same[same["method"] == m][metric].std() for m in methods_here]
        cross_means = [cross[cross["method"] == m][metric].mean() for m in methods_here]
        cross_stds = [cross[cross["method"] == m][metric].std() for m in methods_here]

        bars1 = ax.bar(x - w / 2, same_means, w, yerr=same_stds, label="Same period",
                       color="#2196F3", alpha=0.8, capsize=3)
        bars2 = ax.bar(x + w / 2, cross_means, w, yerr=cross_stds, label="Cross-temporal",
                       color="#FF9800", alpha=0.8, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(methods_here, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontsize=12)
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("IRL_Dir (LSTM) vs Baselines: Same-period vs Cross-temporal", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "irl_vs_baselines_overview.pdf", format="pdf")
    print(f"\nSaved: {OUT_DIR / 'irl_vs_baselines_overview.pdf'}")

    # --- 図2: 手法別 clf_auc_roc の時間推移 ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    same = df[df["same_period"]]
    for method in methods_plot:
        sub = same[same["method"] == method].sort_values("train")
        ax2.plot(sub["train"] + "m", sub["clf_auc_roc"], "o-", label=method,
                 color=colors[method], linewidth=2, markersize=7)
    ax2.set_xlabel("Training/Evaluation Period")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Classification AUC-ROC by Period (Same-period Evaluation)")
    ax2.legend()
    ax2.set_ylim(0.5, 0.9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "auc_roc_by_period.pdf", format="pdf")
    print(f"Saved: {OUT_DIR / 'auc_roc_by_period.pdf'}")

    # --- 図3: IRL_Dir の改善幅 (vs RF, vs RF_Dir) ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5))
    for ax, baseline, title in zip(axes3, ["RF", "RF_Dir"],
                                   ["IRL_Dir − RF (global)", "IRL_Dir − RF_Dir"]):
        improvements = []
        for metric in ["clf_auc_roc", "clf_auc_pr", "clf_f1", "spearman_r"]:
            irl_vals = df[df["method"] == "IRL_Dir"][metric].values
            base_vals = df[df["method"] == baseline][metric].values
            # pair-wise差分
            diff = irl_vals - base_vals
            improvements.append(diff)

        bp = ax.boxplot(improvements, labels=["AUC-ROC", "AUC-PR", "F1", "Spearman"],
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#2196F3")
            patch.set_alpha(0.6)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Improvement")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    fig3.suptitle("IRL_Dir Improvement over Baselines (all eval pairs)", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "irl_improvement.pdf", format="pdf")
    print(f"Saved: {OUT_DIR / 'irl_improvement.pdf'}")

    plt.close("all")


def plot_heatmaps(df: pd.DataFrame):
    """手法別 Train Window × Eval Window 三角行列ヒートマップ（指標ごと）."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = METHODS_CLF  # RF, IRL_Dir, RF_Dir
    method_labels = {"RF": "RF (global)", "IRL_Dir": "IRL_Dir (LSTM)", "RF_Dir": "RF_Dir"}
    period_labels = [f"{p}m" for p in PERIODS]

    metrics_to_plot = [
        ("clf_auc_roc", "AUC-ROC"),
        ("clf_auc_pr", "AUC-PR"),
        ("clf_f1", "F1"),
        ("clf_precision", "Precision"),
        ("clf_recall", "Recall"),
        ("spearman_r", "Spearman ρ"),
    ]

    for metric, metric_label in metrics_to_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 全手法の値を集めてカラー範囲を統一
        all_vals = []
        for method in methods:
            sub = df[df["method"] == method]
            for _, row in sub.iterrows():
                v = row[metric]
                if not np.isnan(v):
                    all_vals.append(v)
        vmin = max(0, min(all_vals) - 0.03)
        vmax = min(1, max(all_vals) + 0.02)

        for ax, method in zip(axes, methods):
            # 4x4 行列を作る (train x eval), NaN で上三角をマスク
            matrix = np.full((4, 4), np.nan)
            sub = df[df["method"] == method]
            for _, row in sub.iterrows():
                ti = PERIODS.index(row["train"])
                ei = PERIODS.index(row["eval"])
                matrix[ti, ei] = row[metric]

            # マスク: train > eval の部分（下三角）は存在しないのでそのまま NaN
            # imshow 用にマスク
            masked = np.ma.masked_invalid(matrix)

            cmap = plt.cm.Blues.copy()
            cmap.set_bad(color="white")

            im = ax.imshow(masked, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

            # セル内にテキスト
            for i in range(4):
                for j in range(4):
                    val = matrix[i, j]
                    if np.isnan(val):
                        continue
                    # 値が高い（色が濃い）ときは白文字
                    threshold = vmin + (vmax - vmin) * 0.6
                    color = "white" if val > threshold else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=12, color=color, fontweight="bold")

            ax.set_xticks(range(4))
            ax.set_xticklabels(period_labels, fontsize=11)
            ax.set_yticks(range(4))
            ax.set_yticklabels(period_labels, fontsize=11)
            ax.set_xlabel("Eval Window", fontsize=12)
            ax.set_ylabel("Train Window", fontsize=12)
            ax.set_title(method_labels[method], fontsize=14, fontweight="bold")

        # 共通カラーバー
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(metric_label, fontsize=12)

        fig.suptitle(f"Cross-Temporal Evaluation: {metric_label}",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.93, 0.93])
        fname = f"heatmap_{metric.replace('clf_', '')}.pdf"
        fig.savefig(OUT_DIR / fname, format="pdf")
        print(f"Saved: {OUT_DIR / fname}")
        plt.close(fig)


def main():
    df = build_table()
    print_summary(df)
    plot_comparison(df)
    plot_heatmaps(df)

    # CSVも保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "comparison_all_pairs.csv", index=False)
    print(f"Saved: {OUT_DIR / 'comparison_all_pairs.csv'}")


if __name__ == "__main__":
    main()
