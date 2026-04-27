#!/usr/bin/env python3
"""
lstm_baseline の予測結果を深掘り分析するスクリプト

出力:
  - ディレクトリ別の予測精度
  - 開発者別の予測精度
  - IRL_Dir が RF に勝つ/負けるケースの特徴
  - 確信度 vs 正解率のキャリブレーション
  - 時間的汎化の詳細分析
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path("/Users/kazuki-h/Master_research/outputs/variant_comparison_server/lstm_baseline")


def load_all_predictions():
    """全評価パターンの pair_predictions.csv を読み込む"""
    records = []
    for train_dir in sorted(BASE.glob("train_*")):
        train_win = train_dir.name.replace("train_", "")
        for eval_dir in sorted(train_dir.glob("eval_*")):
            eval_win = eval_dir.name.replace("eval_", "")
            csv_path = eval_dir / "pair_predictions.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df["train_window"] = train_win
            df["eval_window"] = eval_win
            records.append(df)
    if not records:
        print("予測データが見つかりません")
        sys.exit(1)
    return pd.concat(records, ignore_index=True)


def analyze_by_directory(df):
    """ディレクトリ別の分析"""
    print("\n" + "=" * 80)
    print("  ディレクトリ別分析")
    print("=" * 80)

    # 各ディレクトリのペア数とラベル分布
    dir_stats = df.groupby("directory").agg(
        n_pairs=("label", "count"),
        n_pos=("label", "sum"),
        pos_rate=("label", "mean"),
        irl_mean=("irl_dir_prob", "mean"),
        rf_mean=("rf_dir_prob", "mean"),
    ).sort_values("n_pairs", ascending=False)

    print("\n### ペア数上位20ディレクトリ")
    print(dir_stats.head(20).to_string())

    # ディレクトリ別AUC（ペア数10以上かつ両ラベルありのみ）
    print("\n### ディレクトリ別 AUC-ROC（ペア数>=10, 両ラベルあり）")
    dir_aucs = []
    for d, grp in df.groupby("directory"):
        if len(grp) < 10 or grp["label"].nunique() < 2:
            continue
        irl_valid = grp.dropna(subset=["irl_dir_prob"])
        rf_valid = grp.dropna(subset=["rf_dir_prob"])
        irl_auc = roc_auc_score(irl_valid["label"], irl_valid["irl_dir_prob"]) if len(irl_valid) >= 10 and irl_valid["label"].nunique() == 2 else np.nan
        rf_auc = roc_auc_score(rf_valid["label"], rf_valid["rf_dir_prob"]) if len(rf_valid) >= 10 and rf_valid["label"].nunique() == 2 else np.nan
        dir_aucs.append({
            "directory": d,
            "n_pairs": len(grp),
            "pos_rate": grp["label"].mean(),
            "irl_auc": irl_auc,
            "rf_auc": rf_auc,
            "irl_advantage": (irl_auc - rf_auc) if not (np.isnan(irl_auc) or np.isnan(rf_auc)) else np.nan,
        })
    dir_auc_df = pd.DataFrame(dir_aucs).sort_values("irl_advantage", ascending=False)

    print("\n  IRL_Dir が RF_Dir に最も勝っているディレクトリ TOP10:")
    print(dir_auc_df.head(10).to_string(index=False))
    print("\n  IRL_Dir が RF_Dir に最も負けているディレクトリ TOP10:")
    print(dir_auc_df.tail(10).to_string(index=False))

    # 勝ち負けの集計
    valid = dir_auc_df.dropna(subset=["irl_advantage"])
    irl_wins = (valid["irl_advantage"] > 0).sum()
    rf_wins = (valid["irl_advantage"] < 0).sum()
    ties = (valid["irl_advantage"] == 0).sum()
    print(f"\n  IRL_Dir勝ち: {irl_wins}, RF_Dir勝ち: {rf_wins}, 引分: {ties} （{len(valid)}ディレクトリ中）")

    return dir_auc_df


def analyze_by_developer(df):
    """開発者別の分析"""
    print("\n" + "=" * 80)
    print("  開発者別分析")
    print("=" * 80)

    dev_stats = df.groupby("developer").agg(
        n_pairs=("label", "count"),
        n_pos=("label", "sum"),
        pos_rate=("label", "mean"),
        irl_mean=("irl_dir_prob", "mean"),
        rf_mean=("rf_dir_prob", "mean"),
    ).sort_values("n_pairs", ascending=False)

    print("\n### ペア数上位20開発者")
    print(dev_stats.head(20).to_string())

    # 開発者別AUC
    print("\n### 開発者別 AUC-ROC（ペア数>=10, 両ラベルあり）")
    dev_aucs = []
    for d, grp in df.groupby("developer"):
        if len(grp) < 10 or grp["label"].nunique() < 2:
            continue
        irl_valid = grp.dropna(subset=["irl_dir_prob"])
        rf_valid = grp.dropna(subset=["rf_dir_prob"])
        irl_auc = roc_auc_score(irl_valid["label"], irl_valid["irl_dir_prob"]) if len(irl_valid) >= 10 and irl_valid["label"].nunique() == 2 else np.nan
        rf_auc = roc_auc_score(rf_valid["label"], rf_valid["rf_dir_prob"]) if len(rf_valid) >= 10 and rf_valid["label"].nunique() == 2 else np.nan
        dev_aucs.append({
            "developer": d,
            "n_pairs": len(grp),
            "pos_rate": grp["label"].mean(),
            "irl_auc": irl_auc,
            "rf_auc": rf_auc,
            "irl_advantage": (irl_auc - rf_auc) if not (np.isnan(irl_auc) or np.isnan(rf_auc)) else np.nan,
        })
    dev_auc_df = pd.DataFrame(dev_aucs).sort_values("irl_advantage", ascending=False)

    print("\n  IRL_Dir が RF_Dir に最も勝っている開発者 TOP10:")
    print(dev_auc_df.head(10).to_string(index=False))
    print("\n  IRL_Dir が RF_Dir に最も負けている開発者 TOP10:")
    print(dev_auc_df.tail(10).to_string(index=False))

    valid = dev_auc_df.dropna(subset=["irl_advantage"])
    irl_wins = (valid["irl_advantage"] > 0).sum()
    rf_wins = (valid["irl_advantage"] < 0).sum()
    print(f"\n  IRL_Dir勝ち: {irl_wins}, RF_Dir勝ち: {rf_wins} （{len(valid)}開発者中）")

    # 活動量別の分析
    print("\n### 開発者の活動量（ペア数）とIRL優位性の関係")
    valid_sorted = valid.sort_values("n_pairs")
    for label, lo, hi in [("少 (10-30)", 10, 30), ("中 (30-60)", 30, 60), ("多 (60+)", 60, 9999)]:
        subset = valid_sorted[(valid_sorted["n_pairs"] >= lo) & (valid_sorted["n_pairs"] < hi)]
        if len(subset) == 0:
            continue
        mean_adv = subset["irl_advantage"].mean()
        irl_w = (subset["irl_advantage"] > 0).sum()
        print(f"  {label}: {len(subset)}人, IRL平均優位 {mean_adv:+.3f}, IRL勝ち {irl_w}/{len(subset)}")

    return dev_auc_df


def analyze_calibration(df):
    """キャリブレーション分析（確信度 vs 実際の正解率）"""
    print("\n" + "=" * 80)
    print("  キャリブレーション分析（確信度 vs 実際の正解率）")
    print("=" * 80)

    for model_name, col in [("IRL_Dir", "irl_dir_prob"), ("RF_Dir", "rf_dir_prob")]:
        valid = df.dropna(subset=[col])
        if len(valid) == 0:
            continue
        print(f"\n### {model_name}")
        # 確信度を10分割
        valid = valid.copy()
        valid["bin"] = pd.cut(valid[col], bins=10)
        cal = valid.groupby("bin", observed=True).agg(
            mean_prob=(col, "mean"),
            actual_rate=("label", "mean"),
            count=("label", "count"),
        )
        print(cal.to_string())


def analyze_temporal_generalization(df):
    """時間的汎化の詳細分析"""
    print("\n" + "=" * 80)
    print("  時間的汎化分析")
    print("=" * 80)

    print("\n### train_window × eval_window 別 AUC-ROC")
    results = []
    for (tw, ew), grp in df.groupby(["train_window", "eval_window"]):
        irl_valid = grp.dropna(subset=["irl_dir_prob"])
        rf_valid = grp.dropna(subset=["rf_dir_prob"])
        if len(irl_valid) < 10 or irl_valid["label"].nunique() < 2:
            continue
        irl_auc = roc_auc_score(irl_valid["label"], irl_valid["irl_dir_prob"])
        rf_auc = roc_auc_score(rf_valid["label"], rf_valid["rf_dir_prob"])
        irl_ap = average_precision_score(irl_valid["label"], irl_valid["irl_dir_prob"])
        rf_ap = average_precision_score(rf_valid["label"], rf_valid["rf_dir_prob"])

        # train と eval の距離
        train_start = int(tw.split("-")[0])
        eval_start = int(ew.split("-")[0])
        gap = eval_start - train_start  # 0なら同一窓, 3なら隣接, etc.

        results.append({
            "train": tw,
            "eval": ew,
            "gap_months": gap,
            "irl_auc": irl_auc,
            "rf_auc": rf_auc,
            "irl_ap": irl_ap,
            "rf_ap": rf_ap,
            "advantage": irl_auc - rf_auc,
        })
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # gap別の平均
    print("\n### 訓練-評価間のギャップ別 平均AUC")
    gap_summary = res_df.groupby("gap_months").agg(
        n=("irl_auc", "count"),
        irl_auc_mean=("irl_auc", "mean"),
        rf_auc_mean=("rf_auc", "mean"),
        advantage_mean=("advantage", "mean"),
    )
    print(gap_summary.to_string())


def analyze_error_patterns(df):
    """IRL_Dir の誤分類パターン分析"""
    print("\n" + "=" * 80)
    print("  IRL_Dir 誤分類パターン分析")
    print("=" * 80)

    # train_0-3m/eval_0-3m の同一窓データで分析
    subset = df[(df["train_window"] == "0-3m") & (df["eval_window"] == "0-3m")].dropna(subset=["irl_dir_prob"])
    if len(subset) == 0:
        print("データなし")
        return

    # 閾値0.3で分類
    threshold = 0.3
    subset = subset.copy()
    subset["irl_pred"] = (subset["irl_dir_prob"] >= threshold).astype(int)
    subset["correct"] = (subset["irl_pred"] == subset["label"]).astype(int)

    tp = subset[(subset["label"] == 1) & (subset["irl_pred"] == 1)]
    fp = subset[(subset["label"] == 0) & (subset["irl_pred"] == 1)]
    fn = subset[(subset["label"] == 1) & (subset["irl_pred"] == 0)]
    tn = subset[(subset["label"] == 0) & (subset["irl_pred"] == 0)]

    print(f"\n  閾値={threshold} での混同行列: TP={len(tp)}, FP={len(fp)}, FN={len(fn)}, TN={len(tn)}")

    # False Positive（離脱したのに継続と予測）の特徴
    print(f"\n### False Positive（離脱を継続と誤予測）: {len(fp)}件")
    if len(fp) > 0:
        print("  確信度分布:")
        print(f"    mean={fp['irl_dir_prob'].mean():.3f}, median={fp['irl_dir_prob'].median():.3f}")
        print("  上位ディレクトリ:")
        print(fp["directory"].value_counts().head(10).to_string())

    # False Negative（継続しているのに離脱と予測）の特徴
    print(f"\n### False Negative（継続を離脱と誤予測）: {len(fn)}件")
    if len(fn) > 0:
        print("  確信度分布:")
        print(f"    mean={fn['irl_dir_prob'].mean():.3f}, median={fn['irl_dir_prob'].median():.3f}")
        print("  上位ディレクトリ:")
        print(fn["directory"].value_counts().head(10).to_string())

    # RF_Dir との不一致パターン
    subset["rf_pred"] = (subset["rf_dir_prob"] >= 0.2).astype(int)  # RF_Dir の閾値
    both_correct = ((subset["irl_pred"] == subset["label"]) & (subset["rf_pred"] == subset["label"])).sum()
    irl_only = ((subset["irl_pred"] == subset["label"]) & (subset["rf_pred"] != subset["label"])).sum()
    rf_only = ((subset["irl_pred"] != subset["label"]) & (subset["rf_pred"] == subset["label"])).sum()
    both_wrong = ((subset["irl_pred"] != subset["label"]) & (subset["rf_pred"] != subset["label"])).sum()
    print(f"\n### IRL_Dir vs RF_Dir 一致/不一致")
    print(f"  両方正解: {both_correct}, IRL_Dirのみ正解: {irl_only}, RF_Dirのみ正解: {rf_only}, 両方不正解: {both_wrong}")


def analyze_positive_rate_impact(df):
    """正例率とモデル性能の関係"""
    print("\n" + "=" * 80)
    print("  正例率（継続率）とモデル性能の関係")
    print("=" * 80)

    # eval_window別の正例率
    for ew, grp in df.groupby("eval_window"):
        # 1つのtrain_windowから
        sub = grp[grp["train_window"] == grp["train_window"].iloc[0]]
        pos_rate = sub["label"].mean()
        print(f"  eval_{ew}: 正例率={pos_rate:.3f} ({int(sub['label'].sum())}/{len(sub)})")


def main():
    print("=" * 80)
    print("  lstm_baseline 深掘り分析")
    print("=" * 80)

    df = load_all_predictions()
    print(f"\n総ペア数: {len(df)}, 評価パターン数: {df.groupby(['train_window', 'eval_window']).ngroups}")
    print(f"ユニーク開発者数: {df['developer'].nunique()}")
    print(f"ユニークディレクトリ数: {df['directory'].nunique()}")

    analyze_temporal_generalization(df)
    dir_auc_df = analyze_by_directory(df)
    dev_auc_df = analyze_by_developer(df)
    analyze_calibration(df)
    analyze_error_patterns(df)
    analyze_positive_rate_impact(df)

    print("\n" + "=" * 80)
    print("  分析完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
