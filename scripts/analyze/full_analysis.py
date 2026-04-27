#!/usr/bin/env python3
"""
全バリアント（lstm_baseline, lstm_attention, transformer）の包括的分析
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path("/Users/kazuki-h/Master_research/outputs/variant_comparison_server")
VARIANTS = ["lstm_baseline", "lstm_attention", "transformer"]


def load_summary_metrics():
    """全summary_metrics.jsonを読み込む"""
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
    """全pair_predictions.csvを読み込む"""
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


def section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)


def analyze_accuracy(metrics_df):
    """精度の観点での分析"""
    section("1. 精度分析: バリアント比較")

    # IRL_Dir と RF_Dir のみ抽出
    irl = metrics_df[metrics_df["model"] == "IRL_Dir"].copy()
    rf_dir = metrics_df[metrics_df["model"] == "RF_Dir"].copy()
    rf = metrics_df[metrics_df["model"] == "RF"].copy()

    # --- バリアント別 clf_auc_roc 平均 ---
    print("\n### 1.1 clf_auc_roc（ペア分類AUC）バリアント別平均")
    print("-" * 60)
    for v in VARIANTS:
        v_irl = irl[irl["variant"] == v]["clf_auc_roc"]
        v_rf_dir = rf_dir[rf_dir["variant"] == v]["clf_auc_roc"]
        v_rf = rf[rf["variant"] == v]["clf_auc_roc"]
        print(f"  {v:20s}: IRL_Dir={v_irl.mean():.4f} (±{v_irl.std():.4f})  "
              f"RF_Dir={v_rf_dir.mean():.4f}  RF={v_rf.mean():.4f}")

    # --- バリアント別 clf_auc_pr ---
    print("\n### 1.2 clf_auc_pr（PR-AUC）バリアント別平均")
    print("-" * 60)
    for v in VARIANTS:
        v_irl = irl[irl["variant"] == v]["clf_auc_pr"]
        v_rf_dir = rf_dir[rf_dir["variant"] == v]["clf_auc_pr"]
        v_rf = rf[rf["variant"] == v]["clf_auc_pr"]
        print(f"  {v:20s}: IRL_Dir={v_irl.mean():.4f} (±{v_irl.std():.4f})  "
              f"RF_Dir={v_rf_dir.mean():.4f}  RF={v_rf.mean():.4f}")

    # --- バリアント別 spearman_r ---
    print("\n### 1.3 spearman_r（ディレクトリ順位相関）バリアント別平均")
    print("-" * 60)
    for v in VARIANTS:
        v_irl = irl[irl["variant"] == v]["spearman_r"]
        v_rf_dir = rf_dir[rf_dir["variant"] == v]["spearman_r"]
        v_rf = rf[rf["variant"] == v]["spearman_r"]
        print(f"  {v:20s}: IRL_Dir={v_irl.mean():.4f} (±{v_irl.std():.4f})  "
              f"RF_Dir={v_rf_dir.mean():.4f}  RF={v_rf.mean():.4f}")

    # --- 全パターン詳細テーブル ---
    print("\n### 1.4 全パターン clf_auc_roc 詳細")
    print("-" * 90)
    print(f"{'train':>6} {'eval':>6} | {'baseline':>10} {'attention':>10} {'transformer':>12} | {'RF_Dir':>8} {'RF':>8}")
    print("-" * 90)

    # pivot
    for (tw, ew), grp in irl.groupby(["train_window", "eval_window"]):
        row = {}
        for v in VARIANTS:
            val = grp[grp["variant"] == v]["clf_auc_roc"].values
            row[v] = val[0] if len(val) > 0 else np.nan
        # RF_Dir は variant間で同じ（同じデータ）
        rf_d_val = rf_dir[(rf_dir["train_window"] == tw) & (rf_dir["eval_window"] == ew)]["clf_auc_roc"].values
        rf_val = rf[(rf["train_window"] == tw) & (rf["eval_window"] == ew)]["clf_auc_roc"].values
        rf_d = rf_d_val[0] if len(rf_d_val) > 0 else np.nan
        rf_v = rf_val[0] if len(rf_val) > 0 else np.nan
        best = max(row.values())
        print(f"{tw:>6} {ew:>6} | {row.get('lstm_baseline', np.nan):>10.4f} "
              f"{row.get('lstm_attention', np.nan):>10.4f} "
              f"{row.get('transformer', np.nan):>12.4f} | {rf_d:>8.4f} {rf_v:>8.4f}")

    # --- 時間的汎化 ---
    print("\n### 1.5 時間的汎化（gap別 clf_auc_roc平均）")
    print("-" * 60)
    irl_copy = irl.copy()
    irl_copy["gap"] = irl_copy["eval_window"].str.split("-").str[0].astype(int) - \
                      irl_copy["train_window"].str.split("-").str[0].astype(int)
    print(f"{'gap':>4} | {'baseline':>10} {'attention':>10} {'transformer':>12}")
    for gap, grp in irl_copy.groupby("gap"):
        row = {}
        for v in VARIANTS:
            row[v] = grp[grp["variant"] == v]["clf_auc_roc"].mean()
        print(f"{gap:>4} | {row.get('lstm_baseline', np.nan):>10.4f} "
              f"{row.get('lstm_attention', np.nan):>10.4f} "
              f"{row.get('transformer', np.nan):>12.4f}")


def analyze_by_directory(pred_df):
    """ディレクトリ単位の分析"""
    section("2. ディレクトリ単位分析")

    # 全バリアントをまたいでディレクトリ別AUCを計算
    print("\n### 2.1 ディレクトリ別 AUC-ROC（バリアント比較, ペア数>=20）")
    print("-" * 80)

    dir_results = []
    for d, grp in pred_df.groupby("directory"):
        if len(grp) < 60 or grp["label"].nunique() < 2:  # 各variant20以上
            continue
        row = {"directory": d, "n_pairs": len(grp), "pos_rate": grp["label"].mean()}
        for v in VARIANTS:
            v_grp = grp[grp["variant"] == v].dropna(subset=["irl_dir_prob"])
            if len(v_grp) >= 20 and v_grp["label"].nunique() == 2:
                row[f"irl_{v}"] = roc_auc_score(v_grp["label"], v_grp["irl_dir_prob"])
            else:
                row[f"irl_{v}"] = np.nan
        # RF_Dir（variantによらない）
        rf_grp = grp.dropna(subset=["rf_dir_prob"])
        if len(rf_grp) >= 20 and rf_grp["label"].nunique() == 2:
            row["rf_dir"] = roc_auc_score(rf_grp["label"], rf_grp["rf_dir_prob"])
        else:
            row["rf_dir"] = np.nan
        dir_results.append(row)

    dir_df = pd.DataFrame(dir_results)

    # ベスト IRL variant を決定
    dir_df["best_irl"] = dir_df[["irl_lstm_baseline", "irl_lstm_attention", "irl_transformer"]].max(axis=1)
    dir_df["best_variant"] = dir_df[["irl_lstm_baseline", "irl_lstm_attention", "irl_transformer"]].idxmax(axis=1).str.replace("irl_", "")
    dir_df["irl_vs_rf"] = dir_df["best_irl"] - dir_df["rf_dir"]

    # ペア数上位30ディレクトリ
    top_dirs = dir_df.sort_values("n_pairs", ascending=False).head(30)
    print(f"\n  ペア数上位30ディレクトリのAUC:")
    print(f"  {'directory':<30} {'n':>5} {'pos%':>5} {'baseline':>8} {'attention':>9} {'transform':>9} {'RF_Dir':>7} {'best':>12}")
    print("  " + "-" * 100)
    for _, r in top_dirs.iterrows():
        best_v = r["best_variant"] if pd.notna(r["best_variant"]) else "?"
        print(f"  {r['directory']:<30} {r['n_pairs']:>5.0f} {r['pos_rate']:>5.2f} "
              f"{r.get('irl_lstm_baseline', np.nan):>8.3f} "
              f"{r.get('irl_lstm_attention', np.nan):>9.3f} "
              f"{r.get('irl_transformer', np.nan):>9.3f} "
              f"{r.get('rf_dir', np.nan):>7.3f} {best_v:>12}")

    # バリアント間の勝率
    print("\n### 2.2 バリアント間勝率（ディレクトリ別AUCで比較）")
    valid = dir_df.dropna(subset=["irl_lstm_baseline", "irl_lstm_attention", "irl_transformer"])
    for v in VARIANTS:
        wins = (valid[f"irl_{v}"] == valid[["irl_lstm_baseline", "irl_lstm_attention", "irl_transformer"]].max(axis=1)).sum()
        print(f"  {v}: 最高AUCのディレクトリ数 {wins}/{len(valid)}")

    # IRL全体 vs RF_Dir
    valid2 = dir_df.dropna(subset=["best_irl", "rf_dir"])
    irl_wins = (valid2["irl_vs_rf"] > 0).sum()
    rf_wins = (valid2["irl_vs_rf"] < 0).sum()
    print(f"\n  IRL(best) vs RF_Dir: IRL勝ち {irl_wins}, RF勝ち {rf_wins} （{len(valid2)}ディレクトリ）")

    # 正例率別の分析
    print("\n### 2.3 正例率（継続率）別のモデル性能")
    for label, lo, hi in [("低 (0-30%)", 0, 0.3), ("中 (30-60%)", 0.3, 0.6), ("高 (60%+)", 0.6, 1.1)]:
        subset = dir_df[(dir_df["pos_rate"] >= lo) & (dir_df["pos_rate"] < hi)]
        if len(subset) < 3:
            continue
        print(f"\n  {label}: {len(subset)}ディレクトリ")
        for v in VARIANTS:
            col = f"irl_{v}"
            mean_val = subset[col].mean()
            print(f"    {v}: AUC平均={mean_val:.4f}")
        print(f"    RF_Dir: AUC平均={subset['rf_dir'].mean():.4f}")

    return dir_df


def analyze_by_developer(pred_df):
    """開発者単位の分析"""
    section("3. 開発者単位分析")

    dev_results = []
    for dev, grp in pred_df.groupby("developer"):
        if len(grp) < 60 or grp["label"].nunique() < 2:
            continue
        row = {"developer": dev, "n_pairs": len(grp), "pos_rate": grp["label"].mean()}
        for v in VARIANTS:
            v_grp = grp[grp["variant"] == v].dropna(subset=["irl_dir_prob"])
            if len(v_grp) >= 10 and v_grp["label"].nunique() == 2:
                row[f"irl_{v}"] = roc_auc_score(v_grp["label"], v_grp["irl_dir_prob"])
            else:
                row[f"irl_{v}"] = np.nan
        rf_grp = grp.dropna(subset=["rf_dir_prob"])
        if len(rf_grp) >= 10 and rf_grp["label"].nunique() == 2:
            row["rf_dir"] = roc_auc_score(rf_grp["label"], rf_grp["rf_dir_prob"])
        else:
            row["rf_dir"] = np.nan
        dev_results.append(row)

    dev_df = pd.DataFrame(dev_results)
    dev_df["best_irl"] = dev_df[["irl_lstm_baseline", "irl_lstm_attention", "irl_transformer"]].max(axis=1)
    dev_df["best_variant"] = dev_df[["irl_lstm_baseline", "irl_lstm_attention", "irl_transformer"]].idxmax(axis=1).str.replace("irl_", "")
    dev_df["irl_vs_rf"] = dev_df["best_irl"] - dev_df["rf_dir"]

    print(f"\n  分析対象: {len(dev_df)} 開発者")

    # バリアント間勝率
    print("\n### 3.1 バリアント間勝率（開発者別AUCで比較）")
    valid = dev_df.dropna(subset=["irl_lstm_baseline", "irl_lstm_attention", "irl_transformer"])
    for v in VARIANTS:
        wins = (valid[f"irl_{v}"] == valid[["irl_lstm_baseline", "irl_lstm_attention", "irl_transformer"]].max(axis=1)).sum()
        mean_auc = valid[f"irl_{v}"].mean()
        print(f"  {v:20s}: 最高AUC開発者数 {wins}/{len(valid)}, 平均AUC={mean_auc:.4f}")
    print(f"  {'RF_Dir':20s}: 平均AUC={valid['rf_dir'].mean():.4f}")

    # IRL vs RF_Dir
    valid2 = dev_df.dropna(subset=["best_irl", "rf_dir"])
    irl_wins = (valid2["irl_vs_rf"] > 0).sum()
    rf_wins = (valid2["irl_vs_rf"] < 0).sum()
    print(f"\n  IRL(best) vs RF_Dir: IRL勝ち {irl_wins}, RF勝ち {rf_wins} （{len(valid2)}開発者）")

    # 活動量別
    print("\n### 3.2 活動量別のバリアント性能")
    for label, lo, hi in [("少 (60-120)", 60, 120), ("中 (120-240)", 120, 240), ("多 (240+)", 240, 99999)]:
        subset = valid[(valid["n_pairs"] >= lo) & (valid["n_pairs"] < hi)]
        if len(subset) < 3:
            continue
        print(f"\n  {label}: {len(subset)}人")
        for v in VARIANTS:
            print(f"    {v}: AUC平均={subset[f'irl_{v}'].mean():.4f}")
        print(f"    RF_Dir: AUC平均={subset['rf_dir'].mean():.4f}")

    # 正例率別
    print("\n### 3.3 開発者の継続率別分析")
    for label, lo, hi in [("低活動 (pos<30%)", 0, 0.3), ("中活動 (30-60%)", 0.3, 0.6), ("高活動 (60%+)", 0.6, 1.1)]:
        subset = valid[(valid["pos_rate"] >= lo) & (valid["pos_rate"] < hi)]
        if len(subset) < 3:
            continue
        print(f"\n  {label}: {len(subset)}人")
        for v in VARIANTS:
            print(f"    {v}: AUC平均={subset[f'irl_{v}'].mean():.4f}")
        print(f"    RF_Dir: AUC平均={subset['rf_dir'].mean():.4f}")

    # attention/transformerが特に勝っている開発者
    print("\n### 3.4 attention/transformer が baseline に大きく勝つ開発者")
    valid_copy = valid.copy()
    valid_copy["att_vs_base"] = valid_copy["irl_lstm_attention"] - valid_copy["irl_lstm_baseline"]
    valid_copy["trans_vs_base"] = valid_copy["irl_transformer"] - valid_copy["irl_lstm_baseline"]

    print("\n  attention が baseline に最も勝つ開発者 TOP5:")
    top_att = valid_copy.nlargest(5, "att_vs_base")
    for _, r in top_att.iterrows():
        print(f"    {r['developer']:<45} base={r['irl_lstm_baseline']:.3f} att={r['irl_lstm_attention']:.3f} (+{r['att_vs_base']:.3f})")

    print("\n  transformer が baseline に最も勝つ開発者 TOP5:")
    top_trans = valid_copy.nlargest(5, "trans_vs_base")
    for _, r in top_trans.iterrows():
        print(f"    {r['developer']:<45} base={r['irl_lstm_baseline']:.3f} trans={r['irl_transformer']:.3f} (+{r['trans_vs_base']:.3f})")

    return dev_df


def analyze_variant_characteristics(pred_df):
    """バリアント間の特性差分析"""
    section("4. バリアント間の特性差")

    # 各バリアントの予測スコア分布
    print("\n### 4.1 予測スコア分布")
    for v in VARIANTS:
        v_data = pred_df[pred_df["variant"] == v]["irl_dir_prob"].dropna()
        print(f"  {v:20s}: mean={v_data.mean():.4f}, std={v_data.std():.4f}, "
              f"median={v_data.median():.4f}, min={v_data.min():.4f}, max={v_data.max():.4f}")

    # バリアント間の相関
    print("\n### 4.2 バリアント間の予測スコア相関（同一ペア）")
    # pivot: 同一 (developer, directory, eval_window) で各バリアントの予測を並べる
    pivot = pred_df.pivot_table(
        index=["developer", "directory", "train_window", "eval_window"],
        columns="variant",
        values="irl_dir_prob",
    ).dropna()
    if len(pivot) > 100:
        from scipy.stats import spearmanr
        for v1 in VARIANTS:
            for v2 in VARIANTS:
                if v1 >= v2:
                    continue
                corr, _ = spearmanr(pivot[v1], pivot[v2])
                print(f"  {v1} vs {v2}: spearman={corr:.4f}")

    # バリアント間で予測が大きく異なるケース
    print("\n### 4.3 バリアント間で予測が最も乖離するケース")
    if len(pivot) > 0:
        pivot["range"] = pivot[VARIANTS].max(axis=1) - pivot[VARIANTS].min(axis=1)
        top_divergent = pivot.nlargest(10, "range")
        print("  (developer, directory, train, eval) → baseline / attention / transformer")
        for idx, row in top_divergent.iterrows():
            dev, d, tw, ew = idx
            print(f"  {dev[:30]:<30} {d:<25} {tw} {ew}: "
                  f"{row['lstm_baseline']:.3f} / {row['lstm_attention']:.3f} / {row['transformer']:.3f} "
                  f"(range={row['range']:.3f})")


def main():
    print("=" * 80)
    print("  全バリアント包括分析")
    print("  lstm_baseline / lstm_attention / transformer")
    print("=" * 80)

    metrics_df = load_summary_metrics()
    pred_df = load_all_predictions()

    print(f"\n  総評価パターン: {metrics_df.groupby(['variant', 'train_window', 'eval_window']).ngroups}")
    print(f"  総ペア予測数: {len(pred_df)}")

    analyze_accuracy(metrics_df)
    analyze_by_directory(pred_df)
    analyze_by_developer(pred_df)
    analyze_variant_characteristics(pred_df)

    section("分析完了")


if __name__ == "__main__":
    main()
