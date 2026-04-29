"""
期間別レビュアー承諾/非承諾の変動分析

モデル（train_model.py）と同じラベル定義に準拠:
- accept: 指定窓内に依頼あり、1つでも承諾 (正例)
- reject: 指定窓内に依頼あり、全拒否 (負例、weight=1.0)
- weak_neg: 指定窓内に依頼なし、拡張期間(0-12m全体)に依頼あり (弱い負例、weight=0.1)
- exclude: 拡張期間にも依頼なし → 除外

変動の定義:
- 正例(accept) と 負例(reject or weak_neg) が混在するレビュアー
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np


# 時間窓の定義（基準点からの月数）
TIME_WINDOWS = [
    ("0-3m", 0, 3),
    ("3-6m", 3, 6),
    ("6-9m", 6, 9),
    ("9-12m", 9, 12),
]

EXTENDED_WINDOW_MONTHS = 12  # 拡張ラベル期間


def classify_reviewer_per_window(
    df: pd.DataFrame,
    base_date: pd.Timestamp,
) -> pd.DataFrame:
    """レビュアーごとに各時間窓でのラベルを分類（モデルと同じ定義）"""

    # 拡張期間のデータ（0-12m全体）
    ext_start = base_date
    ext_end = base_date + pd.DateOffset(months=EXTENDED_WINDOW_MONTHS)
    ext_df = df[(df["timestamp"] >= ext_start) & (df["timestamp"] < ext_end)]

    # 拡張期間にデータがあるレビュアー
    ext_reviewers = set(ext_df["email"].unique())

    results = []

    for email, group in df.groupby("email"):
        row = {"email": email}
        for window_name, start_m, end_m in TIME_WINDOWS:
            window_start = base_date + pd.DateOffset(months=start_m)
            window_end = base_date + pd.DateOffset(months=end_m)

            window_data = group[
                (group["timestamp"] >= window_start)
                & (group["timestamp"] < window_end)
            ]

            if len(window_data) > 0:
                # 指定窓内に依頼あり
                if window_data["label"].sum() > 0:
                    row[window_name] = "accept"
                else:
                    row[window_name] = "reject"
            elif email in ext_reviewers:
                # 指定窓に依頼なし、拡張期間に依頼あり → 弱い負例
                row[window_name] = "weak_neg"
            else:
                # 拡張期間にも依頼なし → 除外
                row[window_name] = "exclude"

        results.append(row)

    return pd.DataFrame(results)


def analyze_fluctuation(classified: pd.DataFrame) -> dict:
    """変動パターンを集計"""
    window_names = [w[0] for w in TIME_WINDOWS]

    # exclude 以外のステータス
    active_statuses = {"accept", "reject", "weak_neg"}
    negative_statuses = {"reject", "weak_neg"}

    # 全期間にわたるパターン
    patterns = Counter()
    for _, row in classified.iterrows():
        pattern = tuple(row[w] for w in window_names)
        patterns[pattern] += 1

    # 少なくとも2期間でexclude以外のレビュアーのみ
    active = classified.copy()
    active["n_active"] = active[window_names].apply(
        lambda r: sum(1 for v in r if v in active_statuses), axis=1
    )
    multi_period = active[active["n_active"] >= 2]

    # 変動の定義: active な期間で accept と negative(reject/weak_neg) の両方がある
    def has_fluctuation(row):
        states = [row[w] for w in window_names if row[w] in active_statuses]
        has_accept = "accept" in states
        has_negative = any(s in negative_statuses for s in states)
        return has_accept and has_negative

    multi_period = multi_period.copy()
    multi_period["fluctuates"] = multi_period.apply(has_fluctuation, axis=1)

    # 連続する期間での遷移を分析
    transitions = Counter()
    for _, row in multi_period.iterrows():
        states = [(w, row[w]) for w in window_names if row[w] in active_statuses]
        for i in range(len(states) - 1):
            from_w, from_s = states[i]
            to_w, to_s = states[i + 1]
            transitions[(from_s, to_s)] += 1

    return {
        "patterns": patterns,
        "multi_period": multi_period,
        "transitions": transitions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="combined_raw.csv")
    parser.add_argument("--base-date", type=str, default=None,
                        help="Base date (YYYY-MM-DD). Default: 12 months before data end")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # データの期間を確認
    print(f"Data range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # 基準点
    base_date = pd.Timestamp(args.base_date) if args.base_date else (
        df["timestamp"].max() - pd.DateOffset(months=12)
    )
    base_date = base_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    print(f"Base date: {base_date}")
    print(f"Analysis window: {base_date} ~ {base_date + pd.DateOffset(months=12)}")
    print(f"Label definition: same as train_model.py (accept/reject/weak_neg/exclude)")

    # レビュアー分類
    print("\n=== Classifying reviewers per window ===")
    classified = classify_reviewer_per_window(df, base_date)

    window_names = [w[0] for w in TIME_WINDOWS]

    # 各期間の分布
    print("\nPer-window distribution:")
    for w in window_names:
        counts = classified[w].value_counts()
        total = len(classified)
        print(f"  {w}:")
        for status in ["accept", "reject", "weak_neg", "exclude"]:
            n = counts.get(status, 0)
            print(f"    {status}: {n} ({n/total*100:.1f}%)")

    # exclude を除いた統計
    n_exclude_all = sum(
        1 for _, row in classified.iterrows()
        if all(row[w] == "exclude" for w in window_names)
    )
    print(f"\nReviewers excluded from all windows: {n_exclude_all}")
    print(f"Reviewers with at least 1 active window: {len(classified) - n_exclude_all}")

    # 変動分析
    print("\n=== Fluctuation analysis ===")
    result = analyze_fluctuation(classified)
    multi = result["multi_period"]
    n_multi = len(multi)
    n_fluct = multi["fluctuates"].sum()

    print(f"Total reviewers: {len(classified)}")
    print(f"Multi-period reviewers (>=2 non-exclude): {n_multi}")
    print(f"Fluctuating reviewers (accept <-> reject/weak_neg): {n_fluct} ({n_fluct/n_multi*100:.1f}%)")
    print(f"Stable reviewers: {n_multi - n_fluct} ({(n_multi-n_fluct)/n_multi*100:.1f}%)")

    # 遷移マトリクス
    print("\nTransition matrix:")
    trans = result["transitions"]
    for (from_s, to_s), count in sorted(trans.items(), key=lambda x: -x[1]):
        print(f"  {from_s} -> {to_s}: {count}")

    # Top 変動パターン
    print("\nTop 20 behavior patterns:")
    for pattern, count in result["patterns"].most_common(20):
        pstr = " -> ".join(pattern)
        print(f"  {pstr}: {count}")

    # 活動期間数による層別化
    print("\n=== By number of active periods ===")
    active_statuses = {"accept", "reject", "weak_neg"}
    negative_statuses = {"reject", "weak_neg"}

    multi_all = classified.copy()
    multi_all["n_active"] = multi_all[window_names].apply(
        lambda r: sum(1 for v in r if v in active_statuses), axis=1
    )
    for n_act in [1, 2, 3, 4]:
        subset = multi_all[multi_all["n_active"] == n_act]
        if len(subset) == 0:
            continue

        def has_fluct(row):
            states = [row[w] for w in window_names if row[w] in active_statuses]
            return "accept" in states and any(s in negative_statuses for s in states)

        n_f = subset.apply(has_fluct, axis=1).sum() if n_act >= 2 else 0
        fluct_str = f"{n_f} fluctuating ({n_f/len(subset)*100:.1f}%)" if n_act >= 2 else "N/A (single period)"
        print(f"  {n_act} active periods: {len(subset)} reviewers, {fluct_str}")

    # プロジェクト別分析
    print("\n=== Per-project fluctuation ===")
    for project in sorted(df["project"].unique()):
        proj_df = df[df["project"] == project]
        proj_classified = classify_reviewer_per_window(proj_df, base_date)
        proj_result = analyze_fluctuation(proj_classified)
        proj_multi = proj_result["multi_period"]
        if len(proj_multi) == 0:
            continue
        n_f = proj_multi["fluctuates"].sum()
        print(f"  {project}: {len(proj_multi)} multi-period, "
              f"{n_f} fluctuating ({n_f/len(proj_multi)*100:.1f}%)")

    # 結果保存
    classified.to_csv(args.output_dir / "reviewer_fluctuation.csv", index=False)

    summary = {
        "base_date": str(base_date),
        "label_definition": "accept/reject/weak_neg/exclude (same as train_model.py)",
        "total_reviewers": len(classified),
        "multi_period_reviewers": n_multi,
        "fluctuating_reviewers": int(n_fluct),
        "fluctuation_rate": round(n_fluct / n_multi * 100, 1),
        "transitions": {f"{f}->{t}": c for (f, t), c in trans.items()},
    }
    with open(args.output_dir / "fluctuation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
