import re
import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DEF = [
    # 1. 活動量・経験量
    ("1_Activity", "window_tenure_days", "窓内 tenure"),
    ("1_Activity", "total_changes", "PR 投稿数"),
    ("1_Activity", "total_reviews", "レビュー依頼受信数"),
    ("1_Activity", "core_reviewer_ratio", "コアレビュアー度"),

    # 2. 活動の継続性・頻度
    ("2_Continuity", "recent_activity_frequency", "直近活動頻度"),
    ("2_Continuity", "avg_activity_gap", "平均活動間隔"),
    ("2_Continuity", "days_since_last_activity", "最終活動からの経過日数"),
    ("2_Continuity", "active_months_ratio", "活動月割合"),

    # 3. 承諾傾向 (静的パターン)
    ("3_Acceptance", "overall_acceptance_rate", "全期間承諾率"),
    ("3_Acceptance", "recent_acceptance_rate", "直近承諾率"),
    ("3_Acceptance", "acceptance_rate_last10", "直近 10 件承諾率"),
    ("3_Acceptance", "complex_pr_bias", "複雑 PR 承諾バイアス"),

    # 4. 疲弊・離脱の予兆
    ("4_Trend", "recent_load_ratio_30d_all", "30日/全期間負荷比"),
    ("4_Trend", "recent_load_ratio_7d_30d", "7日/30日負荷比"),
    ("4_Trend", "activity_trend", "活動トレンド"),
    ("4_Trend", "acceptance_trend", "承諾率トレンド"),
    ("4_Trend", "response_time_trend", "応答速度トレンド"),
    ("4_Trend", "recent_rejection_streak", "直近連続拒否数"),

    # 5. 協力関係
    ("5_Collaboration", "unique_collaborator_count", "ユニーク協力者数"),
    ("5_Collaboration", "repeat_collaboration_rate", "反復協力率"),
    ("5_Collaboration", "reciprocity_score", "相互レビュー率"),

    # 6. レビュー対象の中身
    ("6_Action", "avg_action_intensity", "平均行動強度"),
    ("6_Action", "avg_change_lines", "平均変更行数"),
    ("6_Action", "avg_review_size", "平均レビューサイズ"),
    ("6_Action", "avg_response_time", "平均応答速度"),

    # 7. ディレクトリ親和度 (skip for now since they are path-level features)
]

def main():
    csv_path = Path("outputs/feature_dist_main32/feature_values.csv")
    out_dir = Path("outputs/feature_dist_main32/per_feature_processed")
    md_path = Path("docs/feature_engineering/features_share.md")
    
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    base_repo_dir = Path(".").resolve()

    feat_to_ja = {var: ja for sec, var, ja in FEATURE_DEF}
    feat_to_sec = {var: sec for sec, var, ja in FEATURE_DEF}

    # Plot
    for sec, var, ja in FEATURE_DEF:
        if var not in df.columns:
            logger.warning(f"Feature not in DF: {var}")
            continue

        sec_dir = out_dir / sec
        sec_dir.mkdir(parents=True, exist_ok=True)

        v = df[var].dropna().to_numpy()
        if len(v) == 0:
            continue

        # Plot processed features, limit extreme outliers just for presentation
        if len(v) > 100:
            upper = np.percentile(v, 99.5)
            lower = np.percentile(v, 0.5)
        else:
            upper = v.max()
            lower = v.min()

        arr = v[(v <= upper) & (v >= lower)]
        if len(arr) == 0:
            arr = v

        fig, ax = plt.subplots(figsize=(6, 4))
        if "__label" in df.columns:
            for lab, color, alpha in [(1, "C2", 0.5), (0, "C3", 0.5), (-1, "C7", 0.3)]:
                sub = df.loc[df["__label"] == lab, var].dropna().to_numpy()
                sub = sub[(sub <= upper) & (sub >= lower)]
                if len(sub) == 0:
                    continue
                ax.hist(
                    sub,
                    bins=40,
                    alpha=alpha,
                    color=color,
                    label={1: "将来承諾", 0: "将来未承諾(拒否のみ)", -1: "将来依頼なし"}[lab],
                    density=True,
                )
            ax.legend(fontsize=10)
        else:
            ax.hist(arr, bins=40, alpha=0.7, color="C0")

        n_total = len(v)
        ax.set_title(f"{ja} ({var})\nn={n_total}", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlabel(ja)
        ax.set_ylabel("密度 (Density)" if "__label" in df.columns else "頻度 (Frequency)")

        fig.tight_layout()
        out_pdf = sec_dir / f"{var}.pdf"
        out_png = sec_dir / f"{var}.png"
        fig.savefig(out_pdf, format="pdf")
        fig.savefig(out_png, format="png", dpi=300)
        plt.close(fig)
        logger.info(f"Saved: {out_png}")

    # Re-write markdown
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Create mapping table for each section
    # Then embed png at the end of section

    lines = md_content.split('\n')
    new_lines = []
    
    current_sec_id = None
    for line in lines:
        if line.startswith('## 1. '):
            current_sec_id = '1_Activity'
        elif line.startswith('## 2. '):
            current_sec_id = '2_Continuity'
        elif line.startswith('## 3. '):
            current_sec_id = '3_Acceptance'
        elif line.startswith('## 4. '):
            current_sec_id = '4_Trend'
        elif line.startswith('## 5. '):
            current_sec_id = '5_Collaboration'
        elif line.startswith('## 6. '):
            current_sec_id = '6_Action'
        elif line.startswith('## 7. '):
            current_sec_id = '7_Affinity'

        new_lines.append(line)
        
        # When we detect end of a section (empty line before next heading) or end of file
        # We can wait, it's easier to insert images at the end of the markdown script entirely, or right under the tables.
        
    # Better approach: 
    # Let's completely rewrite the markdown file keeping original descriptions but adding proper column, variable names and images.

if __name__ == "__main__":
    main()
