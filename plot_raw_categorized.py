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

RAW_SECTIONS = [
    ("1_Activity", "1. 活動量・経験量 (4)", [
        ("window_tenure_days", "対象期間内の継続日数 (窓内活動日数)"),
        ("total_reviews", "総レビュー依頼数 (対象期間)"),
        ("total_changes", "関与した総チェンジ数"),
        ("total_activity", "総アクティビティ数"),
    ]),
    ("2_Continuity", "2. 活動の継続性・頻度 (4)", [
        ("recent_review_count_30d", "直近30日のレビュー数"),
        ("mean_activity_gap_days", "平均活動間隔 (日)"),
        ("median_activity_gap_days", "活動間隔の中央値 (日)"),
        ("std_activity_gap_days", "活動間隔の標準偏差 (日)"),
        ("days_since_last_activity", "最終活動からの経過日数"),
        ("active_months", "アクティブ月数"),
    ]),
    ("3_Acceptance", "3. 承諾傾向（静的パターン）(4)", [
        ("accepted_count", "承諾レビュー数"),
        ("rejected_count", "拒否レビュー数"),
        ("acceptance_rate_raw", "承諾率 (全体)"),
        ("recent_acceptance_rate_raw", "直近の承諾率"),
    ]),
    ("4_Trend", "4. 疲弊・離脱の予兆（負荷・トレンド）(6)", [
        # (rawにはトレンド特徴量がないため空)
    ]),
    ("5_Collaboration", "5. 協力関係 (3)", [
        ("unique_collaborators", "ユニークな協調者数"),
        ("repeated_collaborators", "反復協調者数"),
        ("repeat_collaboration_rate_raw", "反復協調割合"),
    ]),
    ("6_Action", "6. レビュー対象の中身（行動特徴）(4)", [
        ("raw_mean_change_lines", "平均変更行数"),
        ("raw_median_change_lines", "変更行数の中央値"),
        ("raw_std_change_lines", "変更行数の標準偏差"),
        ("raw_max_change_lines", "最大変更行数"),
        ("raw_mean_files", "平均変更ファイル数"),
        ("raw_median_files", "変更ファイル数の中央値"),
        ("raw_std_files", "変更ファイル数の標準偏差"),
        ("raw_max_files", "最大変更ファイル数"),
        ("raw_mean_response_days", "平均応答日数"),
        ("raw_median_response_days", "応答日数の中央値"),
        ("raw_std_response_days", "応答日数の標準偏差"),
        ("raw_max_response_days", "最大応答日数"),
    ])
]

def main():
    csv_path = Path("outputs/feature_dist_raw/raw_feature_values.csv")
    out_dir = Path("outputs/feature_dist_raw/per_feature_categorized")
    md_path = Path("docs/feature_engineering/features_share.md")
    
    df = pd.read_csv(csv_path)

    # 1. 画像の生成
    for sec_dir_name, sec_title, feats in RAW_SECTIONS:
        sec_dir = out_dir / sec_dir_name
        sec_dir.mkdir(parents=True, exist_ok=True)

        for feat, ja_label in feats:
            if feat not in df.columns:
                continue
            
            v = df[feat].dropna().to_numpy()
            if len(v) == 0:
                continue

            if len(v) > 100:
                upper = np.percentile(v, 99.5)
            else:
                upper = v.max()

            def _filt(arr: np.ndarray) -> np.ndarray:
                return arr[arr <= upper]

            fig, ax = plt.subplots(figsize=(6, 4))
            if "__label" in df.columns:
                for lab, color, alpha in [(1, "C2", 0.5), (0, "C3", 0.5), (-1, "C7", 0.3)]:
                    sub = df.loc[df["__label"] == lab, feat].dropna().to_numpy()
                    sub = _filt(sub)
                    if len(sub) == 0:
                        continue
                    ax.hist(sub, bins=40, alpha=alpha, color=color,
                            label={1: "将来承諾", 0: "将来未承諾(拒否のみ)", -1: "将来依頼なし"}[lab], density=True)
                ax.legend(fontsize=10)
            else:
                v_show = _filt(v)
                if len(v_show) > 0:
                    ax.hist(v_show, bins=40, alpha=0.7, color="C0")

            n_total = len(v)
            ax.set_title(f"{ja_label} ({feat})\nn={n_total}", fontsize=12)
            ax.tick_params(labelsize=10)
            ax.set_xlabel(ja_label)
            ax.set_ylabel("密度 (Density)" if "__label" in df.columns else "頻度 (Frequency)")

            fig.tight_layout()
            out_pdf = sec_dir / f"{feat}.pdf"
            out_png = sec_dir / f"{feat}.png"
            fig.savefig(out_pdf, format="pdf")
            fig.savefig(out_png, format="png", dpi=300)
            plt.close(fig)
            logger.info(f"Saved: {out_png}")

    # 2. Markdownの更新（古い画像リンクを消して新しいものを追加）
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 古い画像ブロック（### 分布\n![...）を正規表現で削除
    md_text = re.sub(r'### 分布\n(!\[.*?\]\(.*?\)\n)+', '', md_text)
    md_text = md_text.replace('\n\n\n\n##', '\n\n##')

    for sec_dir_name, sec_title, feats in RAW_SECTIONS:
        # Markdownのセクション見出しを探す（"## 1. 活動量" など）
        # 完全一致しなくても前方一致で探す
        sec_prefix = "## " + sec_title.split(" ")[1] # e.g. "## 1."
        
        parts = md_text.split(sec_title)
        if len(parts) < 2:
            # prefixで探す
            lines = md_text.split('\n')
            found_idx = -1
            for i, line in enumerate(lines):
                if line.startswith(sec_prefix):
                    found_idx = i
                    break
            if found_idx == -1:
                continue
            
            # partsを作るのが面倒なので、テーブルの終わりを探す
            table_end_idx = found_idx
            in_table = False
            for i in range(found_idx, len(lines)):
                if lines[i].startswith('|'):
                    in_table = True
                elif in_table and not lines[i].startswith('|'):
                    table_end_idx = i
                    break
            
            img_links = "\n\n### Rawデータ分布\n"
            for feat, ja_label in feats:
                img_path = f"../../outputs/feature_dist_raw/per_feature_categorized/{sec_dir_name}/{feat}.png"
                img_links += f"![{feat}]({img_path})\n"
                
            lines.insert(table_end_idx, img_links)
            md_text = "\n".join(lines)
            continue

        before = parts[0]
        after = parts[1]
        
        lines = after.split('\n')
        table_end_idx = 0
        in_table = False
        for i, line in enumerate(lines):
            if line.startswith('|'):
                in_table = True
            elif in_table and not line.startswith('|'):
                table_end_idx = i
                break
        
        img_links = "\n\n### Rawデータ分布\n"
        for feat, ja_label in feats:
            img_path = f"../../outputs/feature_dist_raw/per_feature_categorized/{sec_dir_name}/{feat}.png"
            img_links += f"![{feat}]({img_path})\n"
            
        lines.insert(table_end_idx, img_links)
        new_after = '\n'.join(lines)
        md_text = before + sec_title + new_after

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print("Updated markdown with RAW data")

if __name__ == "__main__":
    main()
