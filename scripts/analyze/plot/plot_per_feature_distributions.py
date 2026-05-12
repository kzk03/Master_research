import argparse
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

FEATURE_JA_LABELS = {
    "window_tenure_days": "対象期間内の継続日数 (窓内活動日数)",
    "total_reviews": "総レビュー依頼数 (対象期間)",
    "total_changes": "関与した総チェンジ数",
    "total_activity": "総アクティビティ数",
    "recent_review_count_30d": "直近30日のレビュー数",
    "mean_activity_gap_days": "平均活動間隔 (日)",
    "median_activity_gap_days": "活動間隔の中央値 (日)",
    "std_activity_gap_days": "活動間隔の標準偏差 (日)",
    "days_since_last_activity": "最終活動からの経過日数",
    "unique_collaborators": "ユニークな協調者数",
    "repeated_collaborators": "反復協調者数",
    "repeat_collaboration_rate_raw": "反復協調割合",
    "accepted_count": "承諾レビュー数",
    "rejected_count": "拒否レビュー数",
    "acceptance_rate_raw": "承諾率 (全体)",
    "recent_acceptance_rate_raw": "直近の承諾率",
    "raw_mean_change_lines": "平均変更行数",
    "raw_median_change_lines": "変更行数の中央値",
    "raw_std_change_lines": "変更行数の標準偏差",
    "raw_max_change_lines": "最大変更行数",
    "raw_mean_files": "平均変更ファイル数",
    "raw_median_files": "変更ファイル数の中央値",
    "raw_std_files": "変更ファイル数の標準偏差",
    "raw_max_files": "最大変更ファイル数",
    "raw_mean_response_days": "平均応答日数",
    "raw_median_response_days": "応答日数の中央値",
    "raw_std_response_days": "応答日数の標準偏差",
    "raw_max_response_days": "最大応答日数",
    "active_months": "アクティブ月数"
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--clip-percentile", type=float, default=99.5)
    p.add_argument("--log-scale", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"読み込み: {args.csv}")
    df = pd.read_csv(args.csv)

    for feat in df.columns:
        if feat.startswith("__"):
            continue

        ja_label = FEATURE_JA_LABELS.get(feat, feat)

        v = df[feat].dropna().to_numpy()
        if len(v) == 0:
            continue

        if len(v) > 100:
            upper = np.percentile(v, args.clip_percentile)
        else:
            upper = v.max()

        def _filt(arr: np.ndarray) -> np.ndarray:
            arr = arr[arr <= upper]
            if args.log_scale:
                arr = arr[arr > 0]
            return arr

        fig, ax = plt.subplots(figsize=(6, 4))

        if "__label" in df.columns:
            for lab, color, alpha in [(1, "C2", 0.5), (0, "C3", 0.5), (-1, "C7", 0.3)]:
                sub = df.loc[df["__label"] == lab, feat].dropna().to_numpy()
                sub = _filt(sub)
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
            v_show = _filt(v)
            if len(v_show) > 0:
                ax.hist(v_show, bins=40, alpha=0.7, color="C0")

        for p_val, ls in [(50, ":"), (90, "--"), (99, "-.")]:
            val = np.percentile(v, p_val)
            if args.log_scale and val <= 0:
                continue
            ax.axvline(val, color="k", ls=ls, lw=1.0, label=f"p{p_val}={val:.2f}")

        if args.log_scale:
            ax.set_xscale("log")

        n_total = len(v)
        n_nan = int(df[feat].isna().sum())
        ax.set_title(f"{ja_label} ({feat})\nn={n_total} (nan={n_nan})", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10, loc="upper right")
        
        ax.set_xlabel(ja_label)
        ax.set_ylabel("密度 (Density)" if "__label" in df.columns else "頻度 (Frequency)")

        fig.tight_layout()
        out_pdf = out / f"{feat}.pdf"
        out_png = out / f"{feat}.png"
        fig.savefig(out_pdf, format="pdf")
        fig.savefig(out_png, format="png", dpi=300)
        plt.close(fig)
        logger.info(f"Saved: {out_pdf} and {out_png}")

if __name__ == "__main__":
    main()
