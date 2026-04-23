#!/bin/bash
# [サーバ用] non-MT 3バリアント（0,1,2）の全パイプライン実行
# GPU 1枚前提、逐次実行
#
# 実行フロー:
#   1. 軌跡キャッシュ作成（4窓並列）
#   2. 3バリアントを逐次で学習+評価
#
# 使い方:
#   bash scripts/run_all.sh [outbase]

set -e

OUTBASE="${1:-outputs/variant_comparison_server}"
LOGDIR="$OUTBASE/logs"
mkdir -p "$LOGDIR"

echo "============================================================"
echo "  [サーバ] 全バリアント実行 (GPU=1)"
echo "  出力: $OUTBASE"
echo "  開始: $(date)"
echo "============================================================"

# ── Step 1: 軌跡キャッシュ作成 ──
echo ""
echo "=== Step 1: 軌跡キャッシュ作成 ==="
bash scripts/extract_trajectories_cache.sh "$OUTBASE" 2>&1 | tee "$LOGDIR/extract_trajectories.log"

# ── Step 2: 3バリアント逐次実行 ──
echo ""
echo "=== Step 2: 3バリアント学習+評価（逐次） ==="

VARIANTS=(
    "0 lstm_baseline"
    "1 lstm_attention"
    "2 transformer"
)

for idx in 0 1 2; do
    read -r vtype vname <<< "${VARIANTS[$idx]}"
    echo ""
    echo "--- $vname 開始: $(date) ---"
    bash scripts/run_variant_single.sh "$vtype" "$vname" "$OUTBASE" "0" \
        2>&1 | tee "$LOGDIR/${vname}.log"
    echo "--- $vname 完了: $(date) ---"
done

echo ""
echo "============================================================"
echo "  [サーバ] 全バリアント完了"
echo "  終了: $(date)"
echo "============================================================"

# 結果サマリ
echo ""
echo "=== 結果サマリ ==="
for vname in lstm_baseline lstm_attention transformer; do
    count=$(find "$OUTBASE/$vname" -name "summary_metrics.json" 2>/dev/null | wc -l)
    models=$(find "$OUTBASE/$vname" -name "irl_model.pt" 2>/dev/null | wc -l)
    echo "  $vname: モデル=${models}, 評価=${count}/10"
done
