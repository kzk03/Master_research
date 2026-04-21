#!/bin/bash
# 6バリアントを1コンテナ内でバックグラウンド並列実行
set -e

LOGDIR="outputs/variant_combined_v2/logs"
mkdir -p "$LOGDIR"

echo "=== 6バリアント並列実行開始 ==="

OUTBASE="outputs/variant_combined_v2"

bash scripts/run_variant_single.sh 0 lstm_baseline        "$OUTBASE" > "$LOGDIR/v0.log" 2>&1 &
bash scripts/run_variant_single.sh 1 lstm_attention        "$OUTBASE" > "$LOGDIR/v1.log" 2>&1 &
bash scripts/run_variant_single.sh 2 transformer           "$OUTBASE" > "$LOGDIR/v2.log" 2>&1 &
bash scripts/run_variant_single.sh 3 lstm_multitask        "$OUTBASE" > "$LOGDIR/v3.log" 2>&1 &
bash scripts/run_variant_single.sh 4 lstm_attn_multitask   "$OUTBASE" > "$LOGDIR/v4.log" 2>&1 &
bash scripts/run_variant_single.sh 5 transformer_multitask "$OUTBASE" > "$LOGDIR/v5.log" 2>&1 &

echo "6プロセス起動完了。ログ: $LOGDIR/v{0..5}.log"
echo "進捗確認: tail -1 $LOGDIR/v*.log"

wait
echo "=== 全バリアント完了 ==="
