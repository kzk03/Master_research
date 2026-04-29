#!/bin/bash
# 依存関係分析の実行スクリプト

set -e
cd "$(dirname "$0")/../.."

RAW_JSON="data/raw_json/openstack__*.json"
DATA="data/combined_raw.csv"
PAIR_PRED="outputs/variant_comparison_server/lstm_baseline/train_0-3m/eval_0-3m/pair_predictions.csv"
OUT="experiments/dependency_analysis/results"

echo "=== 01: Co-change graph ==="
uv run python experiments/dependency_analysis/01_cochange_graph.py \
    --raw-json $RAW_JSON \
    --pair-predictions "$PAIR_PRED" \
    --output-dir "$OUT"

echo ""
echo "=== 02: Cross-project reviewer ==="
uv run python experiments/dependency_analysis/02_cross_project_reviewer.py \
    --data "$DATA" \
    --pair-predictions "$PAIR_PRED" \
    --output-dir "$OUT"

echo ""
echo "=== 03: Directory overlap / coherence ==="
uv run python experiments/dependency_analysis/03_dir_overlap_analysis.py \
    --raw-json $RAW_JSON \
    --data "$DATA" \
    --pair-predictions "$PAIR_PRED" \
    --output-dir "$OUT"
