#!/bin/bash
# Engagement 軌跡 + Engagement 評価 のパイロット実験
# train_0-3m × eval_0-3m

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_BASE="outputs/mce_pilot_engagement"
CACHE_DIR="$OUTPUT_BASE/cache"
TRAIN_DIR="$OUTPUT_BASE/event_cold"
EVAL_DIR="$OUTPUT_BASE/eval_event_cold"
EVAL_ACCEPT_DIR="$OUTPUT_BASE/eval_event_cold_accept"

mkdir -p "$CACHE_DIR" "$TRAIN_DIR" "$EVAL_DIR" "$EVAL_ACCEPT_DIR"

RAW_JSON_FILES=(
    data/raw_json/openstack__cinder.json
    data/raw_json/openstack__glance.json
    data/raw_json/openstack__heat.json
    data/raw_json/openstack__horizon.json
    data/raw_json/openstack__ironic.json
    data/raw_json/openstack__keystone.json
    data/raw_json/openstack__neutron.json
    data/raw_json/openstack__nova.json
    data/raw_json/openstack__octavia.json
    data/raw_json/openstack__swift.json
)

TRAJ_CACHE="$CACHE_DIR/engagement_traj_0-3.pkl"

# ── 1) trajectory 抽出 ───────────────────────────────────────────────
if [[ ! -f "$TRAJ_CACHE" ]]; then
    echo "=== 1) 軌跡抽出 ==="
    uv run python scripts/train/extract_engagement_event_trajectories.py \
        --events data/combined_events.csv \
        --reviews data/combined_raw.csv \
        --raw-json "${RAW_JSON_FILES[@]}" \
        --dir-class-mapping outputs/dir_class_mapping_K15.json \
        --train-start 2019-01-01 --train-end 2022-01-01 \
        --future-window-start 0 --future-window-end 3 \
        --sliding-window-days 180 --max-events 256 \
        --n-jobs -1 \
        --output "$TRAJ_CACHE"
fi

# ── 2) 学習 (cold start: random init で MCE-IRL) ─────────────────────
MODEL_FILE="$TRAIN_DIR/mce_event_irl_model.pt"
if [[ ! -f "$MODEL_FILE" ]]; then
    echo "=== 2) MCE-IRL 学習 (cold start) ==="
    uv run python scripts/train/train_mce_event_irl_multiclass.py \
        --trajectories-cache "$TRAJ_CACHE" \
        --dir-class-mapping outputs/dir_class_mapping_K15.json \
        --model-type 0 \
        --epochs 50 --patience 5 --batch-size 32 \
        --learning-rate 3e-4 --hidden-dim 128 --dropout 0.2 \
        --output "$TRAIN_DIR"
fi

# ── 3) Engagement 評価 ─────────────────────────────────────────────
echo "=== 3) Engagement 評価 (eval_0-3m) ==="
uv run python scripts/analyze/eval/eval_mce_engagement_prediction.py \
    --events data/combined_events.csv \
    --data data/combined_raw.csv \
    --raw-json "${RAW_JSON_FILES[@]}" \
    --model "$MODEL_FILE" \
    --dir-class-mapping outputs/dir_class_mapping_K15.json \
    --prediction-time 2022-01-01 \
    --delta-months 3 \
    --window-days 180 \
    --output-dir "$EVAL_DIR"

# ── 4) 既存の Accept 評価 (比較用) ────────────────────────────────
echo "=== 4) Accept 評価 (比較用、既存指標) ==="
uv run python scripts/analyze/eval/eval_mce_event_irl_multiclass_prediction.py \
    --data data/combined_raw.csv \
    --raw-json "${RAW_JSON_FILES[@]}" \
    --model "$MODEL_FILE" \
    --dir-class-mapping outputs/dir_class_mapping_K15.json \
    --prediction-time 2022-01-01 \
    --delta-months 3 \
    --window-days 180 \
    --output-dir "$EVAL_ACCEPT_DIR"

echo "=== 完了 ==="
echo "engagement 評価: $EVAL_DIR/summary_metrics.json"
echo "accept 評価:     $EVAL_ACCEPT_DIR/summary_metrics.json"
