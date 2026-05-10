#!/bin/bash
# tmp: 元 pilot を kill せず未着手の 6-9m と 9-12m を並列で走らせる。
# 各 window は --n-jobs 1（GIL 回避のため逐次）、window 同士はプロセス並列。
set -e
cd "$(dirname "$0")/.."

REVIEWS="data/combined_raw.csv"
RAW_JSON=(
    data/raw_json/openstack__nova.json
    data/raw_json/openstack__cinder.json
    data/raw_json/openstack__neutron.json
    data/raw_json/openstack__ironic.json
    data/raw_json/openstack__glance.json
    data/raw_json/openstack__keystone.json
    data/raw_json/openstack__horizon.json
    data/raw_json/openstack__swift.json
    data/raw_json/openstack__heat.json
    data/raw_json/openstack__octavia.json
)
TRAIN_END="2022-01-01"
EVAL_CUTOFF="2023-01-01"
DEVICE="cpu"
PILOT_DIR="outputs/mce_pilot_multiwindow"
TRAIN_OUT="$PILOT_DIR/monthly_cold_multiwindow"
WINDOWS=("6-9" "9-12")

mkdir -p logs

run_eval_tmp() {
    local window="$1"
    local fs fe
    fs="${window%-*}"
    fe="${window##*-}"
    local out_dir="$PILOT_DIR/eval_${window}m_tmp"
    if [ -f "$out_dir/summary_metrics.json" ]; then
        echo "[eval ${window}m] skip"
        return
    fi
    mkdir -p "$out_dir"
    echo "[eval ${window}m] running... (fs=$fs, fe=$fe)"
    uv run python scripts/analyze/eval/eval_mce_irl_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$fs" \
        --rf-future-start-months "$fs" \
        --irl-dir-model "$TRAIN_OUT/mce_irl_model.pt" \
        --rf-train-end "$TRAIN_END" \
        --device "$DEVICE" --n-jobs 1 \
        --output-dir "$out_dir" \
        --calibrate
    echo "[eval ${window}m] done → $out_dir/summary_metrics.json"
}

pids=()
for win in "${WINDOWS[@]}"; do
    run_eval_tmp "$win" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "=== tmp pilot 全 window 完了 ==="
