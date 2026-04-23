#!/bin/bash
# [サーバ用] 1つのバリアントに対して学習+評価10パターンを実行
# ローカル版との差分:
#   - GPU 使用（--device cuda, CUDA_VISIBLE_DEVICES）
#   - 評価も --device cuda
#   - MT バリアント非対応（non-MT のみ）
#
# 使い方:
#   bash scripts/run_variant_single.sh <variant_id> <variant_name> [outbase] [gpu_id]
#
# 例:
#   bash scripts/run_variant_single.sh 0 lstm_baseline outputs/variant_comparison_server 0

set -e

VTYPE="${1:?Usage: $0 <variant_id> <variant_name> [outbase] [gpu_id]}"
VNAME="${2:?Usage: $0 <variant_id> <variant_name> [outbase] [gpu_id]}"
OUTBASE="${3:-outputs/variant_comparison_server}"
GPU_ID="${4:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

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
TRAIN_START="2019-01-01"
TRAIN_END="2022-01-01"
EVAL_CUTOFF="2023-01-01"
EPOCHS=50
PATIENCE=5

TRAIN_WINDOWS=("0-3" "3-6" "6-9" "9-12")
TRAIN_FS=(0 3 6 9)
TRAIN_FE=(3 6 9 12)

CACHE_DIR="$OUTBASE/trajectory_cache"
mkdir -p "$CACHE_DIR"

# 評価1パターンを実行する関数
run_eval() {
    local model_path="$1"
    local train_win="$2"
    local eval_win="$3"
    local eval_fs="$4"
    local train_fs_val="$5"
    local eval_dir="$OUTBASE/$VNAME/train_${train_win}m/eval_${eval_win}m"
    mkdir -p "$eval_dir"

    if [ -f "$eval_dir/summary_metrics.json" ]; then
        echo "[$VNAME train_${train_win}m -> eval_${eval_win}m] スキップ（評価済み）"
        return
    fi

    echo "[$VNAME train_${train_win}m -> eval_${eval_win}m] 評価開始..."
    uv run python scripts/analyze/eval_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$eval_fs" \
        --rf-future-start-months "$train_fs_val" \
        --irl-dir-model "$model_path" \
        --rf-train-end "$TRAIN_END" \
        --device cuda \
        --n-jobs 4 \
        --output-dir "$eval_dir"
    echo "[$VNAME train_${train_win}m -> eval_${eval_win}m] 評価完了"
}

# 評価並列数（同時にバックグラウンドで走る評価プロセス数）
EVAL_PARALLEL=4

echo ""
echo "============================================================"
echo "  [サーバ] バリアント $VTYPE: $VNAME (GPU=$GPU_ID)"
echo "============================================================"

# ── Non-multi-task: 4モデル逐次訓練（GPU共有のため） ──
for i in 0 1 2 3; do
    win="${TRAIN_WINDOWS[$i]}"
    fs="${TRAIN_FS[$i]}"
    fe="${TRAIN_FE[$i]}"
    model_dir="$OUTBASE/$VNAME/train_${win}m"
    mkdir -p "$model_dir"

    if [ -f "$model_dir/irl_model.pt" ]; then
        echo "[$VNAME train_${win}m] 学習スキップ（学習済み）"
    else
        echo "[$VNAME train_${win}m] 学習開始 (future_window=${fs}-${fe}m, GPU=$GPU_ID)..."
        uv run python scripts/train/train_model.py \
            --directory-level \
            --model-type "$VTYPE" \
            --raw-json "${RAW_JSON[@]}" \
            --reviews "$REVIEWS" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --train-start "$TRAIN_START" \
            --train-end "$TRAIN_END" \
            --future-window-start "$fs" \
            --future-window-end "$fe" \
            --trajectories-cache "$CACHE_DIR/traj_${win}.pkl" \
            --output "$model_dir"
        echo "[$VNAME train_${win}m] 学習完了"
    fi
done

# ── 評価: 最大 EVAL_PARALLEL 個ずつ並列実行 ──
running=0
for ti in 0 1 2 3; do
    train_win="${TRAIN_WINDOWS[$ti]}"
    model_path="$OUTBASE/$VNAME/train_${train_win}m/irl_model.pt"

    if [ ! -f "$model_path" ]; then
        echo "[$VNAME train_${train_win}m] モデルが見つからない、スキップ"
        continue
    fi

    for ei in $(seq $ti 3); do
        eval_win="${TRAIN_WINDOWS[$ei]}"
        eval_fs="${TRAIN_FS[$ei]}"
        train_fs_val="${TRAIN_FS[$ti]}"
        run_eval "$model_path" "$train_win" "$eval_win" "$eval_fs" "$train_fs_val" &
        running=$((running + 1))
        if [ "$running" -ge "$EVAL_PARALLEL" ]; then
            wait
            running=0
        fi
    done
done
if [ "$running" -gt 0 ]; then
    wait
fi
echo "[$VNAME] 評価全パターン完了"

echo ""
echo "[$VNAME] 全パターン完了"
