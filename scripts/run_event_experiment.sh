#!/bin/bash
# ============================================================
# イベント単位系列の実験スクリプト
# ============================================================
# 月次集約版 (run_variant_single.sh) と並行してサーバで実行可能。
# 既存パイプラインには影響しない。
#
# 使い方:
#   bash scripts/run_event_experiment.sh [gpu_id]
#   例: bash scripts/run_event_experiment.sh 0
# ============================================================
set -e

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

REVIEWS="data/combined_raw.csv"
RAW_JSON=(data/raw_json/openstack__*.json)
TRAIN_START="2021-01-01"
TRAIN_END="2023-01-01"
EVAL_CUTOFF="2023-01-01"
EPOCHS=50
PATIENCE=5
OUTBASE="outputs/event_level_experiment"

MAX_EVENTS=256
SLIDING_WINDOW_DAYS=180
BATCH_SIZE=32

VARIANTS=(0 1 2)
VARIANT_NAMES=("lstm_baseline" "lstm_attention" "transformer")
TRAIN_WINDOWS=("0 3" "3 6" "6 9" "9 12")
TRAIN_WIN_LABELS=("0-3" "3-6" "6-9" "9-12")

# 評価1パターンを実行する関数
run_eval() {
    local model_path="$1"
    local train_win="$2"
    local eval_win="$3"
    local eval_fs="$4"
    local train_fs_val="$5"
    local vname="$6"
    local eval_dir="$OUTBASE/$vname/train_${train_win}m/eval_${eval_win}m"
    mkdir -p "$eval_dir"

    if [ -f "$eval_dir/summary_metrics.json" ]; then
        echo "[$vname train_${train_win}m -> eval_${eval_win}m] スキップ（評価済み）"
        return
    fi

    echo "[$vname train_${train_win}m -> eval_${eval_win}m] 評価開始..."
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
        --n-jobs -1 \
        --calibrate \
        --output-dir "$eval_dir"
    echo "[$vname train_${train_win}m -> eval_${eval_win}m] 評価完了"
}

EVAL_PARALLEL=4

echo ""
echo "============================================================"
echo "  [イベント単位系列] 実験開始 (GPU=$GPU_ID)"
echo "  MAX_EVENTS=$MAX_EVENTS, SLIDING_WINDOW=$SLIDING_WINDOW_DAYS days"
echo "============================================================"

# バリアントごとに逐次（メモリ考慮）、訓練窓は逐次（GPU共有）
for v_idx in "${!VARIANTS[@]}"; do
    VTYPE="${VARIANTS[$v_idx]}"
    VNAME="${VARIANT_NAMES[$v_idx]}"

    echo ""
    echo "---- バリアント: $VNAME (model_type=$VTYPE) ----"

    for w_idx in "${!TRAIN_WINDOWS[@]}"; do
        read -r FS FE <<< "${TRAIN_WINDOWS[$w_idx]}"
        WIN="${TRAIN_WIN_LABELS[$w_idx]}"
        MODEL_DIR="$OUTBASE/$VNAME/train_${WIN}m"
        MODEL_PATH="$MODEL_DIR/irl_model.pt"
        CACHE_PATH="$OUTBASE/trajectory_cache/event_train_${WIN}m.pkl"

        # 訓練
        if [ -f "$MODEL_PATH" ]; then
            echo "[$VNAME train_${WIN}m] スキップ（訓練済み）"
        else
            echo "[$VNAME train_${WIN}m] 訓練開始..."
            uv run python scripts/train/train_model_event.py \
                --reviews "$REVIEWS" \
                --raw-json "${RAW_JSON[@]}" \
                --train-start "$TRAIN_START" \
                --train-end "$TRAIN_END" \
                --future-window-start "$FS" \
                --future-window-end "$FE" \
                --epochs "$EPOCHS" \
                --patience "$PATIENCE" \
                --directory-level \
                --model-type "$VTYPE" \
                --max-events "$MAX_EVENTS" \
                --sliding-window-days "$SLIDING_WINDOW_DAYS" \
                --batch-size "$BATCH_SIZE" \
                --n-jobs -1 \
                --trajectories-cache "$CACHE_PATH" \
                --output "$MODEL_DIR"
            echo "[$VNAME train_${WIN}m] 訓練完了"
        fi

        # 評価（4パターンを並列実行）
        eval_count=0
        for e_idx in "${!TRAIN_WINDOWS[@]}"; do
            read -r EFS _ <<< "${TRAIN_WINDOWS[$e_idx]}"
            EWIN="${TRAIN_WIN_LABELS[$e_idx]}"

            # train <= eval の制約
            if [ "$FS" -gt "$EFS" ]; then
                continue
            fi

            run_eval "$MODEL_PATH" "$WIN" "$EWIN" "$EFS" "$FS" "$VNAME" &
            eval_count=$((eval_count + 1))
            if [ "$eval_count" -ge "$EVAL_PARALLEL" ]; then
                wait
                eval_count=0
            fi
        done
        wait
    done
done

echo ""
echo "============================================================"
echo "  全実験完了"
echo "============================================================"
