#!/bin/bash
# 1つのバリアントに対して10パターン（train<=eval制約）を実行
# 訓練・評価ともに並列実行
#
# 使い方:
#   bash scripts/variant/run_variant_single.sh <variant_id> <variant_name> [outbase]

set -e

VTYPE="${1:?Usage: $0 <variant_id> <variant_name> [outbase]}"
VNAME="${2:?Usage: $0 <variant_id> <variant_name> [outbase]}"
OUTBASE="${3:-outputs/variant_comparison_combined}"

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
SAVE_IMPORTANCE="${4:-false}"   # 4番目の引数で importance 保存を制御

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
    local importance_flag=""
    if [ "$SAVE_IMPORTANCE" = "true" ]; then
        importance_flag="--save-importance"
    fi
    uv run python scripts/analyze/eval/eval_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$eval_fs" \
        --rf-future-start-months "$train_fs_val" \
        --irl-dir-model "$model_path" \
        --rf-train-end "$TRAIN_END" \
        --output-dir "$eval_dir" \
        $importance_flag
    echo "[$VNAME train_${train_win}m -> eval_${eval_win}m] 評価完了"
}

echo ""
echo "============================================================"
echo "  バリアント $VTYPE: $VNAME"
echo "============================================================"

if [ "$VTYPE" -ge 3 ]; then
    # ── Multi-task: 1モデル訓練 ──
    model_dir="$OUTBASE/$VNAME/train_all"
    mkdir -p "$model_dir"

    if [ -f "$model_dir/irl_model.pt" ]; then
        echo "[$VNAME] 学習スキップ（学習済み）"
    else
        echo "[$VNAME] Multi-task 学習開始..."
        uv run python scripts/train/train_model.py \
            --directory-level \
            --model-type "$VTYPE" \
            --raw-json "${RAW_JSON[@]}" \
            --reviews "$REVIEWS" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --train-start "$TRAIN_START" \
            --train-end "$TRAIN_END" \
            --future-window-start 0 \
            --future-window-end 3 \
            --trajectories-cache "$CACHE_DIR/traj_mt_0-3.pkl" \
            --output "$model_dir"
        echo "[$VNAME] 学習完了"
    fi

    # 評価: 10パターン並列
    model_path="$model_dir/irl_model.pt"
    for ti in 0 1 2 3; do
        for ei in $(seq $ti 3); do
            train_win="${TRAIN_WINDOWS[$ti]}"
            eval_win="${TRAIN_WINDOWS[$ei]}"
            eval_fs="${TRAIN_FS[$ei]}"
            train_fs_val="${TRAIN_FS[$ti]}"
            run_eval "$model_path" "$train_win" "$eval_win" "$eval_fs" "$train_fs_val" &
        done
    done
    echo "[$VNAME] 評価10パターン並列起動、完了待ち..."
    wait

else
    # ── Non-multi-task: 4モデル並列訓練 ──
    for i in 0 1 2 3; do
        win="${TRAIN_WINDOWS[$i]}"
        fs="${TRAIN_FS[$i]}"
        fe="${TRAIN_FE[$i]}"
        model_dir="$OUTBASE/$VNAME/train_${win}m"
        mkdir -p "$model_dir"

        if [ -f "$model_dir/irl_model.pt" ]; then
            echo "[$VNAME train_${win}m] 学習スキップ（学習済み）"
        else
            echo "[$VNAME train_${win}m] 学習開始 (future_window=${fs}-${fe}m)..."
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
                --output "$model_dir" &
        fi
    done
    echo "[$VNAME] 訓練4窓並列起動、完了待ち..."
    wait

    # 評価: 10パターン並列
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
        done
    done
    echo "[$VNAME] 評価10パターン並列起動、完了待ち..."
    wait
fi

echo ""
echo "[$VNAME] 全パターン完了"
