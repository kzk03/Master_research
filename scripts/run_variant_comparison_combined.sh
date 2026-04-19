#!/bin/bash
# 10プロジェクト統合データセットでの6バリアント比較
# まず1パターン（train_0-3m → eval_0-3m）のみ実行
#
# バリアント:
#   0: LSTM (ベースライン)
#   1: LSTM + Attention
#   2: Transformer
#   3: LSTM + Multi-task (4ヘッド)
#   4: LSTM + Attention + Multi-task
#   5: Transformer + Multi-task

set -e

OUTBASE="outputs/variant_comparison_combined"
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

VARIANTS=(0 1 2 3 4 5)
VARIANT_NAMES=("lstm_baseline" "lstm_attention" "transformer" "lstm_multitask" "lstm_attn_multitask" "transformer_multitask")

# 1パターンのみ: train_0-3m → eval_0-3m
TRAIN_FS=0
TRAIN_FE=3
EVAL_FS=0
TRAIN_WIN="0-3"
EVAL_WIN="0-3"

mkdir -p "$OUTBASE"

for vi in "${!VARIANTS[@]}"; do
    vtype="${VARIANTS[$vi]}"
    vname="${VARIANT_NAMES[$vi]}"

    echo ""
    echo "============================================================"
    echo "  バリアント $vtype: $vname"
    echo "============================================================"

    if [ "$vtype" -ge 3 ]; then
        # ── Multi-task: 1モデルで全窓をカバー ──
        model_dir="$OUTBASE/$vname/train_all"
        mkdir -p "$model_dir"

        if [ -f "$model_dir/irl_model.pt" ]; then
            echo "[$vname] 学習スキップ（学習済み）"
        else
            echo "[$vname] Multi-task 学習開始..."
            uv run python scripts/train/train_model.py \
                --directory-level \
                --model-type "$vtype" \
                --raw-json "${RAW_JSON[@]}" \
                --reviews "$REVIEWS" \
                --epochs "$EPOCHS" \
                --patience "$PATIENCE" \
                --train-start "$TRAIN_START" \
                --train-end "$TRAIN_END" \
                --future-window-start 0 \
                --future-window-end 3 \
                --output "$model_dir"
            echo "[$vname] 学習完了"
        fi

        # 評価: 1パターンのみ
        model_path="$model_dir/irl_model.pt"
        eval_dir="$OUTBASE/$vname/train_${TRAIN_WIN}m/eval_${EVAL_WIN}m"
        mkdir -p "$eval_dir"

        if [ -f "$eval_dir/summary_metrics.json" ]; then
            echo "[$vname train_${TRAIN_WIN}m -> eval_${EVAL_WIN}m] スキップ（評価済み）"
        else
            echo "[$vname train_${TRAIN_WIN}m -> eval_${EVAL_WIN}m] 評価開始..."
            uv run python scripts/analyze/eval_path_prediction.py \
                --data "$REVIEWS" \
                --raw-json "${RAW_JSON[@]}" \
                --prediction-time "$EVAL_CUTOFF" \
                --delta-months 3 \
                --future-start-months "$EVAL_FS" \
                --rf-future-start-months "$TRAIN_FS" \
                --irl-dir-model "$model_path" \
                --rf-train-end "$TRAIN_END" \
                --output-dir "$eval_dir"
            echo "[$vname train_${TRAIN_WIN}m -> eval_${EVAL_WIN}m] 評価完了"
        fi

    else
        # ── Non-multi-task: 1モデル訓練 ──
        model_dir="$OUTBASE/$vname/train_${TRAIN_WIN}m"
        mkdir -p "$model_dir"

        if [ -f "$model_dir/irl_model.pt" ]; then
            echo "[$vname train_${TRAIN_WIN}m] 学習スキップ（学習済み）"
        else
            echo "[$vname train_${TRAIN_WIN}m] 学習開始 (future_window=${TRAIN_FS}-${TRAIN_FE}m)..."
            uv run python scripts/train/train_model.py \
                --directory-level \
                --model-type "$vtype" \
                --raw-json "${RAW_JSON[@]}" \
                --reviews "$REVIEWS" \
                --epochs "$EPOCHS" \
                --patience "$PATIENCE" \
                --train-start "$TRAIN_START" \
                --train-end "$TRAIN_END" \
                --future-window-start "$TRAIN_FS" \
                --future-window-end "$TRAIN_FE" \
                --output "$model_dir"
            echo "[$vname train_${TRAIN_WIN}m] 学習完了"
        fi

        # 評価: 1パターンのみ
        model_path="$model_dir/irl_model.pt"
        eval_dir="$OUTBASE/$vname/train_${TRAIN_WIN}m/eval_${EVAL_WIN}m"
        mkdir -p "$eval_dir"

        if [ -f "$eval_dir/summary_metrics.json" ]; then
            echo "[$vname train_${TRAIN_WIN}m -> eval_${EVAL_WIN}m] スキップ（評価済み）"
        else
            echo "[$vname train_${TRAIN_WIN}m -> eval_${EVAL_WIN}m] 評価開始..."
            uv run python scripts/analyze/eval_path_prediction.py \
                --data "$REVIEWS" \
                --raw-json "${RAW_JSON[@]}" \
                --prediction-time "$EVAL_CUTOFF" \
                --delta-months 3 \
                --future-start-months "$EVAL_FS" \
                --rf-future-start-months "$TRAIN_FS" \
                --irl-dir-model "$model_path" \
                --rf-train-end "$TRAIN_END" \
                --output-dir "$eval_dir"
            echo "[$vname train_${TRAIN_WIN}m -> eval_${EVAL_WIN}m] 評価完了"
        fi
    fi
done

echo ""
echo "============================================================"
echo "  全バリアント完了"
echo "============================================================"
echo "結果: $OUTBASE/"
