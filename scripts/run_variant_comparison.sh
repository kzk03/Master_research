#!/bin/bash
# 6つのIRLバリアント × 10パターンのクロス時間評価
#
# バリアント:
#   0: LSTM (ベースライン)
#   1: LSTM + Attention
#   2: Transformer
#   3: LSTM + Multi-task (4ヘッド)
#   4: LSTM + Attention + Multi-task
#   5: Transformer + Multi-task
#
# Non-multi-task (0,1,2): 4モデル訓練 × 10パターン評価
# Multi-task (3,4,5): 1モデル訓練 × 10パターン評価
# 合計: 4×3 + 1×3 = 15 訓練 + 6×10 = 60 評価

set -e

OUTBASE="outputs/variant_comparison"
REVIEWS="data/nova_raw.csv"
RAW_JSON="data/raw_json/openstack__nova.json"
TRAIN_START="2019-01-01"
TRAIN_END="2022-01-01"
EVAL_CUTOFF="2023-01-01"
EPOCHS=50
PATIENCE=5

VARIANTS=(0 1 2 3 4 5)
VARIANT_NAMES=("lstm_baseline" "lstm_attention" "transformer" "lstm_multitask" "lstm_attn_multitask" "transformer_multitask")

TRAIN_WINDOWS=("0-3" "3-6" "6-9" "9-12")
TRAIN_FS=(0 3 6 9)
TRAIN_FE=(3 6 9 12)

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
                --raw-json "$RAW_JSON" \
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

        # 評価: 10パターン
        model_path="$model_dir/irl_model.pt"
        for ti in 0 1 2 3; do
            for ei in $(seq $ti 3); do
                train_win="${TRAIN_WINDOWS[$ti]}"
                eval_win="${TRAIN_WINDOWS[$ei]}"
                eval_fs="${TRAIN_FS[$ei]}"
                eval_dir="$OUTBASE/$vname/train_${train_win}m/eval_${eval_win}m"
                mkdir -p "$eval_dir"

                if [ -f "$eval_dir/summary_metrics.json" ]; then
                    echo "[$vname train_${train_win}m -> eval_${eval_win}m] スキップ（評価済み）"
                    continue
                fi

                train_fs_val="${TRAIN_FS[$ti]}"
                echo "[$vname train_${train_win}m -> eval_${eval_win}m] 評価開始..."
                uv run python scripts/analyze/eval_path_prediction.py \
                    --prediction-time "$EVAL_CUTOFF" \
                    --delta-months 3 \
                    --future-start-months "$eval_fs" \
                    --rf-future-start-months "$train_fs_val" \
                    --irl-dir-model "$model_path" \
                    --rf-train-end "$TRAIN_END" \
                    --output-dir "$eval_dir"
                echo "[$vname train_${train_win}m -> eval_${eval_win}m] 評価完了"
            done
        done

    else
        # ── Non-multi-task: 4モデル個別訓練 ──
        for i in 0 1 2 3; do
            win="${TRAIN_WINDOWS[$i]}"
            fs="${TRAIN_FS[$i]}"
            fe="${TRAIN_FE[$i]}"
            model_dir="$OUTBASE/$vname/train_${win}m"
            mkdir -p "$model_dir"

            if [ -f "$model_dir/irl_model.pt" ]; then
                echo "[$vname train_${win}m] 学習スキップ（学習済み）"
            else
                echo "[$vname train_${win}m] 学習開始 (future_window=${fs}-${fe}m)..."
                uv run python scripts/train/train_model.py \
                    --directory-level \
                    --model-type "$vtype" \
                    --raw-json "$RAW_JSON" \
                    --reviews "$REVIEWS" \
                    --epochs "$EPOCHS" \
                    --patience "$PATIENCE" \
                    --train-start "$TRAIN_START" \
                    --train-end "$TRAIN_END" \
                    --future-window-start "$fs" \
                    --future-window-end "$fe" \
                    --output "$model_dir"
                echo "[$vname train_${win}m] 学習完了"
            fi
        done

        # 評価: 10パターン
        for ti in 0 1 2 3; do
            train_win="${TRAIN_WINDOWS[$ti]}"
            model_path="$OUTBASE/$vname/train_${train_win}m/irl_model.pt"

            if [ ! -f "$model_path" ]; then
                echo "[$vname train_${train_win}m] モデルが見つからない、スキップ"
                continue
            fi

            for ei in $(seq $ti 3); do
                eval_win="${TRAIN_WINDOWS[$ei]}"
                eval_fs="${TRAIN_FS[$ei]}"
                eval_dir="$OUTBASE/$vname/train_${train_win}m/eval_${eval_win}m"
                mkdir -p "$eval_dir"

                if [ -f "$eval_dir/summary_metrics.json" ]; then
                    echo "[$vname train_${train_win}m -> eval_${eval_win}m] スキップ（評価済み）"
                    continue
                fi

                train_fs_val="${TRAIN_FS[$i]}"
                echo "[$vname train_${train_win}m -> eval_${eval_win}m] 評価開始..."
                uv run python scripts/analyze/eval_path_prediction.py \
                    --prediction-time "$EVAL_CUTOFF" \
                    --delta-months 3 \
                    --future-start-months "$eval_fs" \
                    --rf-future-start-months "$train_fs_val" \
                    --irl-dir-model "$model_path" \
                    --rf-train-end "$TRAIN_END" \
                    --output-dir "$eval_dir"
                echo "[$vname train_${train_win}m -> eval_${eval_win}m] 評価完了"
            done
        done
    fi
done

echo ""
echo "============================================================"
echo "  全バリアント完了"
echo "============================================================"
echo "結果: $OUTBASE/"
