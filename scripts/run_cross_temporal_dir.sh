#!/bin/bash
# 卒論と同じ10パターンのクロス時間評価（ディレクトリ単位）
#
# 訓練期間: 2021-01 ~ 2022-01 (固定)
# 訓練ラベル将来窓: 0-3m, 3-6m, 6-9m, 9-12m (4パターン)
# 評価カットオフ: 2023-01-01 (固定)
# 評価将来窓: 0-3m, 3-6m, 6-9m, 9-12m (訓練<=評価で10パターン)

set -e

OUTBASE="outputs/cross_temporal_dir"
REVIEWS="data/nova_raw.csv"
RAW_JSON="data/raw_json/openstack__nova.json"
TRAIN_START="2021-01-01"
TRAIN_END="2022-01-01"
EVAL_CUTOFF="2023-01-01"
EPOCHS=50
PATIENCE=5

mkdir -p "$OUTBASE"

# 訓練窓の定義
TRAIN_WINDOWS=("0-3" "3-6" "6-9" "9-12")
TRAIN_FS=(0 3 6 9)
TRAIN_FE=(3 6 9 12)

echo "=== Phase 1: IRL_Dir 学習 (4パターン) ==="
for i in 0 1 2 3; do
    win="${TRAIN_WINDOWS[$i]}"
    fs="${TRAIN_FS[$i]}"
    fe="${TRAIN_FE[$i]}"
    model_dir="$OUTBASE/train_${win}m"

    if [ -f "$model_dir/irl_model.pt" ]; then
        echo "[$win] スキップ（学習済み）"
        continue
    fi

    echo "[$win] IRL_Dir 学習開始 (future_window=${fs}-${fe}m)..."
    uv run python scripts/train/train_model.py \
        --directory-level \
        --raw-json "$RAW_JSON" \
        --reviews "$REVIEWS" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --train-start "$TRAIN_START" \
        --train-end "$TRAIN_END" \
        --future-window-start "$fs" \
        --future-window-end "$fe" \
        --output "$model_dir"
    echo "[$win] 学習完了"
done

echo ""
echo "=== Phase 2: 評価 (10パターン, 訓練<=評価) ==="

EVAL_WINDOWS=("0-3" "3-6" "6-9" "9-12")
EVAL_FS=(0 3 6 9)

for ti in 0 1 2 3; do
    train_win="${TRAIN_WINDOWS[$ti]}"
    model_path="$OUTBASE/train_${train_win}m/irl_model.pt"

    if [ ! -f "$model_path" ]; then
        echo "[train_${train_win}m] モデルが見つからない、スキップ"
        continue
    fi

    for ei in 0 1 2 3; do
        # 訓練 <= 評価 の制約
        if [ "$ei" -lt "$ti" ]; then
            continue
        fi

        eval_win="${EVAL_WINDOWS[$ei]}"
        eval_fs="${EVAL_FS[$ei]}"
        eval_dir="$OUTBASE/train_${train_win}m/eval_${eval_win}m"
        mkdir -p "$eval_dir"

        if [ -f "$eval_dir/summary_metrics.json" ]; then
            echo "[train_${train_win}m -> eval_${eval_win}m] スキップ（評価済み）"
            continue
        fi

        echo "[train_${train_win}m -> eval_${eval_win}m] 評価開始..."
        uv run python scripts/analyze/eval_path_prediction.py \
            --prediction-time "$EVAL_CUTOFF" \
            --delta-months 3 \
            --future-start-months "$eval_fs" \
            --irl-dir-model "$model_path" \
            --rf-train-end "$TRAIN_END" \
            --output-dir "$eval_dir"
        echo "[train_${train_win}m -> eval_${eval_win}m] 評価完了"
    done
done

echo ""
echo "=== 全パターン完了 ==="
echo "結果: $OUTBASE/"
