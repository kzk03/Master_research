#!/bin/bash
# MCE-IRL を 4 窓それぞれで独立に学習し、対応窓で評価する。
# multiwindow (4 窓統合 1 モデル) との対照実験 (B-26 の追加実験)。
#
# 出力: outputs/mce_pilot_perwindow/train_{0-3,3-6,6-9,9-12}m/
#         ├ mce_irl_model.pt
#         └ eval_{同じ窓}m/summary_metrics.json
#
# 使い方:
#   bash scripts/pilot/run/run_pilot_mce_perwindow.sh
#
# 前提: outputs/trajectory_cache/traj_{0-3,3-6,6-9,9-12}.pkl が存在すること

set -e
cd "$(dirname "$0")/../../.."

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
DEVICE="cpu"
EVAL_N_JOBS=4
PILOT_DIR="outputs/mce_pilot_perwindow"
SOURCE_CACHE_DIR="outputs/trajectory_cache"
WINDOWS=("0-3" "3-6" "6-9" "9-12")

mkdir -p "$PILOT_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# Phase 0: cache 確認
banner "Phase 0/2: 窓別 cache の確認"
for win in "${WINDOWS[@]}"; do
    f="$SOURCE_CACHE_DIR/traj_${win}.pkl"
    if [ ! -f "$f" ]; then
        echo "ERROR: $f が存在しません。先に scripts/extract_trajectories_cache.sh で生成してください。"
        exit 1
    fi
    echo "  [${win}] ✓ $f ($(du -h "$f" | awk '{print $1}'))"
done

# Phase 1: 窓ごとに学習 + 評価
for win in "${WINDOWS[@]}"; do
    fs="${win%-*}"
    fe="${win##*-}"
    train_out="$PILOT_DIR/train_${win}m"
    cache="$SOURCE_CACHE_DIR/traj_${win}.pkl"
    eval_out="$train_out/eval_${win}m"

    banner "Phase 1/2: train ${win}m (fs=$fs, fe=$fe)"
    if [ -f "$train_out/mce_irl_model.pt" ]; then
        echo "skip (既学習: $train_out)"
    else
        uv run python scripts/train/train_mce_irl.py \
            --directory-level --model-type 0 \
            --reviews "$REVIEWS" \
            --raw-json "${RAW_JSON[@]}" \
            --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
            --future-window-start "$fs" --future-window-end "$fe" \
            --epochs "$EPOCHS" --patience "$PATIENCE" \
            --trajectories-cache "$cache" \
            --skip-threshold \
            --output "$train_out"
    fi

    banner "Phase 2/2: eval ${win}m (prediction_time=$EVAL_CUTOFF)"
    if [ -f "$eval_out/summary_metrics.json" ]; then
        echo "skip (既評価: $eval_out)"
    else
        mkdir -p "$eval_out"
        uv run python scripts/analyze/eval/eval_mce_irl_path_prediction.py \
            --data "$REVIEWS" \
            --raw-json "${RAW_JSON[@]}" \
            --prediction-time "$EVAL_CUTOFF" \
            --delta-months 3 \
            --future-start-months "$fs" \
            --rf-future-start-months "$fs" \
            --irl-dir-model "$train_out/mce_irl_model.pt" \
            --rf-train-end "$TRAIN_END" \
            --device "$DEVICE" --n-jobs "$EVAL_N_JOBS" \
            --output-dir "$eval_out" \
            --calibrate
    fi
done

# サマリ
banner "結果サマリ (IRL_Dir clf_auc_roc)"
for win in "${WINDOWS[@]}"; do
    f="$PILOT_DIR/train_${win}m/eval_${win}m/summary_metrics.json"
    if [ -f "$f" ]; then
        v=$(jq -r '.IRL_Dir.clf_auc_roc // "n/a"' "$f")
        echo "  ${win}m: $v"
    fi
done

banner "全 Phase 完了 [$(stamp)]"
