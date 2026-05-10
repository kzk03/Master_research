#!/bin/bash
# パイロット (混在 dev only): per-dev cache から step_labels が混在 (全 1/全 0 でない)
# dev のみフィルタしたデータで cold 学習。dir 条件付け信号を濃くした学習が
# (dev, dir) AUC を改善するかを検証する。
#
# 前提: outputs/mce_pilot_event_dev_mixed/cache/event_traj_0-3.pkl が
#       filter_mixed_variance_cache.py で生成済みであること。
#
# 出力: outputs/mce_pilot_event_dev_mixed/
#   event_cold/            混在 dev only cold モデル
#   eval_event_cold/       (dev, dir) ペア評価結果
#
# 使い方:
#   nohup bash scripts/pilot/run/run_pilot_mce_event_dev_mixed.sh > logs/mce_pilot_event_dev_mixed.log 2>&1 &

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
TRAIN_START="2019-01-01"
TRAIN_END="2022-01-01"
EVAL_CUTOFF="2023-01-01"
FUTURE_FS=0
FUTURE_FE=3
EPOCHS=50
PATIENCE=5
DEVICE="cpu"

PILOT_DIR="outputs/mce_pilot_event_dev_mixed"
CACHE="$PILOT_DIR/cache/event_traj_0-3.pkl"
mkdir -p "$PILOT_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

if [ ! -f "$CACHE" ]; then
    echo "ERROR: cache が存在しません: $CACHE"
    echo "先に filter_mixed_variance_cache.py を実行してください。"
    exit 1
fi

# ── Phase 1: 混在 dev only cold 学習 ──────────────────────
banner "Phase 1/2: 混在 dev only cold 学習"
if [ -f "$PILOT_DIR/event_cold/mce_event_irl_model.pt" ]; then
    echo "skip (already trained)"
else
    uv run python scripts/train/train_mce_event_irl.py \
        --directory-level --model-type 0 \
        --raw-json "${RAW_JSON[@]}" --reviews "$REVIEWS" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --trajectories-cache "$CACHE" \
        --output "$PILOT_DIR/event_cold"
fi

# ── Phase 2: (dev, dir) 評価 ──────────────────────────────
banner "Phase 2/2: (dev, dir) ペア評価"
if [ -f "$PILOT_DIR/eval_event_cold/summary_metrics.json" ]; then
    echo "skip (already evaluated)"
else
    mkdir -p "$PILOT_DIR/eval_event_cold"
    uv run python scripts/analyze/eval/eval_mce_event_irl_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$FUTURE_FS" \
        --rf-future-start-months "$FUTURE_FS" \
        --irl-dir-model "$PILOT_DIR/event_cold/mce_event_irl_model.pt" \
        --rf-train-end "$TRAIN_END" \
        --device "$DEVICE" --n-jobs 4 \
        --output-dir "$PILOT_DIR/eval_event_cold" \
        --calibrate
fi

# ── 追加: dev レベル評価も同時に走らせる ─────────────────
banner "追加: dev レベル評価 (per-dev 学習目標と整合)"
if [ -f "$PILOT_DIR/eval_dev_level/summary_metrics.json" ]; then
    echo "skip"
else
    uv run python scripts/analyze/eval/eval_mce_event_irl_dev_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --model "$PILOT_DIR/event_cold/mce_event_irl_model.pt" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --output-dir "$PILOT_DIR/eval_dev_level"
fi

banner "全 Phase 完了 [$(stamp)]"
echo "結果比較:"
echo "  混在 only (今回):"
echo "    (dev, dir):  jq '.IRL_Dir.clf_auc_roc' $PILOT_DIR/eval_event_cold/summary_metrics.json"
echo "    dev レベル: jq '.auc_roc' $PILOT_DIR/eval_dev_level/summary_metrics.json"
echo "  全 dev (前回):"
echo "    (dev, dir):  jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot_event_dev/eval_event_cold/summary_metrics.json"
echo "    dev レベル: jq '.auc_roc' outputs/mce_pilot_event_dev/eval_dev_level/summary_metrics.json"
