#!/bin/bash
# パイロット (per-dev × 未来窓 step_labels + マルチクラス accept action):
# 既存パイロット run_pilot_mce_event_dev_multiclass.sh (B-1) のサイクルを引き継ぎ、
#   - 軌跡: per-dev (全 dir 横断, 不活動時間軸あり)
#   - step_labels: 未来 3ヶ月内に dev が (どこかで) accept したか (dev レベル)
#   - step_actions: 各 event の dir を depth=1 親で multiclass (K=15 + other + reject = 17)
# を組み合わせる。
#
# 既存 per-dev binary cache (outputs/mce_pilot_event_dev/cache/event_traj_0-3.pkl) と
# dir_class_mapping_K15.json を再利用するため、cache 抽出は数分で完了する。
#
# 使い方:
#   nohup bash scripts/run_pilot_mce_event_dev_future_multiclass.sh \
#       > logs/mce_pilot_event_dev_future_multiclass_full.log 2>&1 &
# あるいは:
#   bash scripts/start_pilot_event_dev_future_multiclass.sh
#
# 出力先: outputs/mce_pilot_event_dev_future_multiclass/

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
SLIDING_WINDOW_DAYS=180
TOP_K=15

PILOT_DIR="outputs/mce_pilot_event_dev_future_multiclass"
CACHE_DIR="$PILOT_DIR/cache"
DIR_CLASS_MAPPING="outputs/dir_class_mapping_K${TOP_K}.json"
EXISTING_PER_DEV_CACHE="outputs/mce_pilot_event_dev/cache/event_traj_0-3.pkl"

mkdir -p "$CACHE_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# ── Phase 0: dir_class_mapping 確認 (既存) ───────────────────
banner "Phase 0/4: dir_class_mapping (top-${TOP_K}) 確認"
if [ -f "$DIR_CLASS_MAPPING" ]; then
    echo "skip (既存: $DIR_CLASS_MAPPING)"
else
    uv run python scripts/train/build_dir_class_mapping.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --top-k "$TOP_K" \
        --output "$DIR_CLASS_MAPPING"
fi

# ── Phase 1: per-dev cache から未来窓 + multiclass 化 ────────
banner "Phase 1/4: per-dev cache を未来窓 + multiclass 化 (0-3m)"
TARGET_CACHE="$CACHE_DIR/event_traj_0-3.pkl"
if [ -f "$TARGET_CACHE" ]; then
    echo "skip (already exists: $TARGET_CACHE)"
else
    if [ ! -f "$EXISTING_PER_DEV_CACHE" ]; then
        echo "ERROR: 既存 per-dev cache が見つかりません: $EXISTING_PER_DEV_CACHE"
        echo "  先に bash scripts/run_pilot_mce_event_dev_comparison.sh で生成してください"
        exit 1
    fi
    uv run python scripts/train/extract_mce_event_trajectories_future_multiclass.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --reuse-binary-cache "$EXISTING_PER_DEV_CACHE" \
        --dir-class-mapping "$DIR_CLASS_MAPPING" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --output "$TARGET_CACHE"
fi

# ── Phase 2: マルチクラス cold start 学習 ───────────────────
banner "Phase 2/4: 未来窓 step_labels + multiclass action で MCE-IRL 学習"
TRAIN_OUT="$PILOT_DIR/event_cold"
if [ -f "$TRAIN_OUT/mce_event_irl_model.pt" ]; then
    echo "skip (already trained: $TRAIN_OUT/mce_event_irl_model.pt)"
else
    uv run python scripts/train/train_mce_event_irl_multiclass.py \
        --trajectories-cache "$TARGET_CACHE" \
        --dir-class-mapping "$DIR_CLASS_MAPPING" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --model-type 0 \
        --output "$TRAIN_OUT"
fi

# ── Phase 3: (dev, dir) + dev レベル評価 ────────────────────
banner "Phase 3/4: (dev, dir) + dev レベル評価"
EVAL_OUT="$PILOT_DIR/eval_event_cold"
if [ -f "$EVAL_OUT/summary_metrics.json" ]; then
    echo "skip (already evaluated)"
else
    mkdir -p "$EVAL_OUT"
    uv run python scripts/analyze/eval_mce_event_irl_multiclass_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --model "$TRAIN_OUT/mce_event_irl_model.pt" \
        --dir-class-mapping "$DIR_CLASS_MAPPING" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$FUTURE_FS" \
        --window-days "$SLIDING_WINDOW_DAYS" \
        --device "$DEVICE" \
        --output-dir "$EVAL_OUT"
fi

# ── Phase 4: 結果サマリ ─────────────────────────────────
banner "Phase 4/4: 結果サマリ"
echo ""
echo "── 今回 (per-dev × 未来窓 + multiclass) ──"
if [ -f "$EVAL_OUT/summary_metrics.json" ]; then
    echo "  IRL_Dir (dev, dir):"
    jq '.IRL_Dir | {clf_auc_roc, clf_auc_pr, n_pairs, n_pos, n_neg}' \
        "$EVAL_OUT/summary_metrics.json" 2>/dev/null
    echo "  IRL_dev (dev レベル):"
    jq '.IRL_dev | {clf_auc_roc, clf_auc_pr, n_pairs, n_pos, n_neg}' \
        "$EVAL_OUT/summary_metrics.json" 2>/dev/null
fi

echo ""
echo "── 比較 ──"
echo "  per-dev event 即時 multiclass (B-1):"
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot_event_dev_multiclass/eval_event_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"
echo "  pair-future cold (calibrated):"
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot_event_pair_future/eval_event_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"
echo "  pair-future cold (calibrated, IRL_Dir_calibrated):"
jq '.IRL_Dir_calibrated.clf_auc_roc' outputs/mce_pilot_event_pair_future/eval_event_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"
echo "  月次 cold (mce_pilot, ターゲット):"
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot/eval_monthly_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"

banner "全 Phase 完了 [$(stamp)]"
