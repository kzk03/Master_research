#!/bin/bash
# パイロット実行 (Plan B-1: per-dev / マルチクラス accept action MCE-IRL)
#
# 既存パイロット (run_pilot_mce_event_dev_comparison.sh) のサイクル:
#   Phase 1: cache 抽出 (per-dev, 二値 step_actions)
#   Phase 2: cold start 学習
#   Phase 3: (dev, dir) AUC + dev レベル AUC 評価
# を引き継ぎつつ、action 空間を {0=reject, 1..K=accept(dir cluster), K+1=other_accept}
# に拡張したマルチクラス版を学習・評価する。
#
# 使い方:
#   nohup bash scripts/pilot/run/run_pilot_mce_event_dev_multiclass.sh \
#       > logs/mce_pilot_event_dev_multiclass_full.log 2>&1 &
#   または:
#   bash scripts/pilot/start/start_pilot_event_dev_multiclass.sh
#
# 出力先: outputs/mce_pilot_event_dev_multiclass/
#   dir_class_mapping_K15.json は outputs/ 直下を共有

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
FUTURE_FS=0
FUTURE_FE=3
EPOCHS=50
PATIENCE=5
DEVICE="cpu"
MAX_EVENTS=256
SLIDING_WINDOW_DAYS=180
TOP_K=15

PILOT_DIR="outputs/mce_pilot_event_dev_multiclass"
CACHE_DIR="$PILOT_DIR/cache"
DIR_CLASS_MAPPING="outputs/dir_class_mapping_K${TOP_K}.json"
EXISTING_BINARY_CACHE="outputs/mce_pilot_event_dev/cache/event_traj_0-3.pkl"

mkdir -p "$CACHE_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# ── Phase 0: dir_class_mapping 生成 ─────────────────────────
banner "Phase 0/4: dir_class_mapping (top-${TOP_K}) 生成"
if [ -f "$DIR_CLASS_MAPPING" ]; then
    echo "skip (already exists: $DIR_CLASS_MAPPING)"
else
    uv run python scripts/train/build_dir_class_mapping.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --top-k "$TOP_K" \
        --output "$DIR_CLASS_MAPPING"
fi

# ── Phase 1: per-dev / マルチクラス cache 抽出 ───────────────
banner "Phase 1/4: per-dev イベント / マルチクラス cache 抽出 (0-3m)"
TARGET_CACHE="$CACHE_DIR/event_traj_0-3.pkl"
if [ -f "$TARGET_CACHE" ]; then
    echo "skip (already exists: $TARGET_CACHE)"
else
    REUSE_FLAGS=()
    if [ -f "$EXISTING_BINARY_CACHE" ]; then
        echo "既存 binary cache を再利用: $EXISTING_BINARY_CACHE"
        REUSE_FLAGS+=(--reuse-binary-cache "$EXISTING_BINARY_CACHE")
    fi
    uv run python scripts/train/extract_mce_event_trajectories_multiclass.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --max-events "$MAX_EVENTS" --sliding-window-days "$SLIDING_WINDOW_DAYS" --n-jobs -1 \
        --per-dev \
        --dir-class-mapping "$DIR_CLASS_MAPPING" \
        "${REUSE_FLAGS[@]}" \
        --output "$TARGET_CACHE"
fi

# ── Phase 2: マルチクラス cold start 学習 ───────────────────
banner "Phase 2/4: マルチクラス MCE-IRL cold start 学習"
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

# ── Phase 3: 評価 ((dev, dir) + dev レベル) ────────────────
banner "Phase 3/4: (dev, dir) + dev レベル評価"
EVAL_OUT="$PILOT_DIR/eval_event_cold"
if [ -f "$EVAL_OUT/summary_metrics.json" ]; then
    echo "skip (already evaluated)"
else
    mkdir -p "$EVAL_OUT"
    uv run python scripts/analyze/eval/eval_mce_event_irl_multiclass_prediction.py \
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

# ── Phase 4: 結果サマリ ────────────────────────────────────
banner "Phase 4/4: 結果サマリ"
echo ""
echo "── multiclass の結果 ──"
if [ -f "$EVAL_OUT/summary_metrics.json" ]; then
    echo "  IRL_Dir (dev, dir):"
    jq '.IRL_Dir | {clf_auc_roc, clf_auc_pr, n_pairs, n_pos, n_neg}' \
        "$EVAL_OUT/summary_metrics.json" 2>/dev/null || \
        cat "$EVAL_OUT/summary_metrics.json"
    echo "  IRL_dev (dev レベル):"
    jq '.IRL_dev | {clf_auc_roc, clf_auc_pr, n_pairs, n_pos, n_neg}' \
        "$EVAL_OUT/summary_metrics.json" 2>/dev/null || true
fi

echo ""
echo "── 比較対象 (既存) ──"
echo "  per-dev event 二値 cold (mce_pilot_event_dev): "
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot_event_dev/eval_event_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"
echo "  per-dev event 二値 cold dev レベル:"
jq '.auc_roc' outputs/mce_pilot_event_dev/eval_dev_level/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"
echo "  月次 cold (mce_pilot):"
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot/eval_monthly_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"

banner "全 Phase 完了 [$(stamp)]"
