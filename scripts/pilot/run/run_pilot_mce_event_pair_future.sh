#!/bin/bash
# パイロット (イベント単位 (dev, dir) ペア / 未来窓 step_labels):
# 月次 IRL の step_labels 定義 (「月 t 末から先 3ヶ月以内に accept したか」) を
# イベント時刻に適用したバリアント。
#
# 既存パイロット run_pilot_mce_event_comparison.sh のサイクルを引き継ぎ、
#   Phase 1: cache 抽出 → 既存 cache を再利用して step_labels を未来窓化
#   Phase 2: 二値 MCE-IRL cold start 学習
#   Phase 3: (dev, dir) AUC 評価 (eval_mce_event_irl_path_prediction.py)
# を実行する。
#
# 既存 (dev, dir) ペア cache (outputs/mce_pilot_event/cache/event_traj_0-3.pkl) を
# 再利用するため、Phase 1 は数分の後処理だけで完了する。
#
# 使い方:
#   nohup bash scripts/pilot/run/run_pilot_mce_event_pair_future.sh \
#       > logs/mce_pilot_event_pair_future_full.log 2>&1 &
# あるいは:
#   bash scripts/pilot/start/start_pilot_event_pair_future.sh

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
EPOCHS=150
PATIENCE=15
DEVICE="cpu"

PILOT_DIR="outputs/mce_pilot_event_pair_future"
CACHE_DIR="$PILOT_DIR/cache"
EXISTING_PAIR_CACHE="outputs/mce_pilot_event/cache/event_traj_0-3.pkl"

mkdir -p "$CACHE_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# ── Phase 1: 未来窓 step_labels 化 cache 生成 ──────────────
banner "Phase 1/3: (dev, dir) ペア cache を未来窓化 (0-3m)"
TARGET_CACHE="$CACHE_DIR/event_traj_0-3.pkl"
if [ -f "$TARGET_CACHE" ]; then
    echo "skip (already exists: $TARGET_CACHE)"
else
    if [ ! -f "$EXISTING_PAIR_CACHE" ]; then
        echo "ERROR: 既存 (dev, dir) ペア cache が見つかりません: $EXISTING_PAIR_CACHE"
        echo "  先に bash scripts/pilot/run/run_pilot_mce_event_comparison.sh で生成してください"
        exit 1
    fi
    uv run python scripts/train/extract_mce_event_trajectories_future.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --reuse-binary-cache "$EXISTING_PAIR_CACHE" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --output "$TARGET_CACHE"
fi

# ── Phase 2: 二値 MCE-IRL cold start 学習 ─────────────────
banner "Phase 2/3: 未来窓 step_labels で二値 MCE-IRL cold start 学習"
TRAIN_OUT="$PILOT_DIR/event_cold"
if [ -f "$TRAIN_OUT/mce_event_irl_model.pt" ]; then
    echo "skip (already trained: $TRAIN_OUT/mce_event_irl_model.pt)"
else
    uv run python scripts/train/train_mce_event_irl.py \
        --directory-level --model-type 0 \
        --raw-json "${RAW_JSON[@]}" --reviews "$REVIEWS" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --trajectories-cache "$TARGET_CACHE" \
        --output "$TRAIN_OUT"
fi

# ── Phase 3: (dev, dir) 評価 ─────────────────────────────
run_eval() {
    local label="$1"
    local model_path="$2"
    local out_dir="$PILOT_DIR/eval_$label"
    if [ -f "$out_dir/summary_metrics.json" ]; then
        echo "[eval $label] skip (already evaluated)"
        return
    fi
    if [ ! -f "$model_path" ]; then
        echo "[eval $label] model not found: $model_path → skip"
        return
    fi
    mkdir -p "$out_dir"
    echo "[eval $label] running..."
    uv run python scripts/analyze/eval/eval_mce_event_irl_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$FUTURE_FS" \
        --rf-future-start-months "$FUTURE_FS" \
        --irl-dir-model "$model_path" \
        --rf-train-end "$TRAIN_END" \
        --device "$DEVICE" --n-jobs 4 \
        --output-dir "$out_dir" \
        --calibrate
    echo "[eval $label] done → $out_dir/summary_metrics.json"
}

banner "Phase 3/3: 評価"
run_eval event_cold "$TRAIN_OUT/mce_event_irl_model.pt"

# ── 結果サマリ ───────────────────────────────────────
banner "結果サマリ"
echo ""
echo "── 未来窓 step_labels (今回) ──"
if [ -f "$PILOT_DIR/eval_event_cold/summary_metrics.json" ]; then
    echo "  IRL_Dir.clf_auc_roc:"
    jq '.IRL_Dir.clf_auc_roc' "$PILOT_DIR/eval_event_cold/summary_metrics.json" 2>/dev/null
fi
echo ""
echo "── 比較 ──"
echo "  即時 binary (旧, 0-3m mce_pilot_event):"
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot_event/eval_event_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"
echo "  per-dev event multiclass B-1 (今回の前):"
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot_event_dev_multiclass/eval_event_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"
echo "  月次 cold (mce_pilot, ターゲット):"
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot/eval_monthly_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"

banner "全 Phase 完了 [$(stamp)]"
