#!/bin/bash
# パイロット実行 (イベント単位): train_0-3m → eval_0-3m の 1 パターンで
#   - イベント単位 MCE-IRL (cold start, state_dim=27)
# を学習・評価する。
#
# 月次パイロット (run_pilot_mce_comparison.sh) と同じテストセット・同じ
# メトリクスで比較できるため、Focal baseline / 月次 MCE-IRL cold/warm
# との fair comparison が可能。
#
# warm-start は state_dim=27 ↔ 23 の不整合により Focal baseline からは
# 不可能（同じイベント単位で学習済の checkpoint があれば --init-from で渡せる）。
#
# 出力: outputs/mce_pilot_event/
#   cache/                 イベント軌跡キャッシュ
#   event_cold/            イベント cold モデル
#   eval_event_cold/       イベント cold 評価結果 (summary_metrics.json)
#
# 使い方:
#   nohup bash scripts/pilot/run/run_pilot_mce_event_comparison.sh > logs/mce_pilot_event_full.log 2>&1 &
# あるいは:
#   bash scripts/pilot/start/start_pilot_event.sh

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

PILOT_DIR="outputs/mce_pilot_event"
CACHE_DIR="$PILOT_DIR/cache"
mkdir -p "$CACHE_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# ── Phase 1: イベント cache 抽出 ───────────────────────────
banner "Phase 1/3: イベント cache 抽出 (0-3m)"
if [ -f "$CACHE_DIR/event_traj_0-3.pkl" ]; then
    echo "skip (already exists)"
else
    uv run python scripts/train/extract_mce_event_trajectories.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --max-events "$MAX_EVENTS" --sliding-window-days "$SLIDING_WINDOW_DAYS" --n-jobs -1 \
        --output "$CACHE_DIR/event_traj_0-3.pkl"
fi

# ── Phase 2: イベント cold start 学習 ──────────────────────
banner "Phase 2/3: イベント MCE-IRL cold start 学習"
if [ -f "$PILOT_DIR/event_cold/mce_event_irl_model.pt" ]; then
    echo "skip (already trained)"
else
    uv run python scripts/train/train_mce_event_irl.py \
        --directory-level --model-type 0 \
        --raw-json "${RAW_JSON[@]}" --reviews "$REVIEWS" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --trajectories-cache "$CACHE_DIR/event_traj_0-3.pkl" \
        --output "$PILOT_DIR/event_cold"
fi

# ── Phase 3: 評価 (1 モデル × 1 パターン) ──────────────────
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

banner "Phase 3/3: 評価 (1 モデル)"
run_eval event_cold "$PILOT_DIR/event_cold/mce_event_irl_model.pt"

banner "全 Phase 完了 [$(stamp)]"
echo "次は: jq でサマリ抽出 → 月次 cold/warm および Focal baseline と比較"
echo "  jq '.IRL_Dir.clf_auc_roc' $PILOT_DIR/eval_event_cold/summary_metrics.json"
echo "  jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot/eval_monthly_cold/summary_metrics.json"
echo "  jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot/eval_monthly_warm/summary_metrics.json"
echo "  jq '.IRL_Dir.clf_auc_roc' outputs/variant_comparison_server/lstm_baseline/train_0-3m/eval_0-3m/summary_metrics.json"
