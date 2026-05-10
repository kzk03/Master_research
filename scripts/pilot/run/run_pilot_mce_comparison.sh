#!/bin/bash
# パイロット実行: train_0-3m → eval_0-3m の 1 パターンで
#   - 月次 MCE-IRL (cold start)
#   - 月次 MCE-IRL (warm start from Focal-supervised lstm_baseline)
# を比較し、Focal baseline (既存 outputs/variant_comparison_server/lstm_baseline/
# train_0-3m/eval_0-3m/summary_metrics.json) との AUC 差を測る。
# (イベント単位は GPU 不在のため本パイロットでは対象外)
#
# 出力: outputs/mce_pilot/
#   cache/                 軌跡キャッシュ
#   monthly_cold/          月次 cold モデル
#   monthly_warm/          月次 warm モデル
#   eval_monthly_cold/     月次 cold 評価結果 (summary_metrics.json)
#   eval_monthly_warm/
#
# 使い方:
#   nohup bash scripts/pilot/run/run_pilot_mce_comparison.sh > logs/mce_pilot_full.log 2>&1 &

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

PILOT_DIR="outputs/mce_pilot"
CACHE_DIR="$PILOT_DIR/cache"
mkdir -p "$CACHE_DIR" logs

FOCAL_CKPT="outputs/variant_comparison_server/lstm_baseline/train_0-3m/irl_model.pt"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# ── Phase 1: 月次 cache 抽出 ───────────────────────────────
banner "Phase 1/4: 月次 cache 抽出 (0-3m)"
if [ -f "$CACHE_DIR/monthly_traj_0-3.pkl" ]; then
    echo "skip (already exists)"
else
    uv run python scripts/train/extract_mce_trajectories.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --n-jobs -1 \
        --output "$CACHE_DIR/monthly_traj_0-3.pkl"
fi

# ── Phase 2: 月次 cold start 学習 ──────────────────────────
banner "Phase 2/4: 月次 MCE-IRL cold start 学習"
if [ -f "$PILOT_DIR/monthly_cold/mce_irl_model.pt" ]; then
    echo "skip (already trained)"
else
    uv run python scripts/train/train_mce_irl.py \
        --directory-level --model-type 0 \
        --raw-json "${RAW_JSON[@]}" --reviews "$REVIEWS" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --trajectories-cache "$CACHE_DIR/monthly_traj_0-3.pkl" \
        --output "$PILOT_DIR/monthly_cold"
fi

# ── Phase 3: 月次 warm start 学習 ─────────────────────────
banner "Phase 3/4: 月次 MCE-IRL warm start 学習 (init from Focal baseline)"
if [ -f "$PILOT_DIR/monthly_warm/mce_irl_model.pt" ]; then
    echo "skip (already trained)"
elif [ ! -f "$FOCAL_CKPT" ]; then
    echo "WARN: Focal baseline checkpoint not found ($FOCAL_CKPT) → skip warm-start"
else
    uv run python scripts/train/train_mce_irl.py \
        --directory-level --model-type 0 \
        --raw-json "${RAW_JSON[@]}" --reviews "$REVIEWS" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start "$FUTURE_FS" --future-window-end "$FUTURE_FE" \
        --trajectories-cache "$CACHE_DIR/monthly_traj_0-3.pkl" \
        --init-from "$FOCAL_CKPT" --init-lr-scale 0.1 \
        --output "$PILOT_DIR/monthly_warm"
fi

# ── Phase 4: 評価 (2 モデル × 1 パターン) ──────────────────
run_eval() {
    local label="$1"
    local model_path="$2"
    local eval_script="$3"
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
    uv run python "$eval_script" \
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

banner "Phase 4/4: 評価 (2 モデル)"
run_eval monthly_cold "$PILOT_DIR/monthly_cold/mce_irl_model.pt" \
         scripts/analyze/eval/eval_mce_irl_path_prediction.py
run_eval monthly_warm "$PILOT_DIR/monthly_warm/mce_irl_model.pt" \
         scripts/analyze/eval/eval_mce_irl_path_prediction.py

banner "全 Phase 完了 [$(stamp)]"
echo "次は: jq でサマリ抽出 → 利点欠点判定"
echo "  jq '{IRL_Dir}' $PILOT_DIR/eval_*/summary_metrics.json"
