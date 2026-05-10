#!/bin/bash
# パイロット v2: 訓練/評価窓を 1 年スライド (Plan A: 純粋スライド)
#
#   v1 (mce_pilot_multiwindow):
#     TRAIN  2019-01-01 → 2022-01-01 (3年)
#     EVAL   prediction_time = 2023-01-01 から 4 窓 (0-3, 3-6, 6-9, 9-12m)
#
#   v2 (本スクリプト):
#     TRAIN  2020-01-01 → 2023-01-01 (3年, 1年後ろにスライド)
#     EVAL   prediction_time = 2024-01-01 から 4 窓 (同上)
#
# 学習期間は 3 年で固定し、時間軸だけずらす（学習量を変えずに直近データの効果を見る）。
# v1 で観測された AUC-PR の concept drift 改善を狙う。
#
# 出力先: outputs/mce_pilot_multiwindow_v2/
#
# 使い方:
#   nohup bash scripts/pilot/run/run_pilot_mce_multiwindow_v2.sh \
#       > logs/mce_pilot_multiwindow_v2_full.log 2>&1 &

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
# v1 から 1 年スライド
TRAIN_START="2020-01-01"
TRAIN_END="2023-01-01"
EVAL_CUTOFF="2024-01-01"
EPOCHS=50
PATIENCE=5
DEVICE="cpu"

PILOT_DIR="outputs/mce_pilot_multiwindow_v2"
# v2 用に専用 cache ディレクトリ（v1 の outputs/trajectory_cache とは混ぜない）
SOURCE_CACHE_DIR="outputs/trajectory_cache_v2"
CACHE_DIR="$PILOT_DIR/cache"
WINDOWS=("0-3" "3-6" "6-9" "9-12")
FS=(0 3 6 9)
FE=(3 6 9 12)

mkdir -p "$SOURCE_CACHE_DIR" "$CACHE_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# ── Phase 0: v2 用 4 窓 cache を生成（並列） ─────────────────
banner "Phase 0/3: v2 軌跡 cache を 4 窓並列で生成 (TRAIN_END=$TRAIN_END)"
pids=()
for i in 0 1 2 3; do
    win="${WINDOWS[$i]}"
    cache_file="$SOURCE_CACHE_DIR/traj_${win}.pkl"
    if [ -f "$cache_file" ]; then
        echo "[${win}] 既存、スキップ: $cache_file"
        continue
    fi
    echo "[${win}] 軌跡抽出開始..."
    uv run python scripts/train/extract_trajectories.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" \
        --train-end "$TRAIN_END" \
        --future-window-start "${FS[$i]}" \
        --future-window-end "${FE[$i]}" \
        --output "$cache_file" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "Phase 0 完了。生成された cache:"
ls -lh "$SOURCE_CACHE_DIR"/*.pkl 2>/dev/null

# ── Phase 1: 4 cache を merge して MCE-IRL 用統合 cache 生成 ──
banner "Phase 1/3: 4 窓 cache を merge"
TARGET_CACHE="$CACHE_DIR/monthly_traj_multiwindow.pkl"
if [ -f "$TARGET_CACHE" ]; then
    echo "skip (既存: $TARGET_CACHE)"
else
    INPUTS=()
    for win in "${WINDOWS[@]}"; do
        INPUTS+=("$SOURCE_CACHE_DIR/traj_${win}.pkl")
    done
    uv run python scripts/train/merge_mce_trajectories_multiwindow.py \
        --inputs "${INPUTS[@]}" \
        --window-labels "${WINDOWS[@]}" \
        --output "$TARGET_CACHE"
fi

# ── Phase 2: 統合 cache で月次 MCE-IRL cold start 学習 ─────
banner "Phase 2/3: 統合 cache で月次 MCE-IRL cold start 学習"
TRAIN_OUT="$PILOT_DIR/monthly_cold_multiwindow"
if [ -f "$TRAIN_OUT/mce_irl_model.pt" ]; then
    echo "skip (既学習: $TRAIN_OUT)"
else
    uv run python scripts/train/train_mce_irl.py \
        --directory-level --model-type 0 \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
        --future-window-start 0 --future-window-end 3 \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --trajectories-cache "$TARGET_CACHE" \
        --skip-threshold \
        --output "$TRAIN_OUT"
fi

# ── Phase 3: 4 窓それぞれで評価 ───────────────────────────
run_eval() {
    local window="$1"
    local fs fe
    fs="${window%-*}"
    fe="${window##*-}"
    local out_dir="$PILOT_DIR/eval_${window}m"
    if [ -f "$out_dir/summary_metrics.json" ]; then
        echo "[eval ${window}m] skip"
        return
    fi
    local model_path="$TRAIN_OUT/mce_irl_model.pt"
    if [ ! -f "$model_path" ]; then
        echo "[eval ${window}m] model not found ($model_path), skip"
        return
    fi
    mkdir -p "$out_dir"
    echo "[eval ${window}m] running... (fs=$fs, fe=$fe, prediction_time=$EVAL_CUTOFF)"
    uv run python scripts/analyze/eval/eval_mce_irl_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$fs" \
        --rf-future-start-months "$fs" \
        --irl-dir-model "$model_path" \
        --rf-train-end "$TRAIN_END" \
        --device "$DEVICE" --n-jobs 4 \
        --output-dir "$out_dir" \
        --calibrate
    echo "[eval ${window}m] done → $out_dir/summary_metrics.json"
}

banner "Phase 3/3: 4 窓 (0-3, 3-6, 6-9, 9-12m) で評価 (prediction_time=$EVAL_CUTOFF)"
for win in "${WINDOWS[@]}"; do
    run_eval "$win"
done

# ── サマリ ────────────────────────────────────────────
banner "結果サマリ"
echo ""
echo "── v2 IRL_Dir clf_auc_roc ──"
for win in "${WINDOWS[@]}"; do
    f="$PILOT_DIR/eval_${win}m/summary_metrics.json"
    if [ -f "$f" ]; then
        v=$(jq -r '.IRL_Dir.clf_auc_roc // "n/a"' "$f")
        echo "  ${win}m: $v"
    fi
done
echo ""
echo "── 比較: v1 IRL_Dir clf_auc_roc ──"
for win in "${WINDOWS[@]}"; do
    f="outputs/mce_pilot_multiwindow/eval_${win}m/summary_metrics.json"
    if [ -f "$f" ]; then
        v=$(jq -r '.IRL_Dir.clf_auc_roc // "n/a"' "$f")
        echo "  ${win}m: $v"
    fi
done

banner "v2 全 Phase 完了 [$(stamp)]"
