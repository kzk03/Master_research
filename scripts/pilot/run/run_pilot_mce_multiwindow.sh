#!/bin/bash
# パイロット (月次 MCE-IRL × 4 訓練窓統合) 統合版。
# 旧 v1 / v2 / tmp の 3 スクリプトを CLI フラグで切替可能な 1 本に統合した。
#
#   Phase 0: 4 窓 (0-3, 3-6, 6-9, 9-12m) の月次軌跡 cache を確認
#            --auto-cache 指定時は不足分を並列で自動生成
#   Phase 1: 4 cache を merge して統合 cache 生成
#   Phase 2: 統合 cache で月次 MCE-IRL cold start 学習 (二値 action)
#   Phase 3: 各窓で (dev, dir) AUC 評価
#            --parallel-eval 指定時は窓をプロセス並列で評価
#
# サーバ前提 (~24GB pickle ロードを許容するメモリ環境)。
#
# 使い方 (v1 相当: TRAIN 2019-2022, EVAL 2023+, 既存 cache 前提):
#   bash scripts/pilot/run/run_pilot_mce_multiwindow.sh
#   # あるいは: bash scripts/pilot/start/start_pilot_multiwindow.sh
#
# 使い方 (v2 相当: TRAIN 2020-2023, EVAL 2024+, cache 自動生成):
#   bash scripts/pilot/run/run_pilot_mce_multiwindow.sh \
#       --train-start 2020-01-01 --train-end 2023-01-01 \
#       --eval-cutoff 2024-01-01 \
#       --pilot-dir outputs/mce_pilot_multiwindow_v2 \
#       --source-cache-dir outputs/trajectory_cache_v2 \
#       --auto-cache
#
# 使い方 (tmp 相当: 残窓だけ並列で resume):
#   bash scripts/pilot/run/run_pilot_mce_multiwindow.sh \
#       --windows "6-9 9-12" --parallel-eval --eval-suffix _tmp --eval-n-jobs 1

set -e
cd "$(dirname "$0")/../../.."

# ── デフォルト設定 (v1 相当) ───────────────────────────────
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
PILOT_DIR="outputs/mce_pilot_multiwindow"
SOURCE_CACHE_DIR="outputs/trajectory_cache"
WINDOWS=("0-3" "3-6" "6-9" "9-12")
AUTO_CACHE=0
PARALLEL_EVAL=0
EVAL_SUFFIX=""
EVAL_N_JOBS=4

# ── CLI パース ──────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --train-start) TRAIN_START="$2"; shift 2 ;;
        --train-end) TRAIN_END="$2"; shift 2 ;;
        --eval-cutoff) EVAL_CUTOFF="$2"; shift 2 ;;
        --pilot-dir) PILOT_DIR="$2"; shift 2 ;;
        --source-cache-dir) SOURCE_CACHE_DIR="$2"; shift 2 ;;
        --windows) read -r -a WINDOWS <<< "$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --auto-cache) AUTO_CACHE=1; shift ;;
        --parallel-eval) PARALLEL_EVAL=1; shift ;;
        --eval-suffix) EVAL_SUFFIX="$2"; shift 2 ;;
        --eval-n-jobs) EVAL_N_JOBS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

CACHE_DIR="$PILOT_DIR/cache"
TRAIN_OUT="$PILOT_DIR/monthly_cold_multiwindow"
TARGET_CACHE="$CACHE_DIR/monthly_traj_multiwindow.pkl"

mkdir -p "$CACHE_DIR" "$SOURCE_CACHE_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# ── Phase 0: 4 窓 cache の確認 (必要なら生成) ───────────────
banner "Phase 0/3: 4 窓 cache の確認 (auto_cache=$AUTO_CACHE)"
if [ "$AUTO_CACHE" -eq 1 ]; then
    pids=()
    for win in "${WINDOWS[@]}"; do
        cache_file="$SOURCE_CACHE_DIR/traj_${win}.pkl"
        if [ -f "$cache_file" ]; then
            echo "  [${win}] 既存、スキップ: $cache_file"
            continue
        fi
        fs="${win%-*}"
        fe="${win##*-}"
        echo "  [${win}] 軌跡抽出開始..."
        uv run python scripts/train/extract_trajectories.py \
            --reviews "$REVIEWS" \
            --raw-json "${RAW_JSON[@]}" \
            --train-start "$TRAIN_START" \
            --train-end "$TRAIN_END" \
            --future-window-start "$fs" \
            --future-window-end "$fe" \
            --output "$cache_file" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
else
    MISSING=0
    for win in "${WINDOWS[@]}"; do
        f="$SOURCE_CACHE_DIR/traj_${win}.pkl"
        if [ -f "$f" ]; then
            size=$(du -h "$f" | awk '{print $1}')
            echo "  [${win}] ✓ $f ($size)"
        else
            echo "  [${win}] ✗ $f が存在しません"
            MISSING=$((MISSING+1))
        fi
    done
    if [ "$MISSING" -gt 0 ]; then
        echo ""
        echo "ERROR: $MISSING 個の cache が不足しています。"
        echo "  bash scripts/extract_trajectories_cache.sh で先に生成するか、--auto-cache を指定してください。"
        exit 1
    fi
fi

# ── Phase 1: 統合 cache 生成 ──────────────────────────────
banner "Phase 1/3: 4 窓 cache を merge"
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

# ── Phase 2: 月次 MCE-IRL cold start 学習 ───────────────────
banner "Phase 2/3: 統合 cache で月次 MCE-IRL cold start 学習"
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

# ── Phase 3: 各窓で評価 ──────────────────────────────────
run_eval() {
    local window="$1"
    local fs="${window%-*}"
    local fe="${window##*-}"
    local out_dir="$PILOT_DIR/eval_${window}m${EVAL_SUFFIX}"
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
        --device "$DEVICE" --n-jobs "$EVAL_N_JOBS" \
        --output-dir "$out_dir" \
        --calibrate
    echo "[eval ${window}m] done → $out_dir/summary_metrics.json"
}

banner "Phase 3/3: ${#WINDOWS[@]} 窓で評価 (parallel=$PARALLEL_EVAL)"
if [ "$PARALLEL_EVAL" -eq 1 ]; then
    pids=()
    for win in "${WINDOWS[@]}"; do
        run_eval "$win" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
else
    for win in "${WINDOWS[@]}"; do
        run_eval "$win"
    done
fi

# ── サマリ ─────────────────────────────────────────────
banner "結果サマリ"
echo ""
echo "── multiwindow IRL_Dir clf_auc_roc ──"
for win in "${WINDOWS[@]}"; do
    f="$PILOT_DIR/eval_${win}m${EVAL_SUFFIX}/summary_metrics.json"
    if [ -f "$f" ]; then
        v=$(jq -r '.IRL_Dir.clf_auc_roc // "n/a"' "$f")
        echo "  ${win}m: $v"
    fi
done

banner "全 Phase 完了 [$(stamp)]"
