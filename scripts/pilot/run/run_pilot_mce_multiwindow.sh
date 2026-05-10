#!/bin/bash
# パイロット (Plan A-3: 月次 MCE-IRL × 4 訓練窓統合)
#
#   Phase 1: 4 窓 (0-3, 3-6, 6-9, 9-12m) の月次軌跡 cache を統合
#            (既に extract_trajectories_cache.sh で生成済みの cache を再利用)
#   Phase 2: 統合 cache で月次 MCE-IRL cold start 学習 (二値 action)
#   Phase 3: 4 窓それぞれで (dev, dir) AUC 評価
#
# サーバ前提 (~24GB pickle ロードを許容するメモリ環境)。
# ローカル 16GB マシンでは OOM の可能性あり、その場合は
# scripts/train/extend_mce_monthly_trajectories_multiwindow.py で 1 cache 版を試すこと。
#
# 出力先: outputs/mce_pilot_multiwindow/
#
# 使い方:
#   nohup bash scripts/pilot/run/run_pilot_mce_multiwindow.sh \
#       > logs/mce_pilot_multiwindow_full.log 2>&1 &
# あるいは:
#   bash scripts/pilot/start/start_pilot_multiwindow.sh

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
EPOCHS=50
PATIENCE=5
DEVICE="cpu"

PILOT_DIR="outputs/mce_pilot_multiwindow"
CACHE_DIR="$PILOT_DIR/cache"
SOURCE_CACHE_DIR="outputs/trajectory_cache"
WINDOWS=("0-3" "3-6" "6-9" "9-12")

mkdir -p "$CACHE_DIR" logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "========================================"; echo "  [$(stamp)] $*"; echo "========================================"; }

# ── Phase 0: 既存 4 窓 cache の存在確認 ────────────────────
banner "Phase 0/3: 既存 4 窓 cache の存在確認"
MISSING=0
for win in "${WINDOWS[@]}"; do
    f="$SOURCE_CACHE_DIR/traj_${win}.pkl"
    if [ -f "$f" ]; then
        size_gb=$(du -h "$f" | awk '{print $1}')
        echo "  [${win}] ✓ $f ($size_gb)"
    else
        echo "  [${win}] ✗ $f が存在しません"
        MISSING=$((MISSING+1))
    fi
done
if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING 個の cache が不足しています。"
    echo "  先に bash scripts/extract_trajectories_cache.sh を実行して 4 窓分の cache を生成してください。"
    exit 1
fi

# ── Phase 1: 4 cache を merge して統合 cache 生成 ──────────
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
# train_mce_irl.py は mce_irl_model.pt を保存する (model_class='mce_irl', state_dim=23)
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
    local model_path
    if [ -f "$TRAIN_OUT/mce_irl_model.pt" ]; then
        model_path="$TRAIN_OUT/mce_irl_model.pt"
    else
        echo "[eval ${window}m] model not found ($TRAIN_OUT/mce_irl_model.pt), skip"
        return
    fi
    mkdir -p "$out_dir"
    echo "[eval ${window}m] running... (fs=$fs, fe=$fe)"
    # 月次 MCE-IRL (state_dim=23) 用の評価器を使用。
    # eval_mce_event_irl_path_prediction.py は state_dim=27 専用なので使わない。
    # --irl-model は意図的に未指定 (デフォルトの存在しないパス) → global IRL flow は
    # スキップされ、ディレクトリ単位 (IRL_Dir) のみ評価される。
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

banner "Phase 3/3: 4 窓 (0-3, 3-6, 6-9, 9-12m) で評価"
for win in "${WINDOWS[@]}"; do
    run_eval "$win"
done

# ── サマリ ────────────────────────────────────────────
banner "結果サマリ"
echo ""
echo "── multiwindow IRL_Dir clf_auc_roc ──"
for win in "${WINDOWS[@]}"; do
    f="$PILOT_DIR/eval_${win}m/summary_metrics.json"
    if [ -f "$f" ]; then
        v=$(jq -r '.IRL_Dir.clf_auc_roc // "n/a"' "$f")
        echo "  ${win}m: $v"
    fi
done
echo ""
echo "── 比較 (既存月次 cold, 単一 0-3m 学習) ──"
jq '.IRL_Dir.clf_auc_roc' outputs/mce_pilot/eval_monthly_cold/summary_metrics.json 2>/dev/null \
    || echo "  (not available)"

banner "全 Phase 完了 [$(stamp)]"
