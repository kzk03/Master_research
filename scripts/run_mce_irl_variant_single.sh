#!/bin/bash
# [サーバ用 / MCE-IRL 専用] 1つのバリアントに対して MCE-IRL 学習 + 評価10パターンを実行
#
# run_variant_single.sh の MCE-IRL 版。出力ディレクトリ・軌跡キャッシュ・モデル
# ファイル名はすべて MCE 系で別名を使い、既存 (Focal-supervised) の成果物と
# 衝突しないようにしている。
#
# 既存版との差分:
#   - 訓練スクリプト: scripts/train/train_mce_irl.py
#   - 評価スクリプト: scripts/analyze/eval_mce_irl_path_prediction.py
#   - 軌跡キャッシュ: outputs/mce_irl_trajectory_cache/mce_traj_*.pkl
#   - モデルファイル: mce_irl_model.pt
#   - デフォルト OUTBASE: outputs/mce_irl_variant_comparison_server
#   - MT バリアント (3,4,5) は未対応（MCE-IRL は二値 action 設計のため）
#
# 使い方:
#   bash scripts/run_mce_irl_variant_single.sh <variant_id> <variant_name> [outbase] [gpu_id] [save_importance]
#
# 例:
#   bash scripts/run_mce_irl_variant_single.sh 0 lstm_baseline outputs/mce_irl_variant_comparison_server 0
#   bash scripts/run_mce_irl_variant_single.sh 1 lstm_attention outputs/mce_irl_variant_comparison_server 0
#   bash scripts/run_mce_irl_variant_single.sh 2 transformer    outputs/mce_irl_variant_comparison_server 0

set -e

VTYPE="${1:?Usage: $0 <variant_id> <variant_name> [outbase] [gpu_id] [save_importance]}"
VNAME="${2:?Usage: $0 <variant_id> <variant_name> [outbase] [gpu_id] [save_importance]}"
OUTBASE="${3:-outputs/mce_irl_variant_comparison_server}"
GPU_ID="${4:-0}"
SAVE_IMPORTANCE="${5:-false}"

# MCE-IRL は variant 0/1/2 のみ対応 (multi-task ヘッドは持たない)
case "$VTYPE" in
    0|1|2) ;;
    *)
        echo "ERROR: MCE-IRL は variant 0/1/2 のみ対応 (指定: $VTYPE)" >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"

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

TRAIN_WINDOWS=("0-3" "3-6" "6-9" "9-12")
TRAIN_FS=(0 3 6 9)
TRAIN_FE=(3 6 9 12)

# MCE-IRL 専用キャッシュ (既存 outputs/trajectory_cache とは別物)
CACHE_DIR="outputs/mce_irl_trajectory_cache"
mkdir -p "$CACHE_DIR"

# warm-start 設定 (環境変数で制御):
#   INIT_FROM_BASE=outputs/variant_comparison_server/$VNAME を指定すると、
#   各窓で $INIT_FROM_BASE/train_${win}m/irl_model.pt から重みを load して
#   MCE NLL で fine-tune する (Focal-supervised → MCE-IRL warm-start)。
#   未設定なら cold start。
INIT_FROM_BASE="${INIT_FROM_BASE:-}"
INIT_LR_SCALE="${INIT_LR_SCALE:-0.1}"

# 評価1パターンを実行する関数
run_eval() {
    local model_path="$1"
    local train_win="$2"
    local eval_win="$3"
    local eval_fs="$4"
    local train_fs_val="$5"
    local eval_dir="$OUTBASE/$VNAME/train_${train_win}m/eval_${eval_win}m"
    mkdir -p "$eval_dir"

    if [ -f "$eval_dir/summary_metrics.json" ]; then
        echo "[MCE-IRL $VNAME train_${train_win}m -> eval_${eval_win}m] スキップ（評価済み）"
        return
    fi

    echo "[MCE-IRL $VNAME train_${train_win}m -> eval_${eval_win}m] 評価開始..."
    local importance_flag=""
    if [ "$SAVE_IMPORTANCE" = "true" ]; then
        importance_flag="--save-importance"
    fi
    uv run python scripts/analyze/eval_mce_irl_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$eval_fs" \
        --rf-future-start-months "$train_fs_val" \
        --irl-dir-model "$model_path" \
        --rf-train-end "$TRAIN_END" \
        --device cuda \
        --n-jobs 4 \
        --output-dir "$eval_dir" \
        --calibrate \
        $importance_flag
    echo "[MCE-IRL $VNAME train_${train_win}m -> eval_${eval_win}m] 評価完了"
}

# 評価並列数（同時にバックグラウンドで走る評価プロセス数）
EVAL_PARALLEL=4

echo ""
echo "============================================================"
echo "  [サーバ / MCE-IRL] バリアント $VTYPE: $VNAME (GPU=$GPU_ID)"
echo "  OUTBASE: $OUTBASE"
echo "  CACHE  : $CACHE_DIR"
echo "============================================================"

# ── 軌跡キャッシュを事前生成（窓ごとに 1 度だけ） ──
# extract_mce_trajectories.py は出力先がすでに存在すればスキップする。
# 必ず CACHE_DIR (outputs/mce_irl_trajectory_cache) 配下に書き、
# Focal-supervised の outputs/trajectory_cache/ とは混ざらないようにする。
for i in 0 1 2 3; do
    win="${TRAIN_WINDOWS[$i]}"
    fs="${TRAIN_FS[$i]}"
    fe="${TRAIN_FE[$i]}"
    cache_path="$CACHE_DIR/mce_traj_${win}.pkl"
    if [ -f "$cache_path" ]; then
        echo "[MCE-IRL cache ${win}m] スキップ（キャッシュ済み: $cache_path）"
    else
        echo "[MCE-IRL cache ${win}m] 抽出開始 (future_window=${fs}-${fe}m)..."
        uv run python scripts/train/extract_mce_trajectories.py \
            --reviews "$REVIEWS" \
            --raw-json "${RAW_JSON[@]}" \
            --train-start "$TRAIN_START" \
            --train-end "$TRAIN_END" \
            --future-window-start "$fs" \
            --future-window-end "$fe" \
            --n-jobs -1 \
            --output "$cache_path"
        echo "[MCE-IRL cache ${win}m] 抽出完了 → $cache_path"
    fi
done

# ── 4訓練窓を逐次訓練（GPU 共有のため） ──
for i in 0 1 2 3; do
    win="${TRAIN_WINDOWS[$i]}"
    fs="${TRAIN_FS[$i]}"
    fe="${TRAIN_FE[$i]}"
    model_dir="$OUTBASE/$VNAME/train_${win}m"
    mkdir -p "$model_dir"

    if [ -f "$model_dir/mce_irl_model.pt" ]; then
        echo "[MCE-IRL $VNAME train_${win}m] 学習スキップ（学習済み）"
    else
        # warm-start: $INIT_FROM_BASE が指定されていて該当 checkpoint があれば load
        warm_args=()
        if [ -n "$INIT_FROM_BASE" ]; then
            init_ckpt="$INIT_FROM_BASE/train_${win}m/irl_model.pt"
            if [ -f "$init_ckpt" ]; then
                warm_args=(--init-from "$init_ckpt" --init-lr-scale "$INIT_LR_SCALE")
                echo "[MCE-IRL $VNAME train_${win}m] warm-start from $init_ckpt (LR×$INIT_LR_SCALE)"
            else
                echo "[MCE-IRL $VNAME train_${win}m] WARN: warm-start checkpoint 不在 ($init_ckpt) → cold start"
            fi
        fi
        echo "[MCE-IRL $VNAME train_${win}m] 学習開始 (future_window=${fs}-${fe}m, GPU=$GPU_ID)..."
        uv run python scripts/train/train_mce_irl.py \
            --directory-level \
            --model-type "$VTYPE" \
            --raw-json "${RAW_JSON[@]}" \
            --reviews "$REVIEWS" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --train-start "$TRAIN_START" \
            --train-end "$TRAIN_END" \
            --future-window-start "$fs" \
            --future-window-end "$fe" \
            --trajectories-cache "$CACHE_DIR/mce_traj_${win}.pkl" \
            --output "$model_dir" \
            "${warm_args[@]}"
        echo "[MCE-IRL $VNAME train_${win}m] 学習完了"
    fi
done

# ── 評価: 最大 EVAL_PARALLEL 個ずつ並列実行 ──
running=0
for ti in 0 1 2 3; do
    train_win="${TRAIN_WINDOWS[$ti]}"
    model_path="$OUTBASE/$VNAME/train_${train_win}m/mce_irl_model.pt"

    if [ ! -f "$model_path" ]; then
        echo "[MCE-IRL $VNAME train_${train_win}m] モデルが見つからない、スキップ"
        continue
    fi

    for ei in $(seq $ti 3); do
        eval_win="${TRAIN_WINDOWS[$ei]}"
        eval_fs="${TRAIN_FS[$ei]}"
        train_fs_val="${TRAIN_FS[$ti]}"
        run_eval "$model_path" "$train_win" "$eval_win" "$eval_fs" "$train_fs_val" &
        running=$((running + 1))
        if [ "$running" -ge "$EVAL_PARALLEL" ]; then
            wait
            running=0
        fi
    done
done
if [ "$running" -gt 0 ]; then
    wait
fi
echo "[MCE-IRL $VNAME] 評価全パターン完了"

echo ""
echo "[MCE-IRL $VNAME] 全パターン完了"
