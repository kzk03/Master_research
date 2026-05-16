#!/bin/bash
# ============================================================
#  MCE-IRL パイプライン一括実行: filter → 学習 → 評価
# ============================================================
# プロジェクト集合 (scope) を 1 引数で指定するだけで、
#   1. 必要なら combined_raw_<tag>.csv をフィルタ生成
#   2. MCE-IRL バリアントを学習 (4 訓練窓)
#   3. 10 パターンの評価
# まで run_mce_irl_variant_single.sh 経由で実行する。
#
# 使い方:
#   bash scripts/run_mce_pipeline.sh <scope> [outbase] [gpu_id] [save_importance]
#
# scope:
#   main32          → filter_combined.py --main           (32 main repos, 推奨デフォルト)
#   tier-<T1,T2,..> → filter_combined.py --tier <T1,T2>   (例: tier-大, tier-大,中)
#   teams-<t1,..>   → filter_combined.py --teams <t1,..>  (例: teams-nova,neutron)
#   all             → フィルタなし、combined_raw_231.csv を直接使用
#   <既存tag>       → data/combined_raw_<tag>.csv が既にあればそのまま流用
#
# 例:
#   bash scripts/run_mce_pipeline.sh main32
#   bash scripts/run_mce_pipeline.sh tier-大 outputs/tier_大_mce 0
#   bash scripts/run_mce_pipeline.sh all     outputs/all_mce     0
#
# 本スクリプトはデフォルトで variant 0 (LSTM) のみ実行する。
# variant 1 (LSTM+Attention) / 2 (Transformer) は下部でコメントアウトしてあるので、
# 必要なら復元すること。

set -e

SCOPE="${1:?Usage: $0 <scope> [outbase] [gpu_id] [save_importance]}"
OUTBASE_ARG="${2:-}"
GPU_ID="${3:-0}"
SAVE_IMPORTANCE="${4:-false}"

# ── scope → (TAG, FILTER_ARGS) ──
FILTER_ARGS=""
case "$SCOPE" in
    main32)
        TAG="main32"
        FILTER_ARGS="--main"
        ;;
    tier-*)
        T="${SCOPE#tier-}"
        TAG="tier_${T//,/_}"
        FILTER_ARGS="--tier $T"
        ;;
    teams-*)
        T="${SCOPE#teams-}"
        TAG="teams_${T//,/_}"
        FILTER_ARGS="--teams $T"
        ;;
    all)
        TAG="231"
        FILTER_ARGS=""
        ;;
    *)
        # 既存タグとして扱う (data/combined_raw_<scope>.csv が存在すれば使う)
        TAG="$SCOPE"
        FILTER_ARGS=""
        ;;
esac

REVIEWS="data/combined_raw_${TAG}.csv"
RAW_JSON_LIST="data/combined_raw_${TAG}.raw_json_list.txt"

# ── filter (必要なら) ──
if [ ! -f "$REVIEWS" ]; then
    if [ -z "$FILTER_ARGS" ]; then
        echo "ERROR: $REVIEWS が無く、自動フィルタ条件も指定されていません。" >&2
        echo "  scope=$SCOPE に対応する scripts/pipeline/filter_combined.py を手動で実行してください。" >&2
        exit 1
    fi
    echo "[pipeline] filter: $FILTER_ARGS → $REVIEWS"
    uv run python scripts/pipeline/filter_combined.py $FILTER_ARGS
else
    echo "[pipeline] reuse existing: $REVIEWS"
fi

# ── raw_json リストの準備 ──
if [ "$SCOPE" = "all" ]; then
    # 231 repos すべての raw_json をリストアップして一時ファイルに保存
    RAW_JSON_LIST="data/combined_raw_231.raw_json_list.txt"
    if [ ! -f "$RAW_JSON_LIST" ]; then
        ls data/raw_json/openstack__*.json | tr '\n' ' ' > "$RAW_JSON_LIST"
        echo " " >> "$RAW_JSON_LIST"
        echo "[pipeline] generated: $RAW_JSON_LIST"
    fi
fi

if [ ! -f "$RAW_JSON_LIST" ]; then
    echo "ERROR: $RAW_JSON_LIST が見つかりません。filter を再実行してください。" >&2
    exit 1
fi

# ── 出力ディレクトリ ──
OUTBASE="${OUTBASE_ARG:-outputs/${TAG}_mce}"
mkdir -p "$OUTBASE"

echo ""
echo "============================================================"
echo "  MCE-IRL pipeline"
echo "    scope     : $SCOPE  (tag=$TAG)"
echo "    reviews   : $REVIEWS"
echo "    raw_json  : $(wc -w < "$RAW_JSON_LIST") files"
echo "    outbase   : $OUTBASE"
echo "    gpu_id    : $GPU_ID"
echo "============================================================"
echo ""

# variant スクリプトに環境変数で渡す
# CACHE_TAG: 軌跡キャッシュを scope ごとに分離するためのキー (TAG をそのまま使う)
#            外部から CACHE_TAG を export しておけばそちらが優先される (特徴量定義を
#            変えたときに旧キャッシュを残しつつ新キャッシュを別ディレクトリに作る用)。
export REVIEWS RAW_JSON_LIST_FILE="$RAW_JSON_LIST" CACHE_TAG="${CACHE_TAG:-$TAG}"

# ── variant 0: LSTM (デフォルト、基本これだけ) ──
bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline "$OUTBASE" "$GPU_ID" "$SAVE_IMPORTANCE"

# ── variant 1: LSTM + Attention (基本オフ、必要なら復元) ──
# bash scripts/variant/run_mce_irl_variant_single.sh 1 lstm_attention "$OUTBASE" "$GPU_ID"

# ── variant 2: Transformer (基本オフ、必要なら復元) ──
# bash scripts/variant/run_mce_irl_variant_single.sh 2 transformer "$OUTBASE" "$GPU_ID"

echo ""
echo "[pipeline] DONE: $OUTBASE"
