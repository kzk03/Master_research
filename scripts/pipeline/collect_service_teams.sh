#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenStack 33 service teams 配下の 244 repo を Gerrit から一括収集
# (245 governance 配下 service team 全 repo − 1 sunbeam-charms = 244)
#
# サーバ実行向けに以下の安全機構を備える:
#   - 並列 worker 起動を stagger（先頭から同時に API を叩かない）
#   - 各 repo 完了後に cooldown sleep（Gerrit への負荷軽減）
#   - エラー時は追加 sleep でバックオフ
#   - 既取得 repo (data/raw_json/openstack__X.json + data/raw_csv/openstack__X.csv)
#     は自動 SKIP → 再実行で残り分のみ取得
#   - SIGINT/SIGTERM で子プロセスを掃除して終了
#   - 全実行ログを logs/collect_service_teams/_summary_*.log にも記録
#
# 使い方（サーバ・バックグラウンド推奨）:
#   nohup bash scripts/pipeline/collect_service_teams.sh > /tmp/collect.log 2>&1 &
#   tail -f /tmp/collect.log
#
# 引数:
#   $1: tier フィルタ（カンマ区切り、空なら全 tier）   default: ""
#       選択肢: 大 / 中 / 小 / 極小 / 非分類
#       例) 大,中
#   $2: 並列度                                       default: 4
#   $3: 入力 CSV パス                                 default: data/service_teams_repos.csv
#
# 失敗 repo の個別再実行例:
#   uv run python scripts/pipeline/build_dataset.py \
#       --gerrit-url https://review.opendev.org \
#       --project openstack/zun --start-date 2020-01-01 --end-date 2026-01-01 \
#       --response-window 14 \
#       --output data/raw_csv/openstack__zun.csv \
#       --raw-output data/raw_json/openstack__zun.json
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -u

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GERRIT_URL="https://review.opendev.org"
START_DATE="2020-01-01"
END_DATE="2026-01-01"
RESPONSE_WINDOW=14

# レート制限対策（Gerrit API への配慮）
STAGGER_DELAY=3        # 並列 worker の起動間隔（秒）。最初に N 並列を一斉起動しないため
SLEEP_PER_REPO=5       # 1 repo 完了後の cooldown sleep（秒）
SLEEP_ON_ERROR=20      # エラー終了時の追加 sleep（秒）

# CLI 引数
TIER_FILTER="${1:-}"
PARALLEL="${2:-4}"
INPUT_CSV="${3:-data/service_teams_repos.csv}"

# 出力ディレクトリとログ
LOG_DIR="logs/collect_service_teams"
SUMMARY_LOG="${LOG_DIR}/_summary_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data/raw_json data/raw_csv "$LOG_DIR"

# stdout & summary ログの両方に流す
exec > >(tee -a "$SUMMARY_LOG") 2>&1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 中断ハンドラ（SIGINT/SIGTERM で子プロセス掃除）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cleanup() {
    echo ""
    echo "[$(date -Iseconds 2>/dev/null || date)] === 中断検出。子プロセスを終了します ==="
    pkill -P $$ 2>/dev/null || true
    sleep 2
    pkill -9 -P $$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 開始バナー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "============================================================================"
echo "[$(date -Iseconds 2>/dev/null || date)] Collection START"
echo "  TIER_FILTER : '${TIER_FILTER:-all}'"
echo "  PARALLEL    : $PARALLEL"
echo "  STAGGER     : ${STAGGER_DELAY}s between worker starts"
echo "  COOLDOWN    : ${SLEEP_PER_REPO}s after each repo (success)"
echo "  ERROR SLEEP : ${SLEEP_ON_ERROR}s after each repo (failure)"
echo "  INPUT_CSV   : $INPUT_CSV"
echo "  SUMMARY LOG : $SUMMARY_LOG"
echo "  PID         : $$"
echo "============================================================================"

if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: $INPUT_CSV が存在しません" >&2
    exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 対象 repo を CSV から抽出（sunbeam は無条件で除外）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mapfile -t REPOS < <(python3 -c "
import csv
tier_filter = set('$TIER_FILTER'.split(',')) if '$TIER_FILTER' else None
with open('$INPUT_CSV') as f:
    for r in csv.DictReader(f):
        # excluded_reason 列が空でなければスキップ（sunbeam / retired xstatic-* など）
        if r.get('excluded_reason'):
            continue
        # 旧データ互換: excluded_reason 列がない場合のフォールバック
        if r['team'] == 'sunbeam':
            continue
        if tier_filter and r['tier'] not in tier_filter:
            continue
        print(f\"{r['tier']}|{r['team']}|{r['repo']}\")
")

TOTAL=${#REPOS[@]}

# 既取得カウント
EXISTING=0
for line in "${REPOS[@]}"; do
    repo="${line##*|}"
    safe="${repo//\//__}"
    if [ -f "data/raw_json/${safe}.json" ] && [ -f "data/raw_csv/${safe}.csv" ]; then
        ((EXISTING++))
    fi
done
TODO=$((TOTAL - EXISTING))

echo "対象 repo 件数: $TOTAL"
echo "既取得 (SKIP) : $EXISTING"
echo "新規取得      : $TODO"
echo ""

if [ "$TODO" -eq 0 ]; then
    echo "新規取得対象なし。終了します。"
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# fetch_one : 並列実行される単位
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fetch_one() {
    local line="$1"
    local tier="${line%%|*}"
    local rest="${line#*|}"
    local team="${rest%%|*}"
    local repo="${rest##*|}"
    local repo_safe="${repo//\//__}"
    local json_path="data/raw_json/${repo_safe}.json"
    local csv_path="data/raw_csv/${repo_safe}.csv"
    local log_path="${LOG_DIR}/${repo_safe}.log"

    if [ -f "$json_path" ] && [ -f "$csv_path" ]; then
        echo "[$(date +%H:%M:%S)] [SKIP]  $repo"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] [START] $repo (team=$team, tier=$tier)"
    local started=$(date +%s)

    uv run python scripts/pipeline/build_dataset.py \
        --gerrit-url "$GERRIT_URL" \
        --project "$repo" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --response-window "$RESPONSE_WINDOW" \
        --output "$csv_path" \
        --raw-output "$json_path" \
        > "$log_path" 2>&1

    local rc=$?
    local elapsed=$(( $(date +%s) - started ))

    if [ $rc -eq 0 ]; then
        local size
        size=$(stat -c%s "$json_path" 2>/dev/null || stat -f%z "$json_path" 2>/dev/null || echo 0)
        echo "[$(date +%H:%M:%S)] [DONE]  $repo (${elapsed}s, ${size} bytes)"
        sleep "$SLEEP_PER_REPO"
    else
        echo "[$(date +%H:%M:%S)] [FAIL]  $repo (rc=$rc, see $log_path)  ← cooldown ${SLEEP_ON_ERROR}s"
        sleep "$SLEEP_ON_ERROR"
    fi
    return $rc
}

export -f fetch_one
export GERRIT_URL START_DATE END_DATE RESPONSE_WINDOW LOG_DIR
export SLEEP_PER_REPO SLEEP_ON_ERROR STAGGER_DELAY

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 並列実行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB_START=$(date +%s)

if command -v parallel >/dev/null 2>&1; then
    echo "[INFO] GNU parallel で実行 (--delay $STAGGER_DELAY --jobs $PARALLEL)"
    printf '%s\n' "${REPOS[@]}" | \
        parallel -j "$PARALLEL" \
                 --delay "$STAGGER_DELAY" \
                 --line-buffer \
                 --halt soon,fail=20 \
                 fetch_one
    PARALLEL_RC=$?
else
    echo "[INFO] xargs -P で実行（GNU parallel 未インストール、stagger は randomized sleep で代替）"
    # 各ジョブ先頭でランダム sleep を入れて起動を分散
    printf '%s\n' "${REPOS[@]}" | \
        xargs -I {} -P "$PARALLEL" bash -c '
            sleep $((RANDOM % STAGGER_DELAY + 1))
            fetch_one "$@"
        ' _ {}
    PARALLEL_RC=$?
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 完了サマリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOB_END=$(date +%s)
ELAPSED=$((JOB_END - JOB_START))
HOUR=$((ELAPSED / 3600))
MIN=$((ELAPSED % 3600 / 60))
SEC=$((ELAPSED % 60))

ACTUAL=$(ls data/raw_json/ 2>/dev/null | wc -l)

echo ""
echo "============================================================================"
echo "[$(date -Iseconds 2>/dev/null || date)] Collection END (rc=$PARALLEL_RC)"
echo "  経過時間: ${HOUR}h ${MIN}m ${SEC}s"
echo "  raw_json/ 配下 repo 数: $ACTUAL"

# 失敗ログをサマリに出す
FAILED_LOGS=$(grep -lE "Traceback|Exception|HTTPError" "$LOG_DIR"/openstack__*.log 2>/dev/null || true)
if [ -n "$FAILED_LOGS" ]; then
    FAIL_COUNT=$(echo "$FAILED_LOGS" | wc -l)
    echo ""
    echo "失敗の疑いがあるログ: $FAIL_COUNT 件（詳細は各ログ参照）"
    echo "$FAILED_LOGS" | head -30
    if [ "$FAIL_COUNT" -gt 30 ]; then
        echo "  ... 他 $((FAIL_COUNT - 30)) 件"
    fi
else
    echo "失敗ログ: なし"
fi
echo "============================================================================"
