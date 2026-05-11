#!/bin/bash
# Plan A-3 (月次 MCE-IRL × 4 訓練窓統合) パイロットを nohup で
# バックグラウンド起動するヘルパ。
#
# 使い方: bash scripts/pilot/start/start_pilot_multiwindow.sh
#
# 前提: outputs/trajectory_cache/traj_{0-3,3-6,6-9,9-12}.pkl が存在すること
#       (なければ scripts/extract_trajectories_cache.sh で先に生成)

cd "$(dirname "$0")/../../.."
mkdir -p logs

# 二重起動防止: 既存の PID ファイルがあり、そのプロセスがまだ生きていれば停止
PID_FILE="logs/mce_pilot_multiwindow_full.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && [ -d "/proc/$OLD_PID" ]; then
        echo "ERROR: 既に PID=$OLD_PID で実行中です。"
        echo "       終了するなら: kill $OLD_PID  (確実に殺すなら: kill -9 $OLD_PID)"
        echo "       強制再起動するなら: rm $PID_FILE して再実行"
        exit 1
    fi
fi

# ログを上書きせず追記モードに変更しつつ、開始マーカーを入れる
LOG_FILE="logs/mce_pilot_multiwindow_full.log"
{
    echo ""
    echo "============================================================"
    echo "  start_pilot_multiwindow.sh: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
} >> "$LOG_FILE"

nohup bash scripts/pilot/run/run_pilot_mce_multiwindow.sh \
    >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
echo "started: PID=$PID"
echo "log:     $LOG_FILE"
echo ""
echo "確認:    tail -f $LOG_FILE"
echo "中断:    kill $PID  (子プロセスごと: pkill -TERM -P $PID; kill $PID)"
