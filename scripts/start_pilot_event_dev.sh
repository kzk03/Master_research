#!/bin/bash
# パイロット (run_pilot_mce_event_dev_comparison.sh, per-dev イベント単位) を
# nohup でバックグラウンド起動するヘルパ。
# 使い方: bash scripts/start_pilot_event_dev.sh

cd "$(dirname "$0")/.."
mkdir -p logs
nohup bash scripts/run_pilot_mce_event_dev_comparison.sh > logs/mce_pilot_event_dev_full.log 2>&1 &
PID=$!
echo "$PID" > logs/mce_pilot_event_dev_full.pid
echo "started: PID=$PID"
echo "log:     logs/mce_pilot_event_dev_full.log"
echo ""
echo "確認:    tail -f logs/mce_pilot_event_dev_full.log"
echo "中断:    pkill -TERM -P $PID; kill $PID"
