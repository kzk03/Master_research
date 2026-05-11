#!/bin/bash
# パイロット (run_pilot_mce_comparison.sh) を nohup でバックグラウンド起動するヘルパ。
# 使い方: bash scripts/pilot/start/start_pilot.sh

cd "$(dirname "$0")/../../.."
mkdir -p logs
nohup bash scripts/pilot/run/run_pilot_mce_comparison.sh > logs/mce_pilot_full.log 2>&1 &
PID=$!
echo "$PID" > logs/mce_pilot_full.pid
echo "started: PID=$PID"
echo "log:     logs/mce_pilot_full.log"
echo ""
echo "確認:    tail -f logs/mce_pilot_full.log"
echo "中断:    kill $PID"
