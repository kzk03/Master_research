#!/bin/bash
# パイロット (イベント単位 (dev, dir) ペア / 未来窓 step_labels) を nohup で
# バックグラウンド起動するヘルパ。
# 使い方: bash scripts/pilot/start/start_pilot_event_pair_future.sh

cd "$(dirname "$0")/../../.."
mkdir -p logs
nohup bash scripts/pilot/run/run_pilot_mce_event_pair_future.sh \
    > logs/mce_pilot_event_pair_future_full.log 2>&1 &
PID=$!
echo "$PID" > logs/mce_pilot_event_pair_future_full.pid
echo "started: PID=$PID"
echo "log:     logs/mce_pilot_event_pair_future_full.log"
echo ""
echo "確認:    tail -f logs/mce_pilot_event_pair_future_full.log"
echo "中断:    pkill -TERM -P $PID; kill $PID"
