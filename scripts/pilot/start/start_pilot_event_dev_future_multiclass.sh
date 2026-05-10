#!/bin/bash
# パイロット (per-dev × 未来窓 step_labels + マルチクラス accept action) を nohup で
# バックグラウンド起動するヘルパ。
# 使い方: bash scripts/pilot/start/start_pilot_event_dev_future_multiclass.sh

cd "$(dirname "$0")/../../.."
mkdir -p logs
nohup bash scripts/pilot/run/run_pilot_mce_event_dev_future_multiclass.sh \
    > logs/mce_pilot_event_dev_future_multiclass_full.log 2>&1 &
PID=$!
echo "$PID" > logs/mce_pilot_event_dev_future_multiclass_full.pid
echo "started: PID=$PID"
echo "log:     logs/mce_pilot_event_dev_future_multiclass_full.log"
echo ""
echo "確認:    tail -f logs/mce_pilot_event_dev_future_multiclass_full.log"
echo "中断:    pkill -TERM -P $PID; kill $PID"
