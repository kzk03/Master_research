#!/bin/bash
# 軌跡抽出のみ実行してキャッシュを作成する
# 4つの将来窓 × 2種類（non-MT, MT）= 5ファイル
#
# 出力:
#   $OUTBASE/trajectory_cache/traj_0-3.pkl
#   $OUTBASE/trajectory_cache/traj_3-6.pkl
#   $OUTBASE/trajectory_cache/traj_6-9.pkl
#   $OUTBASE/trajectory_cache/traj_9-12.pkl
#   $OUTBASE/trajectory_cache/traj_mt_0-3.pkl

set -e

OUTBASE="${1:-outputs/variant_combined_v2}"
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

CACHE_DIR="$OUTBASE/trajectory_cache"
mkdir -p "$CACHE_DIR"

WINDOWS=("0-3" "3-6" "6-9" "9-12")
FS=(0 3 6 9)
FE=(3 6 9 12)

echo "=== 軌跡キャッシュ作成開始（5並列） ==="

# Non-multi-task: 4つの将来窓を並列実行
for i in 0 1 2 3; do
    win="${WINDOWS[$i]}"
    echo "[${win}] 軌跡抽出開始..."
    uv run python scripts/train/extract_trajectories.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" \
        --train-end "$TRAIN_END" \
        --future-window-start "${FS[$i]}" \
        --future-window-end "${FE[$i]}" \
        --output "$CACHE_DIR/traj_${win}.pkl" &
done

# Multi-task も並列
echo "[MT] 軌跡抽出開始..."
uv run python scripts/train/extract_trajectories.py \
    --reviews "$REVIEWS" \
    --raw-json "${RAW_JSON[@]}" \
    --train-start "$TRAIN_START" \
    --train-end "$TRAIN_END" \
    --future-window-start 0 \
    --future-window-end 3 \
    --multitask \
    --output "$CACHE_DIR/traj_mt_0-3.pkl" &

echo "5プロセス起動完了。完了を待機中..."
wait

echo ""
echo "=== 軌跡キャッシュ作成完了 ==="
ls -lh "$CACHE_DIR"/*.pkl
