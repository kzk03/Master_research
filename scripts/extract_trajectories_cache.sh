#!/bin/bash
# [サーバ用] 軌跡抽出のみ実行してキャッシュを作成する
# MT を除外し、non-MT 4窓を並列実行
#
# 出力:
#   outputs/trajectory_cache/traj_0-3.pkl
#   outputs/trajectory_cache/traj_3-6.pkl
#   outputs/trajectory_cache/traj_6-9.pkl
#   outputs/trajectory_cache/traj_9-12.pkl

set -e

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

CACHE_DIR="outputs/trajectory_cache"
mkdir -p "$CACHE_DIR"

WINDOWS=("0-3" "3-6" "6-9" "9-12")
FS=(0 3 6 9)
FE=(3 6 9 12)

echo "=== [サーバ] 軌跡キャッシュ作成開始（4並列） ==="

for i in 0 1 2 3; do
    win="${WINDOWS[$i]}"
    cache_file="$CACHE_DIR/traj_${win}.pkl"
    if [ -f "$cache_file" ]; then
        echo "[${win}] キャッシュ済み、スキップ: $cache_file"
        continue
    fi
    echo "[${win}] 軌跡抽出開始..."
    uv run python scripts/train/extract_trajectories.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" \
        --train-end "$TRAIN_END" \
        --future-window-start "${FS[$i]}" \
        --future-window-end "${FE[$i]}" \
        --output "$cache_file" &
done

echo "プロセス起動完了。完了を待機中..."
wait

echo ""
echo "=== 軌跡キャッシュ作成完了 ==="
ls -lh "$CACHE_DIR"/*.pkl 2>/dev/null || echo "(キャッシュファイルなし)"
