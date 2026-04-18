#!/bin/bash
# 10プロジェクト分のデータ収集（Nova以外の9プロジェクト）
# Novaは既に収集済み (data/nova_raw.csv, data/raw_json/openstack__nova.json)
#
# 依存の強さ:
#   強: neutron, cinder
#   中: ironic, glance, keystone
#   弱: horizon, swift, heat, octavia

set -e

GERRIT_URL="https://review.opendev.org"
START_DATE="2020-01-01"
END_DATE="2026-01-01"
RESPONSE_WINDOW=14

PROJECTS=(
    "openstack/neutron"
    "openstack/cinder"
    "openstack/ironic"
    "openstack/glance"
    "openstack/keystone"
    "openstack/horizon"
    "openstack/swift"
    "openstack/heat"
    "openstack/octavia"
)

mkdir -p data/raw_json

for proj in "${PROJECTS[@]}"; do
    # openstack/neutron -> neutron
    name="${proj#openstack/}"
    csv_path="data/${name}_raw.csv"
    json_path="data/raw_json/${proj//\//__}.json"

    if [ -f "$csv_path" ] && [ -f "$json_path" ]; then
        echo "[$name] スキップ（収集済み）"
        continue
    fi

    echo "[$name] データ収集開始..."
    uv run python scripts/pipeline/build_dataset.py \
        --gerrit-url "$GERRIT_URL" \
        --project "$proj" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --response-window "$RESPONSE_WINDOW" \
        --output "$csv_path" \
        --raw-output "$json_path"
    echo "[$name] 完了"
    echo ""
done

echo "=== 全プロジェクト収集完了 ==="
echo "CSVファイル:"
ls -lh data/*_raw.csv
echo ""
echo "JSONファイル:"
ls -lh data/raw_json/*.json
