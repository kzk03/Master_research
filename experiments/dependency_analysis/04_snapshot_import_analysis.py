"""
コードスナップショットからの静的依存解析

OpenStack の git リポジトリを clone し、各時間窓の終了時点の
スナップショットから import 文を解析してプロジェクト間・
モジュール間の依存グラフを構築する。

前提:
- experiments/dependency_analysis/repos/ に対象リポジトリを clone 済み
- clone コマンド例:
    git clone https://opendev.org/openstack/nova repos/nova
    git clone https://opendev.org/openstack/neutron repos/neutron
    ...

使い方:
    uv run python experiments/dependency_analysis/04_snapshot_import_analysis.py \
        --repos-dir experiments/dependency_analysis/repos \
        --raw-json data/raw_json/openstack__nova.json \
        --snapshot-date 2025-03-31 \
        --output-dir experiments/dependency_analysis/results
"""

import ast
import argparse
import json
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import pandas as pd


# OpenStack プロジェクト名の対応
PROJECT_MAP = {
    "nova": "openstack/nova",
    "neutron": "openstack/neutron",
    "cinder": "openstack/cinder",
    "glance": "openstack/glance",
    "keystone": "openstack/keystone",
    "heat": "openstack/heat",
    "horizon": "openstack/horizon",
    "ironic": "openstack/ironic",
    "swift": "openstack/swift",
    "octavia": "openstack/octavia",
}

OPENSTACK_MODULES = set(PROJECT_MAP.keys())


def find_commit_at_date(repo_path: Path, date_str: str) -> str | None:
    """指定日時点の最新コミットハッシュを取得"""
    try:
        result = subprocess.run(
            ["git", "log", f"--until={date_str}", "--format=%H", "-1"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit if commit else None
    except Exception:
        return None


def checkout_snapshot(repo_path: Path, commit_hash: str):
    """指定コミットに checkout"""
    subprocess.run(
        ["git", "checkout", commit_hash, "--quiet"],
        cwd=repo_path,
        check=True,
    )


def restore_head(repo_path: Path):
    """HEAD に戻す"""
    subprocess.run(
        ["git", "checkout", "-", "--quiet"],
        cwd=repo_path,
        check=True,
    )


def extract_imports(py_file: Path) -> list[str]:
    """Python ファイルから import モジュール名を抽出"""
    try:
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(py_file))
    except (SyntaxError, ValueError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


def analyze_snapshot(repo_path: Path, project_name: str) -> dict:
    """1つのプロジェクトスナップショットの import 解析"""
    py_files = list(repo_path.rglob("*.py"))
    print(f"  {project_name}: {len(py_files)} Python files")

    # module -> imported OpenStack modules
    internal_deps = defaultdict(Counter)  # dir -> {other_project_module: count}
    cross_project_deps = Counter()  # other_project -> count

    for py_file in py_files:
        rel_path = py_file.relative_to(repo_path)
        top_dir = str(rel_path.parts[0]) if len(rel_path.parts) > 1 else "(root)"

        imports = extract_imports(py_file)
        for imp in imports:
            # クロスプロジェクト依存（他の OpenStack プロジェクトを import）
            if imp in OPENSTACK_MODULES and imp != project_name:
                cross_project_deps[imp] += 1
                internal_deps[top_dir][imp] += 1

    return {
        "project": project_name,
        "n_py_files": len(py_files),
        "cross_project_deps": dict(cross_project_deps),
        "internal_deps": {k: dict(v) for k, v in internal_deps.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repos-dir",
        type=Path,
        required=True,
        help="Directory containing cloned repos (e.g., repos/nova, repos/neutron)",
    )
    parser.add_argument(
        "--snapshot-dates",
        nargs="+",
        default=["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"],
        help="Dates for snapshots (time window boundaries)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 各プロジェクト × 各時間窓でスナップショット解析
    all_results = []

    for date_str in args.snapshot_dates:
        print(f"\n=== Snapshot at {date_str} ===")

        for project_name in sorted(PROJECT_MAP.keys()):
            repo_path = args.repos_dir / project_name
            if not repo_path.exists():
                print(f"  {project_name}: SKIPPED (not cloned)")
                continue

            commit = find_commit_at_date(repo_path, date_str)
            if not commit:
                print(f"  {project_name}: no commit found before {date_str}")
                continue

            print(f"  {project_name}: checking out {commit[:8]}...")
            checkout_snapshot(repo_path, commit)

            try:
                result = analyze_snapshot(repo_path, project_name)
                result["snapshot_date"] = date_str
                result["commit"] = commit
                all_results.append(result)
            finally:
                restore_head(repo_path)

    # 結果を保存
    output_path = args.output_dir / "snapshot_import_deps.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # クロスプロジェクト依存のサマリ
    print("\n=== Cross-project dependency summary ===")
    for result in all_results:
        deps = result.get("cross_project_deps", {})
        if deps:
            top_deps = sorted(deps.items(), key=lambda x: -x[1])[:5]
            dep_str = ", ".join(f"{k}({v})" for k, v in top_deps)
            print(f"  {result['snapshot_date']} {result['project']}: {dep_str}")


if __name__ == "__main__":
    main()
