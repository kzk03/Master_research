"""
コードスナップショットからのディレクトリ単位静的依存解析

各プロジェクトを指定日時点にチェックアウトし、AST 解析で
ディレクトリ間の import エッジを構築する。

co-change グラフ (01_cochange_graph.py) と同じ 2階層ディレクトリ命名規則
（boilerplate 除外、project本体は2階層、その他は1階層）で出力するため、
top_cochange_pairs.csv / hub_scores.csv と直接 join 可能。

Usage:
    uv run python experiments/dependency_analysis/06_static_dir_deps.py \
        --repos-dir experiments/dependency_analysis/repos \
        --snapshot-date 2022-12-31 \
        --output-dir experiments/dependency_analysis/results
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECTS = [
    "nova", "neutron", "cinder", "glance", "keystone",
    "heat", "horizon", "ironic", "swift", "octavia",
]

# co-change graph と完全に同じ集合
BOILERPLATE_DIRS = {
    "releasenotes", "doc", "(root)", "tools", "etc",
    "api-ref", "api-guide", "devstack", "playbooks", "roles",
    "zuul.d", ".zuul.d", "contrib", "rally-jobs", "specs", "bin",
}


def _file_dir(parts: tuple[str, ...], project_name: str) -> str | None:
    """ファイルパス (e.g. 'nova/compute/api.py') 用 — 末尾はファイル名なので len>=3 で 2階層"""
    if len(parts) < 2:
        return None
    top = parts[0]
    if top in BOILERPLATE_DIRS:
        return None
    if top == project_name and len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    return top


def _module_dir(parts: tuple[str, ...], project_name: str) -> str | None:
    """モジュール名 (e.g. ('nova','compute')) 用 — 末尾はファイル名でないので len>=2 で 2階層"""
    if not parts:
        return None
    top = parts[0]
    if top in BOILERPLATE_DIRS:
        return None
    if top == project_name and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return top


def file_to_module_parts(rel_path: Path) -> list[str]:
    """nova/compute/api.py → ['nova','compute','api']
       nova/compute/__init__.py → ['nova','compute']"""
    parts = list(rel_path.parts)
    if not parts:
        return []
    last = parts[-1]
    if not last.endswith(".py"):
        return []
    stem = last[:-3]
    if stem == "__init__":
        return parts[:-1]
    return parts[:-1] + [stem]


def resolve_imports(
    tree: ast.AST, file_module_parts: list[str]
) -> list[tuple[str, ...]]:
    """AST から (top, sub, ...) の dotted parts を抽出（絶対 + 相対解決済み）"""
    out: list[tuple[str, ...]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    out.append(tuple(alias.name.split(".")))
        elif isinstance(node, ast.ImportFrom):
            level = node.level
            mod = node.module or ""
            if level == 0:
                if not mod:
                    continue
                base = mod.split(".")
            else:
                if level > len(file_module_parts):
                    continue
                base_pkg = file_module_parts[:-level]
                if not base_pkg:
                    continue
                base = list(base_pkg) + (mod.split(".") if mod else [])
                if not base:
                    continue
            # 各 imported name を base に append して emit (深い dst が取れる)
            name_emitted = False
            for alias in node.names:
                if alias.name and alias.name != "*":
                    out.append(tuple(base + [alias.name]))
                    name_emitted = True
            if not name_emitted:
                out.append(tuple(base))
    return out


def analyze_repo(repo_path: Path, project_name: str) -> dict:
    edges: dict[tuple[str, str], int] = defaultdict(int)
    n_files = 0
    n_parse_err = 0

    for py_file in repo_path.rglob("*.py"):
        try:
            rel = py_file.relative_to(repo_path)
        except ValueError:
            continue

        src_dir = _file_dir(rel.parts, project_name)
        if src_dir is None:
            continue

        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, ValueError):
            n_parse_err += 1
            continue

        n_files += 1
        file_mod = file_to_module_parts(rel)
        for imp_parts in resolve_imports(tree, file_mod):
            dst_dir = _module_dir(imp_parts, project_name)
            if dst_dir is None or dst_dir == src_dir:
                continue
            edges[(src_dir, dst_dir)] += 1

    return {
        "project": project_name,
        "n_py_files": n_files,
        "n_parse_err": n_parse_err,
        "edges": dict(edges),
    }


def checkout_at(repo_path: Path, date_str: str) -> str | None:
    res = subprocess.run(
        ["git", "log", f"--until={date_str}", "--format=%H", "-1"],
        cwd=repo_path, capture_output=True, text=True,
    )
    commit = res.stdout.strip()
    if not commit:
        return None
    subprocess.run(
        ["git", "checkout", commit, "--quiet"],
        cwd=repo_path, check=True,
    )
    return commit


def restore_head(repo_path: Path) -> None:
    subprocess.run(
        ["git", "checkout", "-", "--quiet"],
        cwd=repo_path, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-dir", type=Path, required=True)
    parser.add_argument("--snapshot-date", default="2022-12-31")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/dependency_analysis/results"),
    )
    parser.add_argument(
        "--projects", nargs="*", default=PROJECTS,
        help="解析対象プロジェクト（デフォルト全10）",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges_rows: list[dict] = []
    summary: list[dict] = []

    for project in args.projects:
        repo_path = args.repos_dir / project
        if not repo_path.exists():
            print(f"[skip] {project}: repo not cloned at {repo_path}")
            continue

        commit = checkout_at(repo_path, args.snapshot_date)
        if not commit:
            print(f"[skip] {project}: no commit before {args.snapshot_date}")
            continue
        print(f"[run]  {project}: checked out {commit[:8]}", flush=True)

        try:
            r = analyze_repo(repo_path, project)
        finally:
            restore_head(repo_path)

        for (src, dst), w in r["edges"].items():
            edges_rows.append({
                "project": f"openstack/{project}",
                "src": src,
                "dst": dst,
                "weight": w,
            })
        summary.append({
            "project": project,
            "n_py_files": r["n_py_files"],
            "n_parse_err": r["n_parse_err"],
            "n_edges": len(r["edges"]),
            "commit": commit,
            "snapshot_date": args.snapshot_date,
        })
        print(
            f"       {project}: {r['n_py_files']} py files, "
            f"{len(r['edges'])} dir-edges (parse_err={r['n_parse_err']})"
        )

    edges_df = pd.DataFrame(edges_rows)
    edges_path = args.output_dir / "static_dir_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    print(f"\n[OK] edges: {edges_path} ({len(edges_df)} rows)")

    # Hub: in-degree (誰から import されるか) と out-degree (どれだけ外に依存するか)
    if not edges_df.empty:
        in_w = (
            edges_df.groupby(["project", "dst"])["weight"].sum()
            .reset_index().rename(columns={"dst": "directory", "weight": "in_weight"})
        )
        in_n = (
            edges_df.groupby(["project", "dst"])["src"].nunique()
            .reset_index().rename(columns={"dst": "directory", "src": "in_degree"})
        )
        out_w = (
            edges_df.groupby(["project", "src"])["weight"].sum()
            .reset_index().rename(columns={"src": "directory", "weight": "out_weight"})
        )
        out_n = (
            edges_df.groupby(["project", "src"])["dst"].nunique()
            .reset_index().rename(columns={"src": "directory", "dst": "out_degree"})
        )
        hub = (
            in_w.merge(in_n, on=["project", "directory"], how="outer")
            .merge(out_w, on=["project", "directory"], how="outer")
            .merge(out_n, on=["project", "directory"], how="outer")
            .fillna(0)
        )
        hub["static_hub_score"] = hub["in_degree"] + hub["out_degree"]
        hub_path = args.output_dir / "static_hub_scores.csv"
        hub.to_csv(hub_path, index=False)
        print(f"[OK] hub_scores: {hub_path} ({len(hub)} dirs)")

    summary_path = args.output_dir / "static_dir_deps_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] summary: {summary_path}")


if __name__ == "__main__":
    main()
