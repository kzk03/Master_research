"""Gerrit から openstack/* の ACTIVE プロジェクト一覧を取得して JSON 保存する。

Usage:
    uv run python scripts/pipeline/fetch_gerrit_openstack.py
"""

import json
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUT = Path(__file__).resolve().parents[2] / "data" / "gerrit_openstack_projects.json"


def fetch_gerrit_openstack():
    url = "https://review.opendev.org/projects/"

    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info("Fetched Gerrit OpenStack projects successfully.")
    except requests.RequestException as e:
        logger.error(f"Error fetching Gerrit OpenStack projects: {e}")
        return {"error": str(e)}

    raw_json = response.text.replace(")]}'\n", "", 1)
    all_projects = json.loads(raw_json)

    logger.info('アクティブかつプレフィックスがopenstackのプロジェクトを抽出中...')
    active_repositories = {}
    for repo_name, repo_info in all_projects.items():
        if repo_name.startswith('openstack/') and repo_info.get('state') == 'ACTIVE':
            active_repositories[repo_name] = repo_info

    return active_repositories


if __name__ == "__main__":
    projects = fetch_gerrit_openstack()

    if projects and "error" not in projects:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, 'w') as f:
            json.dump(projects, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(projects)} active Gerrit OpenStack projects to {OUT}")
        out_txt = OUT.with_suffix('.txt')
        
        # テキストファイルとして開いて書き込む
        with open(out_txt, 'w') as f:
            for repo_name in projects.keys():
                f.write(f"{repo_name}\n") # \n（改行）を入れて1行ずつ書き込む
                
        logger.info(f"Saved {len(projects)} repository names to {out_txt}")
