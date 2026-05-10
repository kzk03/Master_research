import json
from pathlib import Path

input_path = Path(__file__).resolve().parents[2] / "data" / "gerrit_openstack_projects.json"

def main():
    with open(input_path, 'r') as f:
        projects = json.load(f)
    print(f"Number of active OpenStack projects: {len(projects)}")

if __name__ == "__main__":
    main()