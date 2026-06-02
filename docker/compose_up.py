#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


# The directory containing this exact script.
BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes", type=int, required=True, help="The amount of nodes to use")
    parser.add_argument("--release", action="store_true", help="The compilation mode for the rust compiler")

    args = parser.parse_args()

    env = {
        **os.environ,
        "NODES": str(args.nodes),
        "RELEASE": str(args.release).lower(),
    }

    subprocess.run(["sudo", "-v"], check=True)
    subprocess.run([BASE_DIR / "gen_compose.py"], env=env, check=True)
    subprocess.run(["sudo", "-E", BASE_DIR / "fill_hosts.py"], env=env, check=True)

    project_root = BASE_DIR.parent
    compose_file_path = project_root / "compose.yaml"
    cmd = ["docker", "compose", "-f", compose_file_path, "up", "--build", "-d", "--remove-orphans"]
    subprocess.run(cmd, cwd=project_root)


if __name__ == "__main__":
    main()
