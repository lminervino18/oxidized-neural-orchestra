#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


# The directory containing this exact script.
DOCKER_DIR = Path(__file__).resolve().parent

# The project's directory.
ROOT_DIR = DOCKER_DIR.parent


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
    subprocess.run([DOCKER_DIR / "gen_compose.py"], env=env, check=True, cwd=ROOT_DIR)
    subprocess.run(["sudo", "-E", DOCKER_DIR / "fill_hosts.py"], env=env, check=True)

    cmd = ["docker", "compose", "-f", "compose.yaml", "up", "--build", "-d", "--remove-orphans"]
    subprocess.run(cmd, cwd=ROOT_DIR)


if __name__ == "__main__":
    main()
