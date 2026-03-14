#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


# The directory containing this exact script.
BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers", type=int, required=True, help="The amount of workers to use")
    parser.add_argument("--servers", type=int, required=True, help="The amount of servers to use")
    parser.add_argument("--release", action="store_true", help="The compilation mode for the rust compiler")

    args = parser.parse_args()

    env = {
        **os.environ,
        "WORKERS": str(args.workers),
        "SERVERS": str(args.servers),
        "RELEASE": str(args.release).lower(),
    }

    subprocess.run(["sudo", "-v"], check=True)
    subprocess.run([BASE_DIR / "gen_compose.py"], env=env, check=True)
    subprocess.run(["sudo", "-E", BASE_DIR / "fill_hosts.py"], env=env, check=True)

    compose_file_path = BASE_DIR.parent / "compose.yaml"
    subprocess.run(["docker", "compose", "-f", compose_file_path, "up", "--build", "-d", "--remove-orphans"])


if __name__ == "__main__":
    main()
