#!/usr/bin/env python3

import os
import json
from pathlib import Path

# The directory containing this exact script.
BASE_DIR = Path(__file__).resolve().parent

# The various values for a yaml field.
type YmlField = bool | int | float | str | list[YmlField] | dict[str, YmlField]


def generate_servers(servers: int, release: bool) -> dict[str, YmlField]:
    """
    Generates the servers' part of the compose file.

    # Args
    * `servers` - The amount of servers to create.
    * `release` - If the executable should be compiled as release mode.

    # Returns
    A dictionary containing the servers' part of the compose file.
    """
    mode = "release" if release else "debug"
    log_level = "info" if release else "debug"
    base_port = 40_000

    return {
        f"server-{i}": {
            "container_name": f"server-{i}",
            "build": {
                "dockerfile": "parameter_server/Dockerfile",
                "args": {
                    "MODE": mode,
                },
            },
            "ports": [
                f"{base_port + i}:{base_port + i}",
            ],
            "networks": [
                "training-network",
            ],
            "environment": {
                "HOST": "0.0.0.0",
                "PORT": base_port + i,
                "RUST_LOG": log_level,
            },
        }
        for i in range(servers)
    }


def generate_workers(workers: int, servers: int, release: bool) -> dict[str, YmlField]:
    """
    Generates the workers' part of the compose file.

    # Args
    * `workers` - The amount of workers to create.
    * `servers` - The amount of servers to create.
    * `release` - If the executable should be compiled as release mode.

    # Returns
    A dictionary containing the workers' part of the compose file.
    """
    mode = "release" if release else "debug"
    log_level = "info" if release else "debug"
    base_port = 50_000

    return {
        f"worker-{i}": {
            "container_name": f"worker-{i}",
            "build": {
                "dockerfile": "worker/Dockerfile",
                "args": {
                    "MODE": mode,
                },
            },
            "depends_on": [f"server-{j}" for j in range(servers)],
            "ports": [
                f"{base_port + i}:{base_port + i}",
            ],
            "networks": [
                "training-network",
            ],
            "environment": {
                "HOST": "0.0.0.0",
                "PORT": base_port + i,
                "RUST_LOG": log_level,
            },
        }
        for i in range(workers)
    }


def generate_network() -> dict[str, YmlField]:
    """
    Generates the network the system is going to be running on.

    # Returns
    A dictionary containing the network part of the compose file.
    """
    return {
        "training-network": {
            "driver": "bridge",
        },
    }


def generate_compose(workers: int, servers: int, release: bool) -> dict[str, YmlField]:
    """
    Generates the entire docker compose file in a dictionary.

    # Args
    * `workers` - The amount of workers to create.
    * `servers` - The amount of servers to create.
    * `release` - If the executable should be compiled as release mode.

    # Returns
    A dictionary containing the whole project's docker compose file.
    """
    return {
        "name": "distributed-training",
        "services": generate_servers(servers, release) | generate_workers(workers, servers, release),
        "networks": generate_network(),
    }


def main():
    workers = int(os.environ["WORKERS"])
    servers = int(os.environ["SERVERS"])
    release = os.environ["RELEASE"].lower() == "true"

    docker_compose = generate_compose(workers, servers, release)
    output_path = BASE_DIR.parent / "compose.yaml"

    with open(output_path, "w") as f:
        json.dump(docker_compose, f, indent=2)


if __name__ == "__main__":
    main()
