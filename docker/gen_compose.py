#!/usr/bin/env python3

import os
import json

DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_OUTPUT_PATH = "../compose.yml"

# The various values for a yaml field.
type YmlField = bool | int | float | str | list[YmlField] | dict[str, YmlField]


def generate_servers(release: bool, servers: int) -> dict[str, YmlField]:
    """
    Generates the servers' part of the compose file.

    # Arguments
    * `release` - If the executable should be compiled as release mode.
    * `servers` - The amount of servers to create.

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


def generate_workers(release: bool, workers: int) -> dict[str, YmlField]:
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


def generate_compose(release: bool, servers: int, workers: int) -> dict[str, YmlField]:
    """
    Generates the entire docker compose file in a dictionary.

    # Arguments
    * `release` - If the executable should be compiled as release mode.
    * `servers` - The amount of servers to create.
    * `workers` - The amount of workers to create.

    # Returns
    A dictionary containing the whole project's docker compose file.
    """
    services = generate_servers(release, servers) | generate_workers(release, workers)

    return {
        "name": "distributed-training",
        "services": services,
        "networks": {
            "training-network": {
                "driver": "bridge",
            },
        },
    }


def main():
    CONFIG_PATH = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    docker_compose = generate_compose(**config)
    OUTPUT_PATH = os.environ.get("OUTPUT_PATH", DEFAULT_OUTPUT_PATH)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(docker_compose, f, indent=2)


if __name__ == "__main__":
    main()
