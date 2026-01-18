#!/usr/bin/env python3

from typing import Any
import os
import json

DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_OUTPUT_PATH = "../compose.yml"


def generate_servers(release: bool, servers: int) -> dict[str, Any]:
    """
    Generates the servers' part of the compose file.

    # Arguments
    * `release` - If the executable should be compiled as release mode.
    * `servers` - The amount of servers to create.

    # Returns
    A dictionary containing the servers' part of the compose file.
    """
    mode = "release" if release else "debug"

    return {
        f"server-{i}": {
            "container_name": f"server-{i}",
            "build": {
                "dockerfile": "parameter_server/Dockerfile",
                "args": {
                    "MODE": mode,
                },
            },
        }
        for i in range(1, servers + 1)
    }


def generate_compose(release: bool, servers: int) -> dict:
    """
    Generates the entire docker compose file in a dictionary.

    # Arguments
    * `release` - If the executable should be compiled as release mode.
    * `servers` - The amount of servers to create.

    # Returns
    A dictionary containing the whole project's docker compose file.
    """
    services = generate_servers(release, servers)

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
