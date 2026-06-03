#!/usr/bin/env python3

import os
import json

# base port for nodes.
BASE_PORT = 40_000

# The various values for a yaml field.
type YmlField = bool | int | float | str | list[YmlField] | dict[str, YmlField]


def generate_nodes(nodes: int, release: bool) -> dict[str, YmlField]:
    """
    Generates the node services part of the compose file.

    # Args
    * `nodes` - The amount of nodes to create.
    * `release` - If the executable should be compiled as release mode.

    # Returns
    A dictionary containing the node services part of the compose file.
    """
    mode = "release" if release else "debug"
    log_level = "info" if release else "debug"

    return {
        f"node-{i}": {
            "container_name": f"node-{i}",
            "build": {
                "dockerfile": "node/Dockerfile",
                "args": {
                    "MODE": mode,
                },
            },
            "ports": [
                f"{BASE_PORT + i}:{BASE_PORT + i}",
            ],
            "networks": [
                "training-network",
            ],
            "environment": {
                "HOST": "0.0.0.0",
                "PORT": BASE_PORT + i,
                "RUST_LOG": log_level,
            },
        }
        for i in range(nodes)
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


def generate_compose(nodes: int, release: bool) -> dict[str, YmlField]:
    """
    Generates the entire docker compose file in a dictionary.

    # Args
    * `nodes` - The amount of nodes to create.
    * `release` - If the executable should be compiled as release mode.

    # Returns
    A dictionary containing the whole project's docker compose file.
    """
    return {
        "name": "distributed-training",
        "services": generate_nodes(nodes, release),
        "networks": generate_network(),
    }


def main():
    nodes = int(os.environ["NODES"])
    release = os.environ["RELEASE"].lower() == "true"

    docker_compose = generate_compose(nodes, release)

    with open("compose.yaml", "w") as f:
        json.dump(docker_compose, f, indent=2)


if __name__ == "__main__":
    main()
