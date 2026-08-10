#!/usr/bin/env python3

import json
import os

# base port for nodes.
BASE_PORT = 40_000
# the number of available cpus for each node.
CPUS = 1

# The various values for a yaml field.
type YmlField = bool | int | float | str | list[YmlField] | dict[str, YmlField]


def generate_pumba(nodes: int) -> dict[str, YmlField]:
    """
    Generates the pumba container for running on a simulated network.

    # Args
    * `nodes` - The amount of nodes in the network.

    # Returns
    The pumba container dictionary.
    """
    cmd = [
        "netem --duration 300m",
        "rate --rate 10mbit",
        *(f"node-{i}" for i in range(nodes)),
    ]

    return {
        "pumba": {
            "container_name": "pumba",
            "image": "ghcr.io/alexei-led/pumba:1.1.7",
            "volumes": [
                "/var/run/docker.sock:/var/run/docker.sock",
            ],
            "depends_on": [f"node-{i}" for i in range(nodes)],
            "command": " ".join(cmd),
        }
    }


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
            "deploy": {
                "resources": {
                    "limits": {
                        "cpus": f"{CPUS}",
                    },
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


def generate_services(nodes: int, release: bool, pumba: bool) -> dict[str, YmlField]:
    """
    Generates the services section of the compose file.

    # Args
    * `nodes` - The amount of nodes to create.
    * `release` - If the executable should be compiled as release mode.
    * `pumba` - Whether to simulate limited network bandwidth with Pumba.

    # Returns
    A dictionary containing the services part of the compose file.
    """
    services = generate_nodes(nodes, release)

    if pumba:
        services |= generate_pumba(nodes)

    return services


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


def generate_compose(nodes: int, release: bool, pumba: bool) -> dict[str, YmlField]:
    """
    Generates the entire docker compose file in a dictionary.

    # Args
    * `nodes` - The amount of nodes to create.
    * `release` - If the executable should be compiled as release mode.
    * `pumba` - Whether to simulate limited network bandwidth with Pumba.

    # Returns
    A dictionary containing the whole project's docker compose file.
    """
    return {
        "name": "distributed-training",
        "services": generate_services(nodes, release, pumba),
        "networks": generate_network(),
    }


def main():
    nodes = int(os.environ["NODES"])
    release = os.environ["RELEASE"].lower() == "true"
    pumba = os.environ["PUMBA"].lower() == "true"

    docker_compose = generate_compose(nodes, release, pumba)

    with open("compose.yaml", "w") as f:
        json.dump(docker_compose, f, indent=2)


if __name__ == "__main__":
    main()
