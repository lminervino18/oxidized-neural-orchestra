#!/usr/bin/env python3

import os
import json

DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_OUTPUT_PATH = "../compose.yml"


def generate_header():
    """
    Generates the header part of the docker-compose file.
    """
    return "name: parameter_server_training\nservices:"


def generate_servers(servers: int) -> str:
    """
    Generates the server part of the docker-compose file.

    # Arguments
    * `servers` - The amount of servers to create.

    # Returns
    The string containing the configuration.
    """
    compose_servers = ["\n  # ---------------- Parameter Server Shards ----------------\n"]

    for i in range(1, servers + 1):
        compose_server = f"""
  parameter-server-{i}:
    container_name: parameter-server-{i}
    build:
      dockerfile: parameter_server/Dockerfile
    networks:
      - training-network
    env_file:
      - parameter_server/.env
"""
        compose_servers.append(compose_server)

    return "".join(compose_servers)


def generate_networks():
    """
    Generates the networks part of the docker-compose file.
    """
    return """
networks:
  training-network:
    driver: bridge
"""


def main():
    CONFIG_PATH = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    parts = (
        generate_header(),
        generate_servers(**config),
        generate_networks(),
    )

    output = "".join(parts)
    OUTPUT_PATH = os.environ.get("OUTPUT_PATH", DEFAULT_OUTPUT_PATH)

    with open(OUTPUT_PATH, "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
