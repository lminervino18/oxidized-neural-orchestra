#!/usr/bin/env python3

import os

TAG = "ONO - LOCAL DISTRIBUTED TRAINING TRANSLATIONS"
START_TAG = f"# {TAG} - START"
END_TAG = f"# {TAG} - END"


def build_host_block(workers: int, servers: int) -> str:
    """
    Builds the translation string to incorporate to the /etc/hosts file.

    # Args
    * `workers` - The amount of workers to use to train.
    * `servers` - The amount of servers to use to train.

    # Returns
    The content to write/replace in the /etc/hosts file.
    """
    worker_translations = "\n".join(f"127.0.0.1 worker-{i}" for i in range(workers))
    server_translations = "\n".join(f"127.0.0.1 server-{i}" for i in range(servers))
    return "\n".join((START_TAG, worker_translations, server_translations, END_TAG)) + "\n"


def insert_block(hosts: str, block: str) -> str:
    """
    Appends or replaces the translations block in the given hosts string.

    # Args
    * `hosts` - The content on the hosts translations file.
    * `block` - The block of server and worker addresses.

    # Returns
    The new hosts file content containing all the necessary translations.
    """
    start = hosts.find(START_TAG)
    end = hosts.find(END_TAG)

    if start != -1 and end != -1:
        hosts = hosts[:start] + hosts[end + len(END_TAG) :]

    return hosts.rstrip() + "\n\n" + block


def main():
    workers = int(os.environ["WORKERS"])
    servers = int(os.environ["SERVERS"])

    block = build_host_block(workers, servers)
    hosts_file_path = "/etc/hosts"
    tmp_hosts_file_path = f"{hosts_file_path}.tmp"

    with open(hosts_file_path, "r") as f:
        hosts = f.read()

    new_hosts = insert_block(hosts, block)

    with open(tmp_hosts_file_path, "w") as tmp:
        tmp.write(new_hosts)

    os.replace(tmp_hosts_file_path, hosts_file_path)


if __name__ == "__main__":
    main()
