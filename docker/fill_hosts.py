#!/usr/bin/env python3

import os

TAG = "ONO - LOCAL DISTRIBUTED TRAINING TRANSLATIONS"
START_TAG = f"# {TAG} - START"
END_TAG = f"# {TAG} - END"


def build_host_block(nodes: int) -> str:
    """
    Builds the translation string to incorporate to the /etc/hosts file.

    # Args
    * `nodes` - The amount of nodes to use to train.

    # Returns
    The content to write in the /etc/hosts file.
    """
    if nodes == 0:
        return ""

    worker_translations = "\n".join(f"127.0.0.1 node-{i}" for i in range(nodes))
    return "\n".join((START_TAG, worker_translations, END_TAG)) + "\n"


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

    if len(block) == 0:
        return hosts.rstrip() + "\n"

    return hosts.rstrip() + "\n\n" + block


def main():
    nodes = int(os.environ["NODES"])

    block = build_host_block(nodes)
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
