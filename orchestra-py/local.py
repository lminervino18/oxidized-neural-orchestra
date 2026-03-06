import subprocess
import time
import sys
import os

WORKER_ADDRS = ["127.0.0.1:50000", "127.0.0.1:50001", "127.0.0.1:50002"]
SERVER_ADDRS = ["127.0.0.1:40000", "127.0.0.1:40001"]


def kill_ports(addrs: list[str]) -> None:
    for addr in addrs:
        port = addr.split(":")[1]
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
        )
        for pid in result.stdout.strip().splitlines():
            subprocess.run(["kill", "-9", pid], capture_output=True)


def prebuild(root: str) -> None:
    print("pre-building binaries...")
    subprocess.run(
        ["cargo", "build", "-p", "parameter_server", "-p", "worker"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  build done")


def start_servers(root: str) -> list[subprocess.Popen]:
    procs = []
    for i, addr in enumerate(SERVER_ADDRS):
        _, port_str = addr.split(":")
        env = {**os.environ, "PORT": port_str, "RUST_LOG": "debug"}
        log_file = open(f"/tmp/server-{i}.log", "w")
        p = subprocess.Popen(
            ["cargo", "run", "-p", "parameter_server"],
            cwd=root,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
        print(f"  server-{i} starting (port {port_str}, pid {p.pid})...")
        procs.append(p)
    time.sleep(4)
    print("  servers ready")
    return procs


def start_workers(root: str) -> list[subprocess.Popen]:
    procs = []
    for i, addr in enumerate(WORKER_ADDRS):
        _, port_str = addr.split(":")
        env = {**os.environ, "PORT": port_str, "RUST_LOG": "debug"}
        log_file = open(f"/tmp/worker-{i}.log", "w")
        p = subprocess.Popen(
            ["cargo", "run", "-p", "worker"],
            cwd=root,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
        print(f"  worker-{i} starting (port {port_str}, pid {p.pid})...")
        procs.append(p)
    time.sleep(2)
    print("  workers ready")
    return procs


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("killing any leftover processes on training ports...")
    kill_ports(SERVER_ADDRS + WORKER_ADDRS)
    time.sleep(1)

    prebuild(root)

    print("starting servers...")
    servers = start_servers(root)

    print("starting workers...")
    workers = start_workers(root)

    env = {
        **os.environ,
        "WORKER_ADDRS": ",".join(WORKER_ADDRS),
        "SERVER_ADDRS": ",".join(SERVER_ADDRS),
    }

    try:
        subprocess.run(
            ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")],
            cwd=root,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError:
        sys.exit(1)
    finally:
        print("\nstopping workers and servers...")
        for p in workers + servers:
            p.terminate()
        for p in workers + servers:
            p.wait()


if __name__ == "__main__":
    main()