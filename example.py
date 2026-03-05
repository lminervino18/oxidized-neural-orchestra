import subprocess
import time
import sys
import os

from orchestra import ModelBuilder, Orchestrator, TrainingBuilder

WORKER_ADDRS = ["127.0.0.1:50000", "127.0.0.1:50001", "127.0.0.1:50002"]
SERVER_ADDRS = ["127.0.0.1:40000", "127.0.0.1:40001"]

DATA = [
    1.0, 2.0,
    2.0, 4.0,
    3.0, 6.0,
    4.0, 8.0,
    5.0, 10.0,
    6.0, 12.0,
    7.0, 14.0,
    8.0, 16.0,
]


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
    time.sleep(2)
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


def print_params(params: list[float], output_sizes: list[int], input_size: int) -> None:
    print(f"\ntrained parameters ({len(params)} total):")
    offset = 0
    prev = input_size
    for layer_i, out in enumerate(output_sizes):
        w_count = prev * out
        b_count = out
        weights = params[offset : offset + w_count]
        biases = params[offset + w_count : offset + w_count + b_count]
        print(f"\n  layer {layer_i}  ({prev}x{out})")
        print(f"    weights: {[round(w, 4) for w in weights]}")
        print(f"    biases:  {[round(b, 4) for b in biases]}")
        offset += w_count + b_count
        prev = out


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))

    print("killing any leftover processes on training ports...")
    kill_ports(SERVER_ADDRS + WORKER_ADDRS)
    time.sleep(1)

    prebuild(root)

    print("starting servers...")
    servers = start_servers(root)

    print("starting workers...")
    workers = start_workers(root)

    print("\nbuilding model...")
    mb = ModelBuilder()
    mb.dense(8, 1.0)
    mb.dense(4, 1.0)
    mb.dense(1)
    model = mb.build()

    print("building training config...")
    tb = TrainingBuilder(WORKER_ADDRS, SERVER_ADDRS)
    tb.inline_dataset(DATA, x_size=1, y_size=1)
    tb.barrier_sync()
    tb.max_epochs(100)
    tb.batch_size(4)
    tb.seed(42)
    training = tb.build()

    print("\nstarting training session...")
    orch = Orchestrator()
    session = orch.train(model, training)

    print("waiting for training to complete...")
    try:
        trained = session.wait()
        params = trained.weights()
        print_params(params, [8, 4, 1], input_size=1)
    except RuntimeError as e:
        print(f"training failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print("\nstopping workers and servers...")
        for p in workers + servers:
            p.terminate()
        for p in workers + servers:
            p.wait()

    print("\ndone.")


if __name__ == "__main__":
    main()