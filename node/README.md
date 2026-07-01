# node

The single, role-agnostic binary that every machine in an **Oxidized Neural Orchestra**
cluster runs. A node just listens on a port and waits for the orchestrator to connect;
the spec it receives on connection decides whether it acts as a **worker** or a
**parameter server**. There is nothing to configure per role — the same binary and the
same command run everywhere.

## What it does

- Binds a TCP listener and waits for an orchestrator connection.
- Accepts the connection, receives its spec, and runs the assigned role
  (all-reduce worker, parameter-server worker, or parameter server).
- Answers latency/health probes used by the orchestrator to lay out the topology.

The heavy lifting lives in the library crates it wires together: [`comms`](../comms/)
(networking), [`worker`](../worker/) and [`parameter_server`](../parameter_server/README.md)
(the two runtimes), and [`machine_learning`](../machine_learning/) (the training engine).

## Build

Requires the Rust toolchain (`rustup`). From the repository root:

```bash
cargo build -p node --release
```

## Run

A node is configured entirely through environment variables:

| Variable   | Required | Default   | Meaning |
|------------|----------|-----------|---------|
| `PORT`     | **yes**  | —         | TCP port to listen on. The node exits with an error if it is unset. |
| `HOST`     | no       | `0.0.0.0` | Address to bind. |
| `RUST_LOG` | no       | (off)     | Log filter, e.g. `info`, `debug`, `node=debug`. Logs go to **stderr**. |

Start one node per machine — or several on one machine, each on a different port:

```bash
PORT=40000 RUST_LOG=info cargo run -p node --release
```

```bash
# a 3-node cluster on localhost, in three terminals
PORT=40000 cargo run -p node --release
PORT=40001 cargo run -p node --release
PORT=40002 cargo run -p node --release
```

Then list every node's `host:port` in the `addrs` field of your `training.json`
(or the `addrs=[...]` argument in `orchestra-py`) and drive the run from
[`orchestui`](../orchestui/README.md) or [`orchestra-py`](../orchestra-py/README.md).
Roles are assigned at runtime from `nservers`, not from the port a node listens on.

## Logs

The node uses the `log` facade with `env_logger`, so `RUST_LOG` controls verbosity.
Because it initializes logging for the whole process, the `worker` and
`parameter_server` library logs surface here too. Notable lines:

| Level   | Event |
|---------|-------|
| `info`  | `listening at <addr>`, `orchestrator connected`, `starting/finished worker session`, `starting/finished parameter server session` |
| `warn`  | unexpected connection type, failed node-to-node ping |
| `error` | session or stat-service failure |
| `debug` | per-event routing and per-round ping detail |

A normal run at `RUST_LOG=info` shows bind → connect → session start/finish; drop to
`debug` to trace the orchestrator handshake and health probes.

## Running a whole cluster locally

To simulate a cluster on a single machine, `docker/compose_up.py` brings up `N`
identical node containers (`node-i` on port `40000 + i`) — see the
[root README](../README.md#simulating-locally-with-docker).
