# docker

Simulates a full **Oxidized Neural Orchestra (ONO)** cluster locally as `N` identical `node` containers on one machine.

## Scripts

- `compose_up.py` — entry point: generates the compose file, patches `/etc/hosts`, and brings up `N` nodes with `docker compose up --build -d`.
- `gen_compose.py` — renders `compose.yaml` from `NODES`/`RELEASE`; picks `release`/`debug` build mode and sets `RUST_LOG` accordingly (`info` for release, `debug` otherwise).
- `fill_hosts.py` — maps `node-i` → `127.0.0.1` in `/etc/hosts` inside a tagged block (idempotent: replaces the block on re-run).

## Usage

```bash
python3 docker/compose_up.py --nodes N [--release]
```

Requires `sudo` (it edits `/etc/hosts`) and Docker. Node `i` listens on `40000 + i`. See the [root README](../README.md#simulating-locally-with-docker).
