#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORCHESTUI_DIR="$ROOT/orchestui"

RELEASE="--release"

# Count unique containers by deduplicating repeated addresses.
# If the same host:port appears multiple times, only one container is launched —
# that node handles multiple roles concurrently via the NodeRouter.
read NWORKERS NSERVERS < <(python3 -c "
import json

d = json.load(open('$ORCHESTUI_DIR/training.json'))

worker_hosts = set(a.split(':')[0] for a in d.get('worker_addrs', []))
ps = d.get('algorithm', {}).get('parameter_server', {})
server_hosts = set(a.split(':')[0] for a in ps.get('server_addrs', []))

all_unique = worker_hosts | server_hosts

def max_idx(prefix):
    idxs = [int(h[len(prefix):]) for h in all_unique if h.startswith(prefix) and h[len(prefix):].isdigit()]
    return max(idxs) + 1 if idxs else 0

nworkers = max_idx('worker-')
nservers = max_idx('server-')

print(nworkers, nservers)
")

# Bring up workers and servers via Docker
python3 "$ROOT/docker/compose_up.py" \
  --workers "$NWORKERS" \
  --servers "$NSERVERS" \
  $RELEASE

sleep 2

# Open Docker logs in a new terminal
setsid gnome-terminal --title="docker-logs" -- bash -c "
  docker compose -f $ROOT/compose.yaml logs -f
  exec bash
" &

sleep 0.5

# Run TUI in current terminal
cd "$ROOT"
RUST_LOG=info cargo run ${RELEASE:+--release} -p orchestui 2>/tmp/ono-orchestrator.log