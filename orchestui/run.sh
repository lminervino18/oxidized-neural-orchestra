#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORCHESTUI_DIR="$ROOT/orchestui"

RELEASE="--release"

read NWORKERS NSERVERS < <(python3 -c "
import json

d = json.load(open('$ORCHESTUI_DIR/training.json'))
ps = d.get('algorithm', {}).get('parameter_server', {})

nworkers = len(d.get('worker_addrs', []))
nservers = len(ps.get('server_addrs', []))

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