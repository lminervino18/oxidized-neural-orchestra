#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORCHESTUI_DIR="$ROOT/orchestui"

# Parse --release flag
RELEASE=""
for arg in "$@"; do
  [ "$arg" = "--release" ] && RELEASE="--release"
done

NWORKERS=$(python3 -c "
import json
d = json.load(open('$ORCHESTUI_DIR/training.json'))
print(len(d['worker_addrs']))
")

NSERVERS=$(python3 -c "
import json
d = json.load(open('$ORCHESTUI_DIR/training.json'))
print(len(d['algorithm']['parameter_server']['server_addrs']))
")

python3 "$ROOT/docker/compose_up.py" \
  --workers "$NWORKERS" \
  --servers "$NSERVERS" \
  ${RELEASE:+--release}

sleep 2

setsid gnome-terminal --title="docker-logs" -- bash -c "
  docker compose -f $ROOT/compose.yaml logs -f
  exec bash
" &

sleep 0.5

cd "$ROOT"
RUST_LOG=info cargo run ${RELEASE:+--release} -p orchestui 2>/tmp/ono-orchestrator.log
