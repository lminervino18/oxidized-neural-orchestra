#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORCHESTRA_PY_DIR="$ROOT/orchestra-py"

# Parse --release flag
RELEASE=""
for arg in "$@"; do
  [ "$arg" = "--release" ] && RELEASE="--release"
done

# Read node and server counts from env or defaults (servers must be < nodes)
NODES="${NODES:-6}"
SERVERS="${SERVERS:-3}"

# Bring up the nodes via Docker; the orchestrator assigns server/worker roles at runtime
python3 "$ROOT/docker/compose_up.py" \
  --nodes "$NODES" \
  $RELEASE

sleep 5

# Open Docker logs in a new terminal
setsid gnome-terminal --title="docker-logs" -- bash -c "
  docker compose -f $ROOT/compose.yaml logs -f
  exec bash
" &

sleep 0.5

# Run orchestra-py local.py with Docker hostnames and dataset path
cd "$ROOT"
NODES="$NODES" \
SERVERS="$SERVERS" \
RUST_LOG=info \
python3 "$ORCHESTRA_PY_DIR/local.py"