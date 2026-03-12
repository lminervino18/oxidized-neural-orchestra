#!/bin/bash
# Starts workers and servers in the background (headless, no TUI).
# Called by the example notebook before training.
#
# Usage:
#   ./start_entities.sh <nworkers> <nservers>
#   ./start_entities.sh 3 2

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
NWORKERS="${1:-3}"
NSERVERS="${2:-2}"
LOG_DIR="$ROOT/temp"
mkdir -p "$LOG_DIR"

BASE_WORKER_PORT=50000
BASE_SERVER_PORT=40000

echo "Starting $NSERVERS parameter server(s)..."
for i in $(seq 0 $((NSERVERS - 1))); do
  PORT=$((BASE_SERVER_PORT + i))
  LOG="$LOG_DIR/server-$i.log"
  PORT=$PORT RUST_LOG=info cargo run --release -p parameter_server > "$LOG" 2>&1 &
  echo "  server-$i → port $PORT (log: $LOG)"
done

sleep 1

echo "Starting $NWORKERS worker(s)..."
for i in $(seq 0 $((NWORKERS - 1))); do
  PORT=$((BASE_WORKER_PORT + i))
  LOG="$LOG_DIR/worker-$i.log"
  PORT=$PORT RUST_LOG=info cargo run --release -p worker > "$LOG" 2>&1 &
  echo "  worker-$i → port $PORT (log: $LOG)"
done

echo "All entities started."