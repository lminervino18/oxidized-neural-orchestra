#!/bin/bash
# Kills all workers and servers started by start_entities.sh.
# Called by the example notebook after training completes.

BASE_WORKER_PORT=50000
BASE_SERVER_PORT=40000
NWORKERS="${1:-3}"
NSERVERS="${2:-2}"

echo "Stopping entities..."

for i in $(seq 0 $((NSERVERS - 1))); do
  PORT=$((BASE_SERVER_PORT + i))
  pid=$(lsof -t -i:$PORT 2>/dev/null)
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null && echo "  stopped server on port $PORT (pid $pid)"
  fi
done

for i in $(seq 0 $((NWORKERS - 1))); do
  PORT=$((BASE_WORKER_PORT + i))
  pid=$(lsof -t -i:$PORT 2>/dev/null)
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null && echo "  stopped worker on port $PORT (pid $pid)"
  fi
done

echo "Done."