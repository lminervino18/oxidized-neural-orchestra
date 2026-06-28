#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
echo "[$(date +%H:%M:%S)] === CORE CAMPAIGN START ==="
for cfg in exp_a_sanity exp_b_fixed_global exp_b_fixed_perworker ; do
  echo "[$(date +%H:%M:%S)] >>> running $cfg"
  $PY scripts/run_experiment.py --config "configs/$cfg.json" --name "$cfg"
  echo "[$(date +%H:%M:%S)] <<< done $cfg (exit $?)"
done
echo "[$(date +%H:%M:%S)] === CORE CAMPAIGN COMPLETE ==="
