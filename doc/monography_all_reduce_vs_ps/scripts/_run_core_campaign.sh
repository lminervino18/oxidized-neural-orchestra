#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
echo "[$(date +%H:%M:%S)] === CORE CAMPAIGN START ==="
# REDESIGN campaign: equal-workers fairness, PS with 2 sharded servers,
# FashionMNIST primary + MNIST reference, small-batch recipe.

# Clean gate: purge ALL stale execution data so no old run leaks into the new
# campaign. Artifacts accumulate by run-signature and are never overwritten, so
# they MUST be wiped here. Conceptual figures and paper source are left intact.
echo "[$(date +%H:%M:%S)] purging stale results (raw/processed/artifacts/data)"
rm -f results/raw/*.jsonl results/processed/*.csv results/artifacts/*.safetensors
rm -rf results/data/*

for cfg in exp_a_sanity exp_b_fair_convergence exp_c_throughput exp_d_communication ; do
  echo "[$(date +%H:%M:%S)] >>> running $cfg"
  rm -f "results/raw/$cfg.jsonl"   # fresh run — never append to stale results
  $PY scripts/run_experiment.py --config "configs/$cfg.json" --name "$cfg"
  echo "[$(date +%H:%M:%S)] <<< done $cfg (exit $?)"
done
echo "[$(date +%H:%M:%S)] >>> aggregating"
$PY scripts/aggregate.py
echo "[$(date +%H:%M:%S)] === CORE CAMPAIGN COMPLETE ==="
