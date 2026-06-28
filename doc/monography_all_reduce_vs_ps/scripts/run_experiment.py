#!/usr/bin/env python3
"""Run an AllReduce-vs-Parameter-Server experiment matrix.

Each "run" is a dict with keys:
    dataset, model, strategy (all_reduce|parameter_server|strategy_switch|baseline),
    workers, servers, batch_size (per-worker), max_epochs, offline_epochs, lr,
    loss_fn, seed, subset (optional, train only), eval (bool), label, notes.

The runner loads datasets, spins the Docker cluster (project ``ono-mono``) once per
distinct node-count, runs each config and APPENDS one raw JSON line per run to
``results/raw/<name>.jsonl``. A human log is written to ``logs/``. A failing run is
captured (status="error", traceback in ``error``) and the campaign CONTINUES.

Usage:
    .venv/bin/python scripts/run_experiment.py --smoke
    .venv/bin/python scripts/run_experiment.py --config configs/my_matrix.json
    .venv/bin/python scripts/run_experiment.py --config m.json --name my_campaign
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import ono_harness as H

RAW_DIR = H.RESULTS_DIR / "raw"
LOGS_DIR = H.MONO_DIR / "logs"

# Defaults applied to every run dict before execution.
RUN_DEFAULTS = {
    "dataset": "mnist",
    "model": "nielsen",
    "strategy": "all_reduce",
    "workers": 1,
    "servers": 0,
    "batch_size": 64,
    "max_epochs": 10,
    "offline_epochs": 0,
    "lr": 0.1,
    "loss_fn": "cross_entropy",
    "seed": 42,
    "subset": None,
    "eval": False,
    "early_stopping_tolerance": None,
    "label": None,
    "notes": None,
}


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def normalize(run: dict) -> dict:
    out = dict(RUN_DEFAULTS)
    out.update(run)
    # Role sanity: baseline / all_reduce have 0 servers.
    if out["strategy"] in ("baseline", "all_reduce"):
        out["servers"] = 0
    if out["strategy"] == "baseline":
        out["workers"] = 1
    return out


# ── Cluster manager: bring up the cluster once per distinct node-count ─────────

class Cluster:
    def __init__(self, log: Logger):
        self.log = log
        self.current = None  # currently-running node count (or None)

    def ensure(self, nodes: int):
        if nodes == 0:
            return  # baseline: no cluster needed
        if self.current == nodes:
            return
        if self.current is not None:
            self.log(f"tearing down cluster of {self.current} nodes")
            H.docker_down()
            self.current = None
        self.log(f"bringing up cluster of {nodes} nodes (build may take ~10-15 min "
                 f"on first run)")
        H.docker_up(nodes)
        if not H.wait_nodes_ready(nodes):
            raise RuntimeError(f"cluster of {nodes} nodes did not become ready")
        self.current = nodes
        self.log(f"cluster of {nodes} nodes ready: {H.node_addrs(nodes)}")

    def teardown(self):
        if self.current is not None:
            self.log(f"final teardown of {self.current} nodes")
            H.docker_down()
            self.current = None


def order_runs(runs: list[dict]) -> list[dict]:
    """Group by node count so we build each cluster size once. Baselines (0
    nodes) run last so they never hold up cluster reuse."""
    return sorted(runs, key=lambda r: H.nodes_for(r) if r["strategy"] != "baseline" else 10_000)


def run_campaign(runs: list[dict], name: str) -> tuple[Path, list[dict]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{name}.jsonl"
    log = Logger(LOGS_DIR / f"{name}.log")

    runs = [normalize(r) for r in runs]
    log(f"campaign '{name}': {len(runs)} runs -> {raw_path}")

    # Clean slate before we start.
    H.docker_down()

    cluster = Cluster(log)
    results: list[dict] = []
    try:
        for i, run in enumerate(order_runs(runs), 1):
            label = run.get("label") or f"{run['strategy']}/{run['model']}"
            log(f"--- run {i}/{len(runs)}: {label} "
                f"[{run['strategy']} w={run['workers']} s={run['servers']} "
                f"bs={run['batch_size']} ep={run['max_epochs']} ds={run['dataset']}]")
            try:
                if run["strategy"] != "baseline":
                    cluster.ensure(H.nodes_for(run))
                result = H.run_single(run)
            except Exception:
                # Cluster/setup failure for this run — capture and continue.
                result = H._base_result(run)
                result["status"] = "error"
                result["error"] = traceback.format_exc()
                result = H.derive(result)

            results.append(result)
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

            acc = result.get("test_accuracy")
            log(f"    status={result['status']} epochs_ran={result['epochs_ran']} "
                f"final_loss={result['final_loss']} acc={acc} "
                f"secs={result['train_seconds']}")
            if result["status"] == "error":
                log(f"    ERROR: {(result['error'] or '').strip().splitlines()[-1:]}")
    finally:
        cluster.teardown()
        H.docker_down()
        log("campaign complete")
        log.close()

    return raw_path, results


# ── Smoke ─────────────────────────────────────────────────────────────────────

def smoke_runs() -> list[dict]:
    common = dict(dataset="mnist", model="nielsen", subset=500, max_epochs=2,
                  lr=0.1, loss_fn="cross_entropy", seed=42, eval=True)
    return [
        {**common, "strategy": "all_reduce", "workers": 3, "servers": 0,
         "batch_size": 16, "label": "smoke_ar3"},
        {**common, "strategy": "parameter_server", "workers": 2, "servers": 1,
         "batch_size": 16, "label": "smoke_ps2x1"},
        {**common, "strategy": "baseline", "workers": 1, "servers": 0,
         "batch_size": 16, "label": "smoke_baseline"},
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Run an AR-vs-PS experiment matrix.")
    ap.add_argument("--config", help="path to a JSON file: list of run dicts, or "
                    "{\"runs\": [...], \"name\": ...}")
    ap.add_argument("--smoke", action="store_true", help="run the built-in smoke matrix")
    ap.add_argument("--name", help="campaign name (default: timestamp)")
    args = ap.parse_args()

    if not args.config and not args.smoke:
        ap.error("provide --config <json> or --smoke")

    if args.smoke:
        runs = smoke_runs()
        name = args.name or "smoke"
    else:
        data = json.loads(Path(args.config).read_text())
        if isinstance(data, dict):
            runs = data["runs"]
            name = args.name or data.get("name") or Path(args.config).stem
        else:
            runs = data
            name = args.name or Path(args.config).stem

    raw_path, results = run_campaign(runs, name)

    ok = sum(1 for r in results if r["status"] == "ok")
    err = len(results) - ok
    print(f"\n== summary == {ok} ok / {err} error -> {raw_path}")
    for r in results:
        print(f"  {r.get('label')}: status={r['status']} "
              f"acc={r.get('test_accuracy')} secs={r.get('train_seconds')} "
              f"epochs={r.get('epochs_ran')}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
