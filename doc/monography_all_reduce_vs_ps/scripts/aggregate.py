#!/usr/bin/env python3
"""Consolidate results/raw/*.jsonl into the processed CSVs the paper consumes.

Raw schema (one JSON object per line, written by run_experiment.py):
  dataset, model, strategy, workers, servers, synchronizer, store,
  batch_size_per_worker, effective_global_batch, optimizer, learning_rate,
  seed, max_epochs, offline_epochs, loss_history[list], epochs_ran, final_loss,
  test_accuracy, train_seconds, samples_per_sec, switched, subset, label,
  notes, status, error

results.csv is PER-EPOCH (the prompt's schema). test_loss is always NA (the
engine does not compute it). test_accuracy is attached to the final epoch only.
"""
import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "processed"

RESULTS_COLS = [
    "run_id", "timestamp", "dataset", "model", "strategy", "nodes", "workers",
    "servers", "synchronizer", "store", "batch_size_per_worker",
    "effective_global_batch", "optimizer", "learning_rate", "seed", "max_epochs",
    "offline_epochs", "epoch", "train_loss", "test_loss", "test_accuracy",
    "epoch_time_sec", "total_time_sec", "samples_per_sec", "status", "notes",
]


def load_raw():
    rows = []
    for path in sorted(glob.glob(str(RAW / "*.jsonl"))):
        ts = Path(path).stem.split("_")[0]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # This monography compares only All-Reduce vs Parameter Server.
                if r.get("strategy") == "strategy_switch":
                    continue
                r["_timestamp"] = r.get("timestamp") or ts
                rows.append(r)
    return rows


def g(r, k, default=""):
    v = r.get(k)
    return default if v is None else v


def write_results_csv(raw):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLS, extrasaction="ignore")
        w.writeheader()
        for r in raw:
            workers, servers = g(r, "workers", 0), g(r, "servers", 0)
            nodes = (workers or 0) + (servers or 0)
            loss_hist = r.get("loss_history") or []
            epochs_ran = r.get("epochs_ran") or len(loss_hist)
            total = r.get("train_seconds")
            epoch_t = (total / epochs_ran) if (total and epochs_ran) else ""
            run_id = g(r, "label") or g(r, "run_key") or "run"
            base = dict(
                run_id=run_id, timestamp=r.get("_timestamp", ""),
                dataset=g(r, "dataset"), model=g(r, "model"),
                strategy=g(r, "strategy"), nodes=nodes, workers=workers,
                servers=servers, synchronizer=g(r, "synchronizer", "NA"),
                store=g(r, "store", "NA"),
                batch_size_per_worker=g(r, "batch_size_per_worker"),
                effective_global_batch=g(r, "effective_global_batch"),
                optimizer=g(r, "optimizer", "sgd"),
                learning_rate=g(r, "learning_rate"), seed=g(r, "seed"),
                max_epochs=g(r, "max_epochs"), offline_epochs=g(r, "offline_epochs"),
                test_loss="NA", total_time_sec=g(r, "train_seconds"),
                epoch_time_sec=round(epoch_t, 4) if epoch_t != "" else "",
                samples_per_sec=g(r, "samples_per_sec"),
                status=g(r, "status", "ok"), notes=g(r, "notes"),
            )
            if r.get("status") == "error" or not loss_hist:
                w.writerow({**base, "epoch": "", "train_loss": "",
                            "test_accuracy": g(r, "test_accuracy", "NA")})
                continue
            n = len(loss_hist)
            for i, tl in enumerate(loss_hist):
                acc = g(r, "test_accuracy", "NA") if i == n - 1 else "NA"
                w.writerow({**base, "epoch": i + 1, "train_loss": tl,
                            "test_accuracy": acc})


def write_summaries(raw):
    ok = [r for r in raw if r.get("status") != "error" and r.get("loss_history")]

    # summary_by_strategy.csv — one row per (dataset, model, strategy, nodes)
    with open(OUT / "summary_by_strategy.csv", "w", newline="") as f:
        cols = ["label", "dataset", "model", "strategy", "synchronizer", "store",
                "nodes", "workers", "servers", "batch_size_per_worker",
                "effective_global_batch", "final_loss", "test_accuracy",
                "total_time_sec", "samples_per_sec", "switched"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(ok, key=lambda r: (g(r, "dataset"), g(r, "model"),
                                           g(r, "strategy"), g(r, "workers", 0))):
            nodes = (r.get("workers") or 0) + (r.get("servers") or 0)
            w.writerow({k: g(r, k) for k in cols} | {
                "nodes": nodes, "final_loss": g(r, "final_loss"),
                "total_time_sec": g(r, "train_seconds")})

    # scalability_summary.csv — samples/s, speedup, efficiency vs a 1-node ref
    with open(OUT / "scalability_summary.csv", "w", newline="") as f:
        cols = ["dataset", "model", "strategy", "nodes", "workers",
                "samples_per_sec", "speedup_vs_min", "efficiency"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        groups = {}
        for r in ok:
            # Throughput/scaling must use the fixed-per-worker runs only (label Bp_);
            # mixing in the fixed-global or sanity runs would compare different batches.
            if not str(g(r, "label")).startswith("Bp_"):
                continue
            key = (g(r, "dataset"), g(r, "model"), g(r, "strategy"))
            groups.setdefault(key, []).append(r)
        for (ds, m, st), rs in sorted(groups.items()):
            rs = [r for r in rs if r.get("samples_per_sec")]
            if not rs:
                continue
            ref = min(rs, key=lambda r: r.get("workers", 0))
            base = ref["samples_per_sec"]
            base_w = ref.get("workers", 1) or 1
            for r in sorted(rs, key=lambda r: r.get("workers", 0)):
                wk = r.get("workers", 1) or 1
                sp = r["samples_per_sec"] / base if base else ""
                eff = (sp / (wk / base_w)) if (sp and base_w) else ""
                w.writerow({"dataset": ds, "model": m, "strategy": st,
                            "nodes": (r.get("workers", 0) + r.get("servers", 0)),
                            "workers": wk, "samples_per_sec": r["samples_per_sec"],
                            "speedup_vs_min": round(sp, 3) if sp else "",
                            "efficiency": round(eff, 3) if eff else ""})

    # time_to_accuracy.csv — epochs/time to reach within 10% of own final loss
    with open(OUT / "time_to_accuracy.csv", "w", newline="") as f:
        cols = ["dataset", "model", "strategy", "nodes", "test_accuracy",
                "epochs_to_converge", "time_to_converge_sec", "total_time_sec"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in ok:
            lh = r["loss_history"]
            target = (r.get("final_loss") or lh[-1]) * 1.10
            epc = next((i + 1 for i, v in enumerate(lh) if v <= target), len(lh))
            total = r.get("train_seconds")
            ttc = (total * epc / len(lh)) if (total and lh) else ""
            w.writerow({"dataset": g(r, "dataset"), "model": g(r, "model"),
                        "strategy": g(r, "strategy"),
                        "nodes": (r.get("workers", 0) + r.get("servers", 0)),
                        "test_accuracy": g(r, "test_accuracy"),
                        "epochs_to_converge": epc,
                        "time_to_converge_sec": round(ttc, 2) if ttc != "" else "",
                        "total_time_sec": total})


def write_descriptors(raw):
    """Static descriptors required by the spec: dataset_summary + model_summary."""
    # dataset_summary.csv
    datasets = {
        "mnist": ("60000", "10000", "28x28x1", "10"),
        "fashion_mnist": ("60000", "10000", "28x28x1", "10"),
    }
    if list((ROOT / "datasets" / "emnist").glob("*train*samples*.bin")):
        datasets["emnist"] = ("240000", "40000", "28x28x1", "10")
    with open(OUT / "dataset_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "train_size", "test_size", "input_shape", "classes"])
        for name, (tr, te, sh, cl) in datasets.items():
            w.writerow([name, tr, te, sh, cl])

    # model_summary.csv — architecture + parameter count from the harness.
    try:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import ono_harness as oh
        used = {r.get("model") for r in raw if r.get("model")}
        with open(OUT / "model_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "params", "used"])
            for name in ["nielsen", "lenet5", "mlp_small", "mlp_medium", "mlp_large"]:
                if name in oh.MODELS:
                    w.writerow([name, oh.param_count(name), name in used])
    except Exception as e:
        print("model_summary skipped:", e)


def main():
    raw = load_raw()
    if not raw:
        print("no raw results found in", RAW)
        return
    write_results_csv(raw)
    write_summaries(raw)
    write_descriptors(raw)
    print(f"aggregated {len(raw)} runs -> {OUT}/results.csv (+ summaries + descriptors)")


if __name__ == "__main__":
    main()
