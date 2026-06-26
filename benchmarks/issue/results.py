"""Result persistence: append-only JSONL plus a derived CSV, merged by run key.

History merges newest-over-oldest per run key, so a partial run only replaces
the keys it executed and leaves every other suite's results untouched.
"""

import csv
import json
import statistics
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCH_DIR / "results"
META_PATH = RESULTS_DIR / "run_meta.json"

CSV_FIELDS = [
    "run_key", "suite", "model", "strategy", "ps_variant", "baseline", "switched",
    "workers", "servers", "lr", "batch_size", "offline_epochs", "max_epochs",
    "repeats", "epochs_ran", "train_seconds", "epochs_per_sec", "samples_per_sec",
    "final_loss", "loss_per_sec", "accuracy", "accuracy_per_sec", "error",
]

# Scalar metrics averaged across repeated runs of the same config (+ a *_std field).
AGG_SCALARS = ["train_seconds", "epochs_per_sec", "samples_per_sec", "epochs_ran",
               "final_loss", "loss_per_sec", "accuracy", "accuracy_per_sec"]


def aggregate(reps):
    """Collapse repeated runs of one config into a single record: each scalar
    metric becomes its mean with a `<metric>_std` companion, the loss curve is
    the repeat closest to the mean final loss, and `switched` is OR-ed."""
    oks = [r for r in reps if not r.get("error")]
    if not oks:
        return reps[0]
    agg = dict(oks[0])
    for k in AGG_SCALARS:
        vals = [r[k] for r in oks if r.get(k) is not None]
        if vals:
            agg[k] = statistics.fmean(vals)
            agg[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        else:
            agg[k] = agg[f"{k}_std"] = None
    mean_fl = agg.get("final_loss")
    if mean_fl is not None:
        rep = min((r for r in oks if r.get("final_loss") is not None),
                  key=lambda r: abs(r["final_loss"] - mean_fl), default=oks[0])
        agg["loss_history"] = rep.get("loss_history")
    switched = [r.get("switched") for r in oks if r.get("switched") is not None]
    agg["switched"] = (any(switched) if switched else None)
    agg["repeats"] = len(oks)
    agg["error"] = None
    return agg


def save_result(result, jsonl_path):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_history():
    """Return {run_key: result}, newest file wins for a repeated key."""
    history = {}
    if not RESULTS_DIR.exists():
        return history
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # A valid run_key has the full suite|model|strategy|... shape.
                    # Skip malformed keys left by older/aborted runs.
                    if "|" in rec.get("run_key", ""):
                        history[rec["run_key"]] = rec
        except OSError:
            pass
    return history


def merge(history, new_results):
    merged = dict(history)
    for r in new_results:
        if "run_key" in r:
            merged[r["run_key"]] = r
    return merged


def save_run_meta(full_run_seconds, timestamp):
    """Record wall-clock time of the last full (all suites + models) run."""
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps({"full_run_seconds": full_run_seconds, "timestamp": timestamp}))


def load_run_meta():
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def write_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r.get("suite", ""), r.get("model", ""),
                                                r.get("strategy", ""), r.get("workers", 0))):
            writer.writerow(r)
