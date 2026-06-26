"""Regenerate benchmarks/README.md from the merged result history.

The whole file is rebuilt every run from history, so suites that were not
executed keep their previous numbers (still in history) and their plot files
on disk are left untouched.
"""

from pathlib import Path

from .plots import STRAT, VARIANT_SHORT, _topo
from .suites import ALL_MODELS, ALL_SUITES, AR_WORKER_SCALE, MODELS

README = Path(__file__).resolve().parent.parent / "README.md"
PLOTS = Path(__file__).resolve().parent.parent / "plots"

SUITE_DOCS = {
    "convergence": (
        "Convergence",
        "**Measures:** loss vs epoch and final test accuracy. Strategies are compared "
        f"at a **fixed {3}-worker** topology so the effective batch (workers × batch) "
        "is held constant — the only fair way to attribute differences to the strategy "
        "and not to the batch. The all-reduce **worker-count sweep** lives in its own "
        "figure (effective batch changes there). The dashed line is the single-process "
        "PyTorch reference (same recipe + same early-stopping rule).\n"
        "**Does NOT measure:** wall-clock speed.",
    ),
    "execution-speed": (
        "Execution speed",
        "**Measures:** **samples/sec** (batch-invariant throughput) on a small subset. "
        "Compares raising `offline_epochs` vs raising `batch_size`.\n"
        "**Does NOT measure:** accuracy or convergence.",
    ),
    "convergence-speed": (
        "Convergence speed",
        "**Measures:** loss reduction/sec and accuracy/sec under one shared fixed budget "
        "at the same worker count (only the strategy changes). SS rows state whether the "
        "switch fired.\n"
        "**Does NOT measure:** peak accuracy.",
    ),
    "scalability": (
        "Scalability",
        "**Measures:** how **samples/sec** changes as workers increase, in **separate "
        "panels** for all-reduce and parameter server — PS spends extra server nodes, so "
        "the two do not share a 'workers' axis honestly (the node count is shown per point).\n"
        "**Does NOT measure:** convergence (re-uses the speed budget).",
    ),
}


def _img(name):
    return f"![]({PLOTS.name}/{name})" if (PLOTS / name).exists() else ""


def _table(rows, header):
    sep = "|".join("---" for _ in header)
    lines = ["| " + " | ".join(header) + " |", f"|{sep}|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _fmt(v, spec="{:.3g}"):
    return spec.format(v) if isinstance(v, (int, float)) else "—"


def _strat_cell(r):
    """Strategy label for a table row: PS variant, baseline tag, or SS switch state."""
    if r.get("baseline"):
        return "PyTorch (ref)"
    s = STRAT[r["strategy"]]
    if r.get("ps_variant"):
        s += f" ({VARIANT_SHORT.get(r['ps_variant'], r['ps_variant'])})"
    if r["strategy"] == "strategy_switch" and r.get("switched") is not None:
        s += " · switched" if r["switched"] else " · no switch"
    return s


def _fmt_pm(r, metric, spec="{:.3g}"):
    """Format a metric as `mean ± std` when a std companion is present."""
    val = r.get(metric)
    if val is None:
        return "—"
    std = r.get(f"{metric}_std")
    if std:
        return f"{spec.format(val)} ± {spec.format(std)}"
    return spec.format(val)


def _suite_rows(suite, history):
    runs = sorted((r for r in history.values() if r.get("suite") == suite),
                  key=lambda r: (r["model"], r.get("baseline", False), r["strategy"],
                                 r.get("ps_variant") or "", r["workers"]))
    if suite == "convergence":
        header = ["Model", "Strategy", "Topology", "Eff. batch", "Epochs", "Final loss", "Accuracy"]
        rows = [[r["model"], _strat_cell(r), _topo(r),
                 r.get("workers", 1) * r.get("batch_size", 0),
                 r.get("epochs_ran", "—"), _fmt(r.get("final_loss")),
                 _fmt_pm(r, "accuracy", "{:.3f}")] for r in runs]
    elif suite == "execution-speed":
        header = ["Model", "Strategy", "Topology", "offline", "batch", "Samples/sec", "Epochs/sec"]
        rows = [[r["model"], _strat_cell(r), _topo(r), r["offline_epochs"], r["batch_size"],
                 _fmt_pm(r, "samples_per_sec", "{:.0f}"), _fmt_pm(r, "epochs_per_sec")] for r in runs]
    elif suite == "convergence-speed":
        header = ["Model", "Strategy", "Topology", "Loss/sec", "Accuracy/sec"]
        rows = [[r["model"], _strat_cell(r), _topo(r),
                 _fmt_pm(r, "loss_per_sec"), _fmt_pm(r, "accuracy_per_sec")] for r in runs]
    else:
        header = ["Model", "Strategy", "Workers", "Nodes", "Samples/sec"]
        rows = [[r["model"], _strat_cell(r), r["workers"], r["workers"] + r["servers"],
                 _fmt_pm(r, "samples_per_sec", "{:.0f}")] for r in runs]
    return header, rows


def _suite_images(suite):
    names = {
        "convergence": [f"convergence_loss_{m}.png" for m in ALL_MODELS]
        + [f"convergence_accuracy_{m}.png" for m in ALL_MODELS]
        + [f"convergence_workers_{m}.png" for m in ALL_MODELS],
        "execution-speed": [f"execution_speed_{m}.png" for m in ALL_MODELS],
        "convergence-speed": [f"convergence_speed_{m}.png" for m in ALL_MODELS],
        "scalability": [f"scalability_{m}.png" for m in ALL_MODELS],
    }[suite]
    return [img for img in (_img(n) for n in names) if img]


def _fmt_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _hyper_table():
    header = ["Model", "lr", "Batch", "Epochs", "Loss"]
    rows = []
    for name, model in MODELS.items():
        ref = model.reference
        rows.append([name, ref.lr, ref.batch_size, ref.max_epochs, str(model.loss_fn)])
    return _table(rows, header)


def render(history, meta=None):
    p = [
        "# Strategy Benchmarks",
        "",
        "Compares the three distributed strategies — **parameter server**, **all-reduce** "
        "and **strategy switch** — on two models (**LeNet5** and **Nielsen MNIST**) across "
        "four focused suites. Each suite states what it measures and what it does not.",
        "",
        "## Models",
        "",
        "- **Nielsen MNIST**: `28×28×1 → conv(20, 5×5) → maxpool(2×2) → dense(100) → dense(10) → softmax`.",
        "- **LeNet5**: `conv(6, 5×5, pad2) → maxpool → conv(16, 5×5) → maxpool → dense(120) → dense(84) → dense(10)`, tanh + softmax.",
        "",
        "### Hyper-parameters",
        "",
        "Each model owns its *reference* recipe (in `issue/suites.py`). Convergence "
        "suites train with it; speed/scalability suites override `batch`/`epochs` "
        "since they do not need to converge.",
        "",
        _hyper_table(),
        "",
        "`batch` is applied **per worker** (the dataset is sharded across workers), "
        "so the effective global batch is `batch × workers`. Nielsen uses the "
        "canonical `network3.py` recipe (60 / 10 / 0.1 → ~98.8%); the small batch is "
        "what lets the distributed runs converge.",
        "",
        "## Strategies & variants",
        "",
        "- **PS (blocking)** — `BlockingStore` + `BarrierSync`: workers wait for a full round.",
        "- **AR** — all-reduce ring (averaged gradients).",
        "- **SS** — strategy switch (starts in all-reduce, may switch to PS).",
        "- **PyTorch (ref)** — single-process PyTorch training of the same architecture "
        "and recipe, drawn as a dashed reference line in the convergence plots.",
        "",
        "## Running",
        "",
        "```bash",
        ".venv/bin/python benchmarks/run_issue_benchmarks.py                 # all suites, both models",
        ".venv/bin/python benchmarks/run_issue_benchmarks.py --suite convergence",
        ".venv/bin/python benchmarks/run_issue_benchmarks.py --suite scalability --model lenet5",
        ".venv/bin/python benchmarks/run_issue_benchmarks.py --plots-only    # rebuild plots/README from history",
        "```",
        "",
        "Partial runs only re-run and re-plot the selected suite/model; every other "
        "suite keeps its previous results and figures.",
        "",
        f"All-reduce worker scale: {AR_WORKER_SCALE} (configurable in `issue/suites.py`; "
        "the issue suggests 3/7/11 — kept lighter to fit one host).",
        "",
    ]
    if meta and meta.get("full_run_seconds"):
        p += [f"_Last full run: {_fmt_duration(meta['full_run_seconds'])}"
              + (f" ({meta['timestamp']})" if meta.get("timestamp") else "") + "._", ""]
    for suite in ALL_SUITES:
        title, doc = SUITE_DOCS[suite]
        header, rows = _suite_rows(suite, history)
        p += [f"## {title}", "", doc, ""]
        if rows:
            p += [_table(rows, header), ""]
        else:
            p += ["_No results yet._", ""]
        p += [img for img in _suite_images(suite)]
        p += [""]
    p += [
        "## Methodology & fairness",
        "",
        "- **Effective batch.** `batch` is per worker, so the effective global batch is "
        "`workers × batch`. Convergence compares strategies at a fixed worker count to "
        "hold it constant; the worker-count sweep is shown separately.",
        "- **Batch differs across suites.** Speed/scalability use a larger batch on a "
        "subset (they only need throughput), so their numbers do **not** transfer to the "
        "convergence config (e.g. Nielsen converges at batch 10 but is benchmarked for "
        "speed at batch 64/256 — ~6× faster there).",
        "- **Throughput = samples/sec**, not epochs/sec: an epoch with a bigger batch has "
        "fewer steps, so epochs/sec would reward batch size by construction.",
        "- **Loss scale.** The distributed `loss` is the mean across workers of each "
        "worker's epoch-mean cross-entropy; the PyTorch baseline is the global epoch-mean "
        "cross-entropy — same scale, directly comparable. (Early stopping on the "
        "distributed side keys off the per-epoch **max** worker loss; for the 1-worker "
        "baseline max = mean.)",
        "- **Early stopping** (both distributed and baseline): stop after 3 consecutive "
        "epochs whose loss delta stays within `1e-4` (mirrors the orchestrator's "
        "`ConvergenceTracker`).",
        "- **Repeats.** `--repeats N` repeats the throughput suites (execution-speed, "
        "scalability) N times — tables show `mean ± std` and bars carry error bars. The "
        "expensive convergence suites always run once (their loss/sec numbers are single-shot).",
        "",
        "## Raw results",
        "",
        "- Per-run records: `results/*.jsonl` (append-only, gitignored).",
        "- Flattened table: `results/summary.csv`.",
        "- Trained weights: `results/artifacts/*.safetensors`.",
        "",
        "**Metrics:** `epochs_per_sec = epochs / train_seconds`; "
        "`samples_per_sec = samples × epochs / train_seconds`; "
        "`loss_per_sec = (first_loss − final_loss) / train_seconds`; "
        "`accuracy_per_sec = accuracy / train_seconds`.",
        "",
    ]
    return "\n".join(p)


def write(history, meta=None):
    README.write_text(render(history, meta))
    return README
