"""Regenerate benchmarks/README.md from the merged result history.

The whole file is rebuilt every run from history, so suites that were not
executed keep their previous numbers (still in history) and their plot files
on disk are left untouched.
"""

from pathlib import Path

from .plots import STRAT, VARIANT_SHORT, _topo
from .suites import (ALL_MODELS, ALL_SUITES, AR_WORKER_SCALE, MODELS,
                     SPEED_EPOCHS, SPEED_SUBSET)

README = Path(__file__).resolve().parent.parent / "README.md"
PLOTS = Path(__file__).resolve().parent.parent / "plots"

SUITE_DOCS = {
    "convergence": (
        "Convergence",
        "**Measures:** loss vs epoch and final test accuracy. Strategies are compared "
        "at a **fixed 5-node budget** (PS and SS are 3 workers + 2 servers; all-reduce "
        "runs 5 workers) so they spend the same hardware. We hold **nodes** fixed rather "
        "than **workers** because strategy switch all-reduces over *every* node until it "
        "switches, so a 3w/2s SS is really a 5-node run; at equal node budget it sits "
        "honestly next to all-reduce with that many workers instead of being mislabeled "
        "as a 3-worker run. (Training is **Local SGD**: every worker runs SGD locally at "
        "`batch` per step over its data shard, then the updates are averaged across "
        "workers each epoch — the per-step batch is **not** `workers × batch`.) The "
        "all-reduce **worker-count sweep** lives in its own figure (more workers = smaller "
        "shards + more averaging). The dashed line is the single-process PyTorch reference "
        "(same recipe + same early-stopping rule).\n"
        "**Does NOT measure:** wall-clock speed.",
    ),
    "execution-speed": (
        "Execution speed",
        f"**Measures:** **samples/sec** (batch-invariant throughput) on a fixed "
        f"**{SPEED_SUBSET:,}-sample** training subset over a short **{SPEED_EPOCHS}-epoch** "
        f"budget. Compares raising `offline_epochs` vs raising `batch_size`.\n"
        "**Does NOT measure:** accuracy or convergence.",
    ),
    "convergence-speed": (
        "Convergence speed",
        "**Measures:** loss reduction/sec and accuracy/sec under one shared fixed budget "
        "at the same 5-node budget (only the strategy changes), plus raw **samples/sec** as "
        "a throughput reference decoupled from convergence quality. SS rows state whether "
        "the switch fired.\n"
        "**Does NOT measure:** peak accuracy.",
    ),
    "scalability": (
        "Scalability",
        f"**Measures:** how **samples/sec** changes as workers increase, on the same fixed "
        f"**{SPEED_SUBSET:,}-sample** subset, in **separate panels** for all-reduce and "
        "parameter server — PS spends extra server nodes, so the two do not share a "
        "'workers' axis honestly (the node count is shown per point).\n"
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
        header = ["Model", "Strategy", "Topology", "lr", "Batch/wkr", "Epochs", "Final loss", "Accuracy"]
        rows = [[r["model"], _strat_cell(r), _topo(r), _fmt(r.get("lr")),
                 r.get("batch_size", 0),
                 r.get("epochs_ran", "—"), _fmt(r.get("final_loss")),
                 _fmt_pm(r, "accuracy", "{:.3f}")] for r in runs]
    elif suite == "execution-speed":
        header = ["Model", "Strategy", "Topology", "offline", "batch", "Samples/sec"]
        rows = [[r["model"], _strat_cell(r), _topo(r), r["offline_epochs"], r["batch_size"],
                 _fmt_pm(r, "samples_per_sec", "{:.0f}")] for r in runs]
    elif suite == "convergence-speed":
        # samples/sec is raw throughput (decoupled from convergence quality), so it
        # complements the loss/accuracy-per-sec metrics rather than duplicating them.
        header = ["Model", "Strategy", "Topology", "Samples/sec", "Loss/sec", "Accuracy/sec"]
        rows = [[r["model"], _strat_cell(r), _topo(r),
                 _fmt_pm(r, "samples_per_sec", "{:.0f}"),
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
        "- **Nielsen MNIST**: `28×28×1 → conv(20, 5×5) → maxpool(2×2) → dense(100, sigmoid) → dense(10) → softmax`.",
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
        "`batch` is the **per-worker** mini-batch (the dataset is sharded across workers, "
        "and each worker runs SGD locally at this batch before the per-epoch averaging). "
        "Nielsen uses the canonical `network3.py` recipe (60 / 10 / 0.1 → ~98.8%); the "
        "small batch is what lets the distributed runs converge.",
        "",
        "## Strategies & variants",
        "",
        "- **PS (blocking)** — `BlockingStore` + `BarrierSync`: workers wait for a full round.",
        "- **PS (non-block)** — `BlockingStore` + `NonBlockingSync`: same consistent store, "
        "but workers apply and move on without the barrier (see Methodology for why the "
        "lock-free HogWild store is excluded).",
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
        "- **Local SGD, not large-batch.** Each worker runs SGD locally at the per-worker "
        "`batch` over its data shard, then the updates are averaged across workers each "
        "epoch (FedAvg-style). The per-step batch is **not** `workers × batch`. Adding "
        "workers shards the data finer and averages more often, which is what shifts "
        "convergence — so the strategy comparison fixes the total node budget (workers + "
        "servers) and the worker-count sweep is shown separately. The single-process "
        "baseline uses the same per-step batch, so it is a like-for-like reference, not an "
        "over-batched one.",
        "- **Batch differs across suites.** Speed/scalability use a larger batch on a "
        "subset (they only need throughput), so their numbers do **not** transfer to the "
        "convergence config (e.g. Nielsen converges at batch 10 but is benchmarked for "
        "speed at batch 64/256 — ~6× faster there).",
        "- **Throughput = samples/sec, not epochs/sec.** By construction "
        "`epochs_per_sec = samples_per_sec / samples_per_epoch`, and samples-per-epoch is "
        "**fixed within each suite** (the subset for the speed suites, the full set for the "
        "convergence ones). So next to samples/sec, epochs/sec is the *same* ranking rescaled "
        "by a constant — it carries no extra information within a figure, which is why it is "
        "**not** added per-suite. samples/sec is the one kept because it is invariant: it "
        "counts real work, so it stays comparable across batch sizes and across datasets, "
        "whereas an 'epoch' means 4k samples in the speed suites but 60k in the convergence "
        "ones — not comparable.",
        f"- **Data sizes.** Convergence and convergence-speed train on the **full 60,000-sample "
        f"MNIST** training set; execution-speed and scalability train on a fixed "
        f"**{SPEED_SUBSET:,}-sample** subset (throughput only). Accuracy is always scored on "
        "the full **10,000-sample** test set (`t10k`).",
        "- **PS synchronizer variants.** Parameter server is benchmarked in two variants that "
        "share the **consistent `BlockingStore`** and differ only in the synchronizer, so the "
        "comparison isolates the barrier: **blocking** (`BarrierSync` — every worker waits for "
        "the full aggregation round) vs **non_blocking** (`NonBlockingSync` — workers apply and "
        "move on without the barrier). The lock-free `WildStore` (HogWild) is **deliberately "
        "excluded**: its data races make a run hang non-deterministically (it completed some "
        "trial runs and hung mid-epoch on others), so it has no fair, reproducible number.",
        "- **Accuracy** is measured **once, after training**, over the **full 10,000-sample "
        "MNIST test set** (`t10k`): the trained weights are loaded into the PyTorch argmax "
        "reference and scored on every test sample — not sampled, not evaluated periodically, "
        "so there is no mid-training eval cost folded into `train_seconds`.",
        "- **What `train_seconds` measures.** Host wall-clock of `orchestrate().wait()`. "
        "Every node runs as a Docker container **on a single host**, so (a) compute runs in "
        "genuine parallel only up to the host core count (~8 cores here — readings past ~8 "
        "total nodes oversubscribe and stop being honest) and (b) the network is **loopback**, "
        "so inter-node communication is far cheaper than a real multi-machine cluster. Read "
        "samples/sec and scalability as a **single-host lower bound on comms cost**, not a "
        "true distributed measurement.",
        "- **`loss/sec` and `accuracy/sec` divide by total time**, so they reward raw speed: "
        "a faster strategy that plateaus at a slightly lower final accuracy can still score "
        "higher than a slower one that converges better. Always read them next to the final "
        "accuracy, never alone.",
        "- **PS blocking vs non_blocking is compared on accuracy, not speed.** The two PS "
        "variants are contrasted by their **converged accuracy** (reproducible from the seed, "
        "host-load-independent): removing the barrier costs almost nothing "
        "(nielsen 98.30 → 98.13, lenet5 98.0 ≈ 98.0). Their **timing** is *not* compared: "
        "convergence-speed runs once (repeats=1), and single-shot `train_seconds` on this host "
        "carries ~±25% between-run noise (seen directly in the prior campaign, where the same "
        "blocking config timed 831 s and 658 s), which swamps any real blocking/non_blocking "
        "difference. The non_blocking rows are also from a later campaign than AR/SS, so "
        "cross-strategy *speed* here is only indicative — a same-session, repeated re-benchmark "
        "is pending.",
        "- **Loss scale.** The distributed `loss` is the mean across workers of each "
        "worker's epoch-mean cross-entropy; the PyTorch baseline is the global epoch-mean "
        "cross-entropy — same scale, directly comparable. (Early stopping on the "
        "distributed side keys off the per-epoch **max** worker loss; for the 1-worker "
        "baseline max = mean.)",
        "- **Cross-entropy on logits, softmax once.** The PyTorch baseline computes the "
        "loss with `F.cross_entropy` on raw logits, which folds the softmax into the loss, "
        "so the distribution is normalized once — same softmax-then-cross-entropy the "
        "orchestra trainer applies. The final softmax is only materialized for the argmax "
        "accuracy, where it does not change the result.",
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
