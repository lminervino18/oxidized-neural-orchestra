"""Regenerate benchmarks/README.md from the merged result history.

The whole file is rebuilt every run from history, so suites that were not
executed keep their previous numbers (still in history) and their plot files
on disk are left untouched.
"""

from pathlib import Path

from .plots import STRAT, _topo
from .suites import ALL_MODELS, ALL_SUITES, AR_WORKER_SCALE

README = Path(__file__).resolve().parent.parent / "README.md"
PLOTS = Path(__file__).resolve().parent.parent / "plots"

SUITE_DOCS = {
    "convergence": (
        "Convergence",
        "**Measures:** loss vs epoch and final test accuracy per strategy/topology.\n"
        "**Does NOT measure:** wall-clock speed.",
    ),
    "execution-speed": (
        "Execution speed",
        "**Measures:** epochs/sec on a small subset (no convergence). Compares "
        "raising `offline_epochs` vs raising `batch_size`.\n"
        "**Does NOT measure:** accuracy or convergence.",
    ),
    "convergence-speed": (
        "Convergence speed",
        "**Measures:** loss reduction/sec and accuracy/sec under one shared budget "
        "(same epochs and params; only the strategy changes).\n"
        "**Does NOT measure:** peak accuracy.",
    ),
    "scalability": (
        "Scalability",
        "**Measures:** how throughput (epochs/sec) changes as nodes increase.\n"
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


def _suite_rows(suite, history):
    runs = sorted((r for r in history.values() if r.get("suite") == suite),
                  key=lambda r: (r["model"], r["strategy"], r["workers"]))
    if suite == "convergence":
        header = ["Model", "Strategy", "Topology", "Epochs", "Final loss", "Accuracy"]
        rows = [[r["model"], STRAT[r["strategy"]], _topo(r), r.get("epochs_ran", "—"),
                 _fmt(r.get("final_loss")), _fmt(r.get("accuracy"), "{:.3f}")] for r in runs]
    elif suite == "execution-speed":
        header = ["Model", "Strategy", "Topology", "offline", "batch", "Epochs/sec"]
        rows = [[r["model"], STRAT[r["strategy"]], _topo(r), r["offline_epochs"],
                 r["batch_size"], _fmt(r.get("epochs_per_sec"))] for r in runs]
    elif suite == "convergence-speed":
        header = ["Model", "Strategy", "Topology", "Loss/sec", "Accuracy/sec"]
        rows = [[r["model"], STRAT[r["strategy"]], _topo(r),
                 _fmt(r.get("loss_per_sec")), _fmt(r.get("accuracy_per_sec"))] for r in runs]
    else:
        header = ["Model", "Strategy", "Workers", "Epochs/sec"]
        rows = [[r["model"], STRAT[r["strategy"]], r["workers"],
                 _fmt(r.get("epochs_per_sec"))] for r in runs]
    return header, rows


def _suite_images(suite):
    names = {
        "convergence": [f"convergence_loss_{m}.png" for m in ALL_MODELS]
        + [f"convergence_accuracy_{m}.png" for m in ALL_MODELS],
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
        "## Raw results",
        "",
        "- Per-run records: `results/*.jsonl` (append-only, gitignored).",
        "- Flattened table: `results/summary.csv`.",
        "- Trained weights: `results/artifacts/*.safetensors`.",
        "",
        "**Metrics:** `epochs_per_sec = epochs / train_seconds`; "
        "`loss_per_sec = (first_loss − final_loss) / train_seconds`; "
        "`accuracy_per_sec = accuracy / train_seconds`.",
        "",
    ]
    return "\n".join(p)


def write(history, meta=None):
    README.write_text(render(history, meta))
    return README
