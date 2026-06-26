"""Suite plots. Each suite writes files under its own prefix so a partial run
only regenerates its own figures and never blanks another suite's plots.

Fairness-driven choices:
- Convergence compares strategies at a FIXED worker count (effective batch =
  workers x batch is held constant); the worker-count sweep gets its own figure.
- Scalability separates all-reduce and parameter server (PS spends extra server
  nodes, so they don't share a "workers" axis honestly).
- Throughput is reported as samples/sec (batch-invariant), not epochs/sec.
- Strategy-switch labels state whether the switch actually happened.
- With repeats>1, bars carry mean ± std error bars.
"""

from collections import defaultdict
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

STRAT = {"parameter_server": "PS", "all_reduce": "AR", "strategy_switch": "SS"}
VARIANT_SHORT = {"blocking": "blocking", "non_blocking": "non-block"}
BASELINE_LABEL = "PyTorch (ref)"

# Convergence compares strategies at this fixed worker count (same effective batch).
CONV_WORKERS = 3


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _topo(r):
    return f"{r['workers']}w" if r["servers"] == 0 else f"{r['workers']}w/{r['servers']}s"


def _nodes(r):
    return r["workers"] + r["servers"]


def _eff_batch(r):
    return r["workers"] * r["batch_size"]


def _switch_tag(r):
    sw = r.get("switched")
    if sw is None:
        return ""
    return " (switched)" if sw else " (no switch)"


def _label(r):
    if r.get("baseline"):
        return BASELINE_LABEL
    name = STRAT[r["strategy"]]
    if r.get("ps_variant"):
        name += f"-{VARIANT_SHORT.get(r['ps_variant'], r['ps_variant'])}"
    if r["strategy"] == "strategy_switch":
        name += _switch_tag(r)
    return f"{name} {_topo(r)}"


def _by_model(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[r["model"]].append(r)
    return grouped


def _save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out


def _ok(results, suite):
    return [r for r in results if r.get("suite") == suite and not r.get("error")]


def _errs(runs, metric):
    """Per-bar std (0 when absent). Returns None if no run carries a std."""
    stds = [r.get(f"{metric}_std") for r in runs]
    if not any(s for s in stds):
        return None
    return [s or 0.0 for s in stds]


# ── Convergence ──────────────────────────────────────────────────────────────

def plot_convergence(results, ts):
    """Main convergence figure: strategies at a FIXED worker count + baseline."""
    plt = _plt()
    out = []
    for model, runs in _by_model(_ok(results, "convergence")).items():
        main = [r for r in runs if r.get("baseline") or r["workers"] == CONV_WORKERS]

        curves = [r for r in main if r.get("loss_history")]
        if curves:
            fig, ax = plt.subplots(figsize=(8, 5))
            for r in sorted(curves, key=lambda r: (r.get("baseline", False), _label(r))):
                losses = r["loss_history"]
                xs = range(1, len(losses) + 1)
                if r.get("baseline"):
                    ax.plot(xs, losses, label=_label(r), linewidth=2.0,
                            linestyle="--", color="black", zorder=5)
                else:
                    ax.plot(xs, losses, label=_label(r), linewidth=1.5)
            ax.set_title(f"Convergence · {model} · strategies @ {CONV_WORKERS}w  ({ts})",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            out.append(_save(fig, f"convergence_loss_{model}.png"))
            plt.close(fig)

        acc = [r for r in main if r.get("accuracy") is not None]
        if acc:
            acc.sort(key=lambda r: (r.get("baseline", False), _label(r)))
            fig, ax = plt.subplots(figsize=(max(5.5, len(acc) * 1.5), 4.8))
            vals = [r["accuracy"] for r in acc]
            errs = _errs(acc, "accuracy")
            colors = ["#9e9e9e" if r.get("baseline") else "#4caf50" for r in acc]
            ax.bar(range(len(acc)), vals, color=colors, width=0.6,
                   yerr=errs, capsize=4, ecolor="#37474f")
            for i, v in enumerate(vals):
                off = (errs[i] if errs else 0) + 0.02
                ax.text(i, min(v + off, 1.02), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(range(len(acc)))
            ax.set_xticklabels([_label(r) for r in acc], rotation=25, ha="right", fontsize=8)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("Test accuracy")
            ax.set_title(f"Final accuracy · {model} · @ {CONV_WORKERS}w  ({ts})",
                         fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            out.append(_save(fig, f"convergence_accuracy_{model}.png"))
            plt.close(fig)
    return out


def plot_convergence_workers(results, ts):
    """Separate figure: all-reduce convergence vs worker count, showing how the
    effective batch (workers x batch) changes the curve — kept apart from the
    strategy comparison so the two questions don't get conflated."""
    plt = _plt()
    out = []
    for model, runs in _by_model(_ok(results, "convergence")).items():
        ar = [r for r in runs if r["strategy"] == "all_reduce"
              and not r.get("baseline") and r.get("loss_history")]
        if len(ar) < 2:
            continue
        ar.sort(key=lambda r: r["workers"])
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in ar:
            losses = r["loss_history"]
            ax.plot(range(1, len(losses) + 1), losses, linewidth=1.6,
                    label=f"AR {r['workers']}w (eff batch {_eff_batch(r)})")
        ax.set_title(f"Convergence vs workers · {model} · all-reduce  ({ts})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="effective batch = workers × batch")
        out.append(_save(fig, f"convergence_workers_{model}.png"))
        plt.close(fig)
    return out


def convergence_figures(results, ts):
    return plot_convergence(results, ts) + plot_convergence_workers(results, ts)


# ── Bars ─────────────────────────────────────────────────────────────────────

def _bar(ax, runs, value, fmt):
    vals = [r.get(value) or 0.0 for r in runs]
    errs = _errs(runs, value)
    pad = 0.04 * (max(vals) if vals else 1.0)
    ax.bar(range(len(runs)), vals, color="#1976d2", width=0.6,
           yerr=errs, capsize=4, ecolor="#37474f")
    for i, v in enumerate(vals):
        if v:
            off = (errs[i] if errs else 0) + pad
            ax.text(i, v + off, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([_label(r) for r in runs], rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.20)


def plot_execution_speed(results, ts):
    plt = _plt()
    out = []
    for model, runs in _by_model(_ok(results, "execution-speed")).items():
        runs = [r for r in runs if r.get("samples_per_sec")]
        if not runs:
            continue
        runs.sort(key=lambda r: (r["offline_epochs"], r["batch_size"]))
        algo = f"{STRAT[runs[0]['strategy']]} {_topo(runs[0])}"
        fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.5), 4.8))
        vals = [r["samples_per_sec"] for r in runs]
        errs = _errs(runs, "samples_per_sec")
        pad = 0.04 * (max(vals) if vals else 1.0)
        ax.bar(range(len(runs)), vals, color="#1976d2", width=0.6,
               yerr=errs, capsize=4, ecolor="#37474f")
        for i, v in enumerate(vals):
            off = (errs[i] if errs else 0) + pad
            ax.text(i, v + off, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([f"off={r['offline_epochs']}\nbatch={r['batch_size']}" for r in runs], fontsize=8)
        ax.set_ylabel("Samples / sec")
        ax.set_title(f"Execution speed · {model} · {algo}  ({ts})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.18)
        out.append(_save(fig, f"execution_speed_{model}.png"))
        plt.close(fig)
    return out


def plot_convergence_speed(results, ts):
    plt = _plt()
    out = []
    for model, runs in _by_model(_ok(results, "convergence-speed")).items():
        runs = [r for r in runs if r.get("loss_per_sec") is not None]
        if not runs:
            continue
        runs.sort(key=_label)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(9, len(runs) * 1.8), 5))
        _bar(a1, runs, "loss_per_sec", "{:.3g}")
        a1.set_ylabel("Loss reduction / sec")
        a1.set_title("Loss / sec")
        _bar(a2, runs, "accuracy_per_sec", "{:.3g}")
        a2.set_ylabel("Accuracy / sec")
        a2.set_title("Accuracy / sec")
        fig.suptitle(f"Convergence speed · {model} · @ {CONV_WORKERS}w  ({ts})",
                     fontsize=11, fontweight="bold")
        out.append(_save(fig, f"convergence_speed_{model}.png"))
        plt.close(fig)
    return out


def plot_scalability(results, ts):
    """All-reduce and parameter server in SEPARATE panels: PS spends extra server
    nodes, so overlaying them on one 'workers' axis would misrepresent cost."""
    plt = _plt()
    out = []
    for model, runs in _by_model(_ok(results, "scalability")).items():
        runs = [r for r in runs if r.get("samples_per_sec")]
        if not runs:
            continue
        panels = [("all_reduce", "#1976d2"), ("parameter_server", "#e64a19")]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        any_drawn = False
        for ax, (strat, color) in zip(axes, panels):
            sruns = sorted([r for r in runs if r["strategy"] == strat],
                           key=lambda r: r["workers"])
            if not sruns:
                ax.set_visible(False)
                continue
            any_drawn = True
            xs = list(range(len(sruns)))
            ys = [r["samples_per_sec"] for r in sruns]
            errs = _errs(sruns, "samples_per_sec")
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, color=color, linewidth=1.8)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{r['workers']}w\n({_nodes(r)} nodes)" for r in sruns], fontsize=9)
            ax.set_xlabel("Workers (total nodes)")
            ax.set_title(STRAT[strat] + (" (workers + servers)" if strat == "parameter_server" else ""),
                         fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.margins(x=0.15, y=0.15)
        if not any_drawn:
            plt.close(fig)
            continue
        axes[0].set_ylabel("Samples / sec")
        fig.suptitle(f"Scalability · {model}  ({ts})", fontsize=11, fontweight="bold")
        out.append(_save(fig, f"scalability_{model}.png"))
        plt.close(fig)
    return out


SUITE_PLOTTERS = {
    "convergence": convergence_figures,
    "execution-speed": plot_execution_speed,
    "convergence-speed": plot_convergence_speed,
    "scalability": plot_scalability,
}


def plot_suites(suites, all_results, ts):
    produced = []
    for suite in suites:
        produced.extend(SUITE_PLOTTERS[suite](all_results, ts))
    return produced
