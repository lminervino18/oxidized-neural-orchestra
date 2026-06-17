"""Suite plots. Each suite writes files under its own prefix so a partial run
only regenerates its own figures and never blanks another suite's plots.
"""

from collections import defaultdict
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

STRAT = {"parameter_server": "PS", "all_reduce": "AR", "strategy_switch": "SS"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _topo(r):
    return f"{r['workers']}w" if r["servers"] == 0 else f"{r['workers']}w/{r['servers']}s"


def _label(r):
    return f"{STRAT[r['strategy']]} {_topo(r)}"


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


def plot_convergence(results, ts):
    plt = _plt()
    out = []
    for model, runs in _by_model(_ok(results, "convergence")).items():
        curves = [r for r in runs if r.get("loss_history")]
        if curves:
            fig, ax = plt.subplots(figsize=(8, 5))
            for r in sorted(curves, key=_label):
                losses = r["loss_history"]
                ax.plot(range(1, len(losses) + 1), losses, label=_label(r), linewidth=1.5)
            ax.set_title(f"Convergence · {model}  ({ts})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            out.append(_save(fig, f"convergence_loss_{model}.png"))
            plt.close(fig)

        acc = [r for r in runs if r.get("accuracy") is not None]
        if acc:
            acc.sort(key=_label)
            fig, ax = plt.subplots(figsize=(max(5, len(acc) * 1.2), 4.5))
            vals = [r["accuracy"] for r in acc]
            ax.bar(range(len(acc)), vals, color="#4caf50", width=0.6)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(range(len(acc)))
            ax.set_xticklabels([_label(r) for r in acc], rotation=20, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Test accuracy")
            ax.set_title(f"Final accuracy · {model}  ({ts})", fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            out.append(_save(fig, f"convergence_accuracy_{model}.png"))
            plt.close(fig)
    return out


def _bar(ax, runs, value, fmt):
    vals = [r.get(value) or 0.0 for r in runs]
    ax.bar(range(len(runs)), vals, color="#1976d2", width=0.6)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([_label(r) for r in runs], rotation=20, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_execution_speed(results, ts):
    plt = _plt()
    out = []
    for model, runs in _by_model(_ok(results, "execution-speed")).items():
        runs = [r for r in runs if r.get("epochs_per_sec")]
        if not runs:
            continue
        runs.sort(key=lambda r: (r["offline_epochs"], r["batch_size"]))
        fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.4), 4.5))
        vals = [r["epochs_per_sec"] for r in runs]
        ax.bar(range(len(runs)), vals, color="#1976d2", width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([f"off={r['offline_epochs']}\nbatch={r['batch_size']}" for r in runs], fontsize=8)
        ax.set_ylabel("Epochs / sec")
        ax.set_title(f"Execution speed · {model}  ({ts})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
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
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(8, len(runs) * 1.6), 4.5))
        _bar(a1, runs, "loss_per_sec", "{:.3g}")
        a1.set_ylabel("Loss reduction / sec")
        a1.set_title("Loss / sec")
        _bar(a2, runs, "accuracy_per_sec", "{:.3g}")
        a2.set_ylabel("Accuracy / sec")
        a2.set_title("Accuracy / sec")
        fig.suptitle(f"Convergence speed · {model}  ({ts})", fontsize=11, fontweight="bold")
        out.append(_save(fig, f"convergence_speed_{model}.png"))
        plt.close(fig)
    return out


def plot_scalability(results, ts):
    plt = _plt()
    out = []
    for model, runs in _by_model(_ok(results, "scalability")).items():
        runs = [r for r in runs if r.get("epochs_per_sec")]
        if not runs:
            continue
        by_strat = defaultdict(list)
        for r in runs:
            by_strat[r["strategy"]].append(r)
        fig, ax = plt.subplots(figsize=(7, 5))
        for strat, sruns in sorted(by_strat.items()):
            sruns.sort(key=lambda r: r["workers"])
            ax.plot([r["workers"] for r in sruns], [r["epochs_per_sec"] for r in sruns],
                    marker="o", label=STRAT[strat])
        ax.set_xlabel("Workers")
        ax.set_ylabel("Epochs / sec")
        ax.set_title(f"Scalability · {model}  ({ts})", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        out.append(_save(fig, f"scalability_{model}.png"))
        plt.close(fig)
    return out


SUITE_PLOTTERS = {
    "convergence": plot_convergence,
    "execution-speed": plot_execution_speed,
    "convergence-speed": plot_convergence_speed,
    "scalability": plot_scalability,
}


def plot_suites(suites, all_results, ts):
    produced = []
    for suite in suites:
        produced.extend(SUITE_PLOTTERS[suite](all_results, ts))
    return produced
