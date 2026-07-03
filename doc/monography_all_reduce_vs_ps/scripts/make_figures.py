#!/usr/bin/env python3
"""Data figures for the IEEE monography (All-Reduce vs. Parameter Server).

Built from results/processed/*.csv. Sober, grayscale, legible at single-column
width (~3.4in). Each figure is isolated in try/except so one failure never
blocks the others. Axes are labelled in Spanish; figure captions (in the paper)
carry the interpretation, so nothing is annotated on the plots themselves.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "results" / "processed"
FIG = ROOT / "figures"

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
})

# Grayscale, print-safe styles. AR = solid + circles; PS = dashed + squares.
AR = dict(color="0.0", ls="-", marker="o", label="All-Reduce")
PS = dict(color="0.45", ls="--", marker="s", label="Parameter Server")


def _rows(name):
    p = PROC / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _curve(rows, run_id):
    """Sorted [(epoch, train_loss)] for a single run_id from results.csv."""
    pts = []
    for r in rows:
        if r.get("run_id") != run_id:
            continue
        e, tl = _f(r.get("epoch")), _f(r.get("train_loss"))
        if e is not None and tl is not None:
            pts.append((e, tl))
    pts.sort()
    return pts


# ── Figure 1: convergence (train loss vs. epoch), 2 panels ──────────────────
def fig_convergence_loss():
    rows = _rows("results.csv")
    panels = [
        ("FashionMNIST", "Bg_ar_fashion_n3", "Bg_ps_fashion_n3"),
        ("MNIST", "Bg_ar_mnist_n3", "Bg_ps_mnist_n3"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1))
    drew = False
    for ax, (title, ar_id, ps_id) in zip(axes, panels):
        for run_id, st in ((ar_id, AR), (ps_id, PS)):
            pts = _curve(rows, run_id)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=st["color"], ls=st["ls"], label=st["label"],
                    marker="", markersize=2.5)
            drew = True
        # Log-y spreads the decade the loss traverses; keeps late epochs readable.
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Época")
        ax.tick_params(labelsize=6.5)
        # Curves decay left→right, so the upper-right corner is free of data.
        ax.legend(fontsize=6, frameon=False, loc="upper right")
    axes[0].set_ylabel("Loss de entrenamiento")
    if not drew:
        plt.close(fig)
        raise RuntimeError("no convergence data for Bg_*_n3 runs")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "convergence_loss.pdf")
    plt.close(fig)


# ── Figure 2: throughput vs. number of workers ──────────────────────────────
def fig_throughput_vs_workers():
    rows = _rows("summary_by_strategy.csv")
    by = {"all_reduce": [], "parameter_server": []}
    for r in rows:
        if not (r.get("label") or "").startswith("Bp_"):
            continue
        strat = r.get("strategy")
        if strat in by:
            w, sps = _f(r.get("workers")), _f(r.get("samples_per_sec"))
            if w is not None and sps is not None:
                by[strat].append((w, sps))
    if not any(by.values()):
        raise RuntimeError("no Bp_* throughput rows")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for strat, st in (("all_reduce", AR), ("parameter_server", PS)):
        pts = sorted(by[strat])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"],
                markersize=4, label=st["label"])
    ax.set_xticks([3, 5])
    ax.set_xlabel("Cantidad de workers")
    ax.set_ylabel("Throughput (muestras/s)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "throughput_vs_workers.pdf")
    plt.close(fig)


# ── Figure 3: speedup vs. number of workers ─────────────────────────────────
def fig_speedup_vs_workers():
    rows = _rows("scalability_summary.csv")
    by = {"all_reduce": [], "parameter_server": []}
    for r in rows:
        strat = r.get("strategy")
        if strat in by and r.get("dataset") == "fashion_mnist":
            w, sp = _f(r.get("workers")), _f(r.get("speedup_vs_min"))
            if w is not None and sp is not None:
                by[strat].append((w, sp))
    if not any(by.values()):
        raise RuntimeError("no scalability rows")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    allx = set()
    for strat, st in (("all_reduce", AR), ("parameter_server", PS)):
        pts = sorted(by[strat])
        if not pts:
            continue
        xs, ys = zip(*pts)
        allx.update(xs)
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"],
                markersize=4, label=st["label"])
    if allx:
        ax.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(sorted(allx) or [3, 5])
    ax.set_xlabel("Cantidad de workers")
    ax.set_ylabel("Speedup (vs. 3 workers)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "speedup_vs_workers.pdf")
    plt.close(fig)


# ── Figure 4: communication pressure (throughput vs. model size) ────────────
def fig_communication_pressure():
    rows = _rows("summary_by_strategy.csv")
    # Parameter counts per dense model (fixed by architecture, not in the CSV).
    params = {"mlp_small": 101770, "mlp_medium": 669706, "mlp_large": 1863690}
    by = {"all_reduce": [], "parameter_server": []}
    for r in rows:
        if not (r.get("label") or "").startswith("D_"):
            continue
        strat, model = r.get("strategy"), r.get("model")
        if strat in by and model in params:
            sps = _f(r.get("samples_per_sec"))
            if sps is not None:
                by[strat].append((params[model], sps))
    if not any(by.values()):
        raise RuntimeError("no D_* dense-model rows")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for strat, st in (("all_reduce", AR), ("parameter_server", PS)):
        pts = sorted(by[strat])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"],
                markersize=4, label=st["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Widen x a touch beyond the extreme points so the trend past the crossover
    # region reads clearly (the old figure clipped right at the largest model).
    ax.set_xlim(8e4, 2.6e6)
    ax.set_xlabel("Cantidad de parámetros del modelo (log)")
    ax.set_ylabel("Throughput (muestras/s, log)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "communication_pressure.pdf")
    plt.close(fig)


# ── Figure 5: analytical communication volume per SGD step ───────────────────
def fig_comm_volume_analytical():
    P = 289630          # nielsen model parameters
    BYTES = 4           # float32
    S = 2               # PS servers
    ws = list(range(2, 9))
    # Ring All-Reduce: bandwidth-optimal, per-node traffic ~ constant in W.
    ar = [2.0 * (w - 1) / w * P * BYTES / 1e6 for w in ws]
    # Parameter Server: per-server ingress grows linearly with W (S servers).
    ps = [2.0 * P * BYTES * w / S / 1e6 for w in ws]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(ws, ar, color=AR["color"], ls=AR["ls"], marker=AR["marker"],
            markersize=4, label="All-Reduce (por nodo)")
    ax.plot(ws, ps, color=PS["color"], ls=PS["ls"], marker=PS["marker"],
            markersize=4, label="Parameter Server (por servidor, S=2)")
    ax.set_xticks(ws)
    ax.set_xlabel("Cantidad de workers (W)")
    ax.set_ylabel("Bytes comunicados por paso (MB)")
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "comm_volume_analytical.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    for fn in (fig_convergence_loss, fig_throughput_vs_workers,
               fig_speedup_vs_workers, fig_communication_pressure,
               fig_comm_volume_analytical):
        try:
            fn()
            print("ok:", fn.__name__)
        except Exception as e:  # never let one figure break the build
            print("skip:", fn.__name__, "->", e)


if __name__ == "__main__":
    main()
