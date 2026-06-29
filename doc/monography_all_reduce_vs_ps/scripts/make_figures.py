#!/usr/bin/env python3
"""Data figures for the paper, built from results/processed/*.csv.

Sober, grayscale-friendly, legible at single-column width. Each figure degrades
gracefully if its data is absent (e.g. communication pressure needs MLP runs).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "results" / "processed"
FIG = ROOT / "figures"

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "lines.linewidth": 1.3})

# Distinct, print-safe styles per strategy (no loud colors).
STYLE = {
    "all_reduce":      dict(color="#1f4e79", marker="o", label="All-Reduce"),
    "parameter_server":dict(color="#a6611a", marker="s", label="Parameter Server"),
    "strategy_switch": dict(color="#4d7c4d", marker="^", label="Strategy Switch"),
}
PRETTY = {k: v["label"] for k, v in STYLE.items()}


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


def fig_loss_vs_epoch():
    """Per-epoch train loss by strategy, on the sanity runs at 3 nodes."""
    rows = _rows("results.csv")
    curves = defaultdict(list)  # strategy -> [(epoch, loss)]
    for r in rows:
        if r.get("dataset") != "mnist" or r.get("model") != "nielsen":
            continue
        # sanity runs (Exp A) at 3 nodes. The PyTorch baseline uses a different
        # loss formulation (CE on logits, ~2.3 scale) than the engine (~0.23),
        # so its loss values are not comparable — keep it out of this axis.
        if not r.get("run_id", "").startswith("A_"):
            continue
        if r.get("strategy") == "baseline":
            continue
        e, tl = _f(r.get("epoch")), _f(r.get("train_loss"))
        if e is None or tl is None:
            continue
        curves[r["strategy"]].append((e, tl))
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for strat, pts in curves.items():
        pts.sort()
        xs, ys = zip(*pts)
        st = STYLE.get(strat, {})
        ax.plot(xs, ys, color=st.get("color", "0.4"), ls=st.get("ls", "-"),
                label=st.get("label", strat), marker="", markersize=3)
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida de entrenamiento")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "mnist_loss_vs_epoch.pdf"); plt.close(fig)


def _scal():
    rows = _rows("scalability_summary.csv")
    by = defaultdict(list)
    for r in rows:
        if r.get("dataset") == "mnist":
            by[r["strategy"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: _f(r["nodes"]) or 0)
    return by


def fig_throughput_vs_nodes():
    by = _scal()
    if not by:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for st, rs in by.items():
        s = STYLE.get(st, {})
        xs = [_f(r["nodes"]) for r in rs]
        ys = [_f(r["samples_per_sec"]) for r in rs]
        ax.plot(xs, ys, color=s.get("color", "0.4"), marker=s.get("marker") or "o",
                label=s.get("label", st))
    ax.set_xlabel("Nodos"); ax.set_ylabel("Throughput (samples/s)")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "throughput_vs_nodes.pdf"); plt.close(fig)


def fig_speedup_vs_nodes():
    by = _scal()
    if not by:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    allx = set()
    for st, rs in by.items():
        s = STYLE.get(st, {})
        xs = [_f(r["nodes"]) for r in rs]
        ys = [_f(r["speedup_vs_min"]) for r in rs]
        allx.update(xs)
        ax.plot(xs, ys, color=s.get("color", "0.4"), marker=s.get("marker") or "o",
                label=s.get("label", st))
    if allx:
        lo, hi = min(allx), max(allx)
        ax.plot([lo, hi], [lo / lo, hi / lo], color="0.6", ls=":", label="ideal")
    ax.set_xlabel("Nodos"); ax.set_ylabel("Speedup (vs. mín.)")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "speedup_vs_nodes.pdf"); plt.close(fig)


def fig_communication_pressure():
    """Throughput vs model size, AR vs PS — only if MLP runs exist."""
    rows = _rows("summary_by_strategy.csv")
    mlp = [r for r in rows if (r.get("model") or "").startswith("mlp")
           and _f(r.get("samples_per_sec"))]
    if not mlp:
        return
    order = {"mlp_small": 0, "mlp_medium": 1, "mlp_large": 2}
    by = defaultdict(list)
    for r in mlp:
        by[r["strategy"]].append(r)
    cats = ["small", "medium", "large"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for st, rs in by.items():
        rs.sort(key=lambda r: order.get(r["model"], 99))
        s = STYLE.get(st, {})
        xs = [r["model"].replace("mlp_", "") for r in rs]
        ys = [_f(r["samples_per_sec"]) for r in rs]
        ax.plot(xs, ys, color=s.get("color", "0.4"), marker=s.get("marker") or "o",
                label=s.get("label", st))
    # Mark a strategy that failed at the largest model (e.g. AR ring timeout).
    for st, rs in by.items():
        done = {r["model"].replace("mlp_", "") for r in rs}
        for c in cats:
            if c not in done and any(c in {x["model"].replace("mlp_", "")
                                          for x in by[o]} for o in by if o != st):
                ax.scatter([c], [min(_f(r["samples_per_sec"]) for r in rs)],
                           marker="x", color="firebrick", s=40, zorder=5)
                ax.annotate("no completo", ("large", min(_f(r["samples_per_sec"]) for r in rs)),
                            fontsize=6, color="firebrick", ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(-0.3, 2.3)
    ax.set_xlabel("Tamaño de modelo (MLP)"); ax.set_ylabel("Throughput (samples/s, log)")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "communication_pressure.pdf"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    for fn in (fig_loss_vs_epoch, fig_throughput_vs_nodes,
               fig_speedup_vs_nodes, fig_communication_pressure):
        try:
            fn()
            print("ok:", fn.__name__)
        except Exception as e:  # never let one figure break the build
            print("skip:", fn.__name__, "->", e)


if __name__ == "__main__":
    main()
