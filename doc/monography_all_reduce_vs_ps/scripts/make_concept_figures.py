#!/usr/bin/env python3
"""Conceptual (non-data) figures for the monography: O.N.O. architecture and the
PS-vs-AR contrast. Sober, grayscale, legible at single-column width."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "figures"
GREY, DARK, ACCENT = "0.90", "0.25", "0.55"
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def _box(ax, xy, w, h, text, fc=GREY, ec=DARK, fs=8):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=0.9, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def _arrow(ax, a, b, style="-|>", ls="-", lw=0.9, color=DARK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def _link(ax, ca, cb, pa, pb, style="-|>", ls="-", lw=0.9, color=DARK, rad=0.0):
    """Arrow between two box centers, clipped to the boxes' boundaries."""
    ax.add_patch(FancyArrowPatch(ca, cb, patchA=pa, patchB=pb, arrowstyle=style,
                                 mutation_scale=8, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def architecture():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    # Top label, with a clear gap above the orchestrator box (no overlap).
    ax.text(0.5, 1.06, "asignacion de rol en runtime", ha="center",
            va="center", fontsize=6.4, color=ACCENT)
    _box(ax, (0.33, 0.80), 0.34, 0.14, "orquestador\ncentral", fc="0.82")
    # Three well-separated, identical nodes (wide gaps so the links read clearly).
    roles = ["worker", "server", "worker"]
    xs = [0.05, 0.40, 0.75]
    w = 0.20
    for x, r in zip(xs, roles):
        fc = "0.72" if r == "server" else GREY
        _box(ax, (x, 0.36), w, 0.14, f"nodo\n({r})", fc=fc, fs=7)
        _arrow(ax, (0.50, 0.80), (x + w / 2, 0.50), style="-|>", ls=":", lw=0.7, color=ACCENT)
    # Inter-node communication: one clear bidirectional link in each (now wide) gap.
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + w, 0.43), (xs[i + 1], 0.43), style="<|-|>", lw=1.0)
    ax.text(0.5, 0.27, "comunicacion entre nodos segun la estrategia",
            ha="center", va="center", fontsize=6.3, color=ACCENT)
    ax.text(0.5, 0.16, "nodos identicos; el rol se asigna en cada ejecucion",
            ha="center", va="center", fontsize=6.5, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0.10, 1.16); ax.set_axis_off()
    fig.savefig(FIG / "architecture_ono.pdf", bbox_inches="tight")
    plt.close(fig)


def ps_vs_ar():
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1))
    # ── Parameter Server (parameters sharded across TWO servers) ──
    ax = axes[0]
    servers = []
    for x, name in ((0.04, "servidor 1"), (0.52, "servidor 2")):
        sb = _box(ax, (x, 0.74), 0.44, 0.19, f"{name}\n(shard)", fc="0.72", fs=6.3)
        servers.append((sb, (x + 0.22, 0.835)))
    # Three workers along the bottom; each exchanges grads/params with BOTH shards.
    for x in (0.01, 0.38, 0.75):
        wbox = _box(ax, (x, 0.02), 0.24, 0.16, "worker", fs=6.3)
        wc = (x + 0.12, 0.10)
        for sb, sc in servers:
            _link(ax, wc, sc, wbox, sb, style="<|-|>", lw=0.55, rad=0.10)  # push grad / pull params
    ax.set_title("Parameter Server", fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    # ── Ring All-Reduce ──
    ax = axes[1]
    cx, cy, r = 0.5, 0.5, 0.30
    pts = [(cx + r * np.cos(t), cy + r * np.sin(t))
           for t in np.linspace(0.5 * np.pi, 0.5 * np.pi + 2 * np.pi, 5)[:4]]
    boxes = [_box(ax, (x - 0.11, y - 0.07), 0.22, 0.14, "worker", fs=6.3) for (x, y) in pts]
    for i in range(4):
        _link(ax, pts[i], pts[(i + 1) % 4], boxes[i], boxes[(i + 1) % 4],
              style="-|>", lw=0.8, rad=0.15)
    ax.set_title("Ring All-Reduce", fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "ps_vs_ar_conceptual.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    architecture()
    ps_vs_ar()
    print("wrote architecture_ono.pdf, ps_vs_ar_conceptual.pdf")
