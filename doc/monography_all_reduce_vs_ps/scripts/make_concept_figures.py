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
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.9, edgecolor=ec, facecolor=fc))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, a, b, style="-|>", ls="-", lw=0.9, color=DARK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def architecture():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    # Top label, with a clear gap above the orchestrator box (no overlap).
    ax.text(0.5, 1.05, "configuracion de runtime (rol por nodo)", ha="center",
            va="center", fontsize=6.4, color=ACCENT)
    _box(ax, (0.34, 0.78), 0.32, 0.13, "orchestrator\n(headless)", fc="0.82")
    roles = ["worker", "worker", "server", "worker"]
    xs = [0.04, 0.28, 0.52, 0.76]
    for x, r in zip(xs, roles):
        fc = "0.72" if r == "server" else GREY
        _box(ax, (x, 0.32), 0.20, 0.13, f"node\n({r})", fc=fc, fs=7)
        _arrow(ax, (0.50, 0.78), (x + 0.10, 0.45), style="-|>", ls=":", lw=0.7, color=ACCENT)
    # ring among workers (AR view) + push/pull to server (PS view) shown lightly
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + 0.20, 0.38), (xs[i + 1], 0.38), style="<|-|>", lw=0.7)
    ax.text(0.5, 0.14, "roles asignados en runtime, mismo binario en todos los nodos",
            ha="center", va="center", fontsize=6.5, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0.06, 1.12); ax.set_axis_off()
    fig.savefig(FIG / "architecture_ono.pdf", bbox_inches="tight")
    plt.close(fig)


def ps_vs_ar():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    # ── Parameter Server ──
    ax = axes[0]
    _box(ax, (0.30, 0.74), 0.40, 0.18, "server(s)\n(pesos shardeados)", fc="0.72", fs=6.5)
    for x in (0.02, 0.38, 0.74):
        _box(ax, (x, 0.10), 0.24, 0.16, "worker", fs=6.5)
        _arrow(ax, (x + 0.12, 0.26), (0.5, 0.74), style="-|>", lw=0.7)          # push grad
        _arrow(ax, (0.5, 0.74), (x + 0.12, 0.26), style="-|>", ls=":", lw=0.7, color=ACCENT)  # pull params
    ax.set_title("Parameter Server", fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    # ── Ring All-Reduce ──
    ax = axes[1]
    import numpy as np
    cx, cy, r = 0.5, 0.5, 0.32
    pts = [(cx + r * np.cos(t), cy + r * np.sin(t))
           for t in np.linspace(0.5 * np.pi, 0.5 * np.pi + 2 * np.pi, 5)[:4]]
    for (x, y) in pts:
        _box(ax, (x - 0.11, y - 0.07), 0.22, 0.14, "worker", fs=6.5)
    for i in range(4):
        _arrow(ax, pts[i], pts[(i + 1) % 4], style="-|>", lw=0.8)
    ax.set_title("Ring All-Reduce", fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "ps_vs_ar_conceptual.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    architecture()
    ps_vs_ar()
    print("wrote architecture_ono.pdf, ps_vs_ar_conceptual.pdf")
