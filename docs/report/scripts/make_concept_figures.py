#!/usr/bin/env python3
"""Conceptual (non-data) figures for the report chapter on O.N.O.: architecture,
component relations, algorithm loops and design-decision diagrams. Sober,
grayscale, legible at single-column width. Recovered and extended from the
original monography script (docs/report/oxidized-neural-orchestra-doc-monograph-ar-vs-ps.zip)."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "figures"
GREY, DARK_GREY, DARK, ACCENT = "0.90", "0.72", "0.25", "0.55"
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


def _label(ax, x, y, text, fs=6.3, style="normal", color=DARK):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, style=style, color=color)


def architecture():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.text(0.5, 1.06, "asignación de rol en runtime", ha="center",
            va="center", fontsize=6.4, color=ACCENT)
    _box(ax, (0.33, 0.80), 0.36, 0.14, "orchestrator\ncentral", fc=DARK_GREY, fs=7)
    roles = ["worker", "server", "worker"]
    xs = [0.05, 0.40, 0.75]
    w = 0.20
    for x, r in zip(xs, roles):
        fc = "0.72" if r == "server" else GREY
        _box(ax, (x, 0.36), w, 0.14, f"nodo\n({r})", fc=fc, fs=7)
        _arrow(ax, (0.50, 0.80), (x + w / 2, 0.50), style="-|>", ls=":", lw=0.7, color=ACCENT)
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + w, 0.43), (xs[i + 1], 0.43), style="<|-|>", lw=1.0)
    ax.text(0.5, 0.27, "comunicación entre nodos según la estrategia",
            ha="center", va="center", fontsize=6.3, color=ACCENT)
    ax.text(0.5, 0.16, "nodos idénticos; el rol se asigna en cada ejecución",
            ha="center", va="center", fontsize=6.5, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0.10, 1.16); ax.set_axis_off()
    fig.savefig(FIG / "architecture_ono.pdf", bbox_inches="tight")
    plt.close(fig)


def _ring(ax, n, cx, cy, r, box_w, box_h, label, fs=6.3, color=DARK, rad=0.15):
    pts = [(cx + r * np.cos(t), cy + r * np.sin(t))
           for t in np.linspace(0.5 * np.pi, 0.5 * np.pi + 2 * np.pi, n + 1)[:n]]
    boxes = [_box(ax, (x - box_w / 2, y - box_h / 2), box_w, box_h, label, fs=fs)
             for (x, y) in pts]
    for i in range(n):
        _link(ax, pts[i], pts[(i + 1) % n], boxes[i], boxes[(i + 1) % n],
              style="-|>", lw=0.8, rad=rad, color=color)
    return pts, boxes


def ps_vs_ar():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1))
    ax = axes[0]
    servers = []
    for x, name in ((0.04, "servidor 1"), (0.52, "servidor 2")):
        sb = _box(ax, (x, 0.74), 0.44, 0.19, f"{name}\n(shard)", fc="0.72", fs=6.3)
        servers.append((sb, (x + 0.22, 0.835)))
    for x in (0.01, 0.38, 0.75):
        wbox = _box(ax, (x, 0.02), 0.24, 0.16, "worker", fs=6.3)
        wc = (x + 0.12, 0.10)
        for sb, sc in servers:
            _link(ax, wc, sc, wbox, sb, style="<|-|>", lw=0.55, rad=0.10)
    ax.set_title("Parameter Server", fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax = axes[1]
    _ring(ax, 4, 0.5, 0.5, 0.30, 0.22, 0.14, "worker")
    ax.set_title("Ring All-Reduce", fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "ps_vs_ar_conceptual.pdf", bbox_inches="tight")
    plt.close(fig)


def ps_loop():
    fig, ax = plt.subplots(figsize=(3.7, 2.0))
    wbox = _box(ax, (0.02, 0.38), 0.34, 0.22, "worker\n(gradiente local)", fc=GREY, fs=6.4)
    sbox = _box(ax, (0.62, 0.38), 0.38, 0.22, "servidor\n(optimizador con estado)", fc=DARK_GREY, fs=5.7)
    _arrow(ax, (0.37, 0.64), (0.61, 0.64), lw=1.0)
    _arrow(ax, (0.61, 0.34), (0.37, 0.34), lw=1.0)
    _label(ax, 0.5, 0.83, "empuja gradiente", fs=6.4, color=ACCENT)
    _label(ax, 0.5, 0.17, "devuelve parámetros actualizados", fs=6.4, color=ACCENT)
    ax.text(0.5, 0.02, "el ciclo se repite en cada ronda de sincronización",
            ha="center", va="center", fontsize=6.2, style="italic")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.04, 0.98); ax.set_axis_off()
    fig.savefig(FIG / "ps_loop.pdf", bbox_inches="tight")
    plt.close(fig)


def ar_loop():
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.0))
    ax = axes[0]
    _ring(ax, 4, 0.5, 0.52, 0.32, 0.22, 0.14, "worker")
    ax.set_title("scatter", fontsize=7.5)
    ax.text(0.5, 0.03, "$N_W - 1$ iteraciones", ha="center", va="center",
            fontsize=6.2, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02); ax.set_axis_off()
    ax = axes[1]
    _ring(ax, 4, 0.5, 0.52, 0.32, 0.22, 0.14, "worker", color=ACCENT)
    ax.set_title("gather", fontsize=7.5)
    ax.text(0.5, 0.03, "$N_W - 1$ iteraciones", ha="center", va="center",
            fontsize=6.2, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02); ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / "ar_loop.pdf", bbox_inches="tight")
    plt.close(fig)


def strategy_switch_loop():
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    _ring(ax, 3, 0.20, 0.50, 0.14, 0.11, 0.09, "worker", fs=5.2, rad=0.10)
    _label(ax, 0.20, 0.92, "fase 1: All-Reduce", fs=6.6)
    _arrow(ax, (0.42, 0.50), (0.62, 0.50), style="-|>", lw=1.1)
    _label(ax, 0.52, 0.65, "ventana de N pérdidas", fs=6.0, color=ACCENT)
    _label(ax, 0.52, 0.58, "promedio < umbral", fs=6.0, color=ACCENT)
    _label(ax, 0.52, 0.38, "(dispara el switch)", fs=6.0, style="italic", color=ACCENT)
    sbox = _box(ax, (0.66, 0.68), 0.32, 0.14, "servidor", fc=DARK_GREY, fs=6.6)
    sc = (0.82, 0.68)
    for x, wc_x in ((0.70, 0.775), (0.87, 0.945)):
        wbox = _box(ax, (x, 0.10), 0.15, 0.12, "worker", fs=6.0)
        _link(ax, (wc_x, 0.22), sc, wbox, sbox, style="<|-|>", lw=0.7, rad=0.12)
    _label(ax, 0.82, 0.97, "fase 2: Parameter Server\nasincrónico", fs=6.6)
    ax.set_xlim(0.0, 1.06); ax.set_ylim(0.0, 1.05); ax.set_axis_off()
    fig.savefig(FIG / "strategy_switch_loop.pdf", bbox_inches="tight")
    plt.close(fig)


def connect_accept():
    fig, ax = plt.subplots(figsize=(2.8, 2.1))
    ax.plot([0.15, 0.15], [0.08, 0.90], color=DARK, lw=0.9)
    ax.plot([0.85, 0.85], [0.08, 0.90], color=DARK, lw=0.9)
    _box(ax, (0.01, 0.90), 0.28, 0.10, "nodo A", fc=GREY, fs=6.8)
    _box(ax, (0.71, 0.90), 0.28, 0.10, "nodo B", fc=GREY, fs=6.8)
    _arrow(ax, (0.15, 0.62), (0.85, 0.62), lw=1.0)
    _label(ax, 0.5, 0.68, "Connect(id, rol)", fs=6.4)
    _arrow(ax, (0.85, 0.34), (0.15, 0.34), lw=1.0)
    _label(ax, 0.5, 0.40, "Accept(id, rol)", fs=6.4)
    ax.text(0.5, 0.14, "cada extremo construye su handle\ntipado a partir del rol declarado",
            ha="center", va="center", fontsize=6.1, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0.05, 1.02); ax.set_axis_off()
    fig.savefig(FIG / "connect_accept.pdf", bbox_inches="tight")
    plt.close(fig)


def components_overview():
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    _box(ax, (0.02, 0.88), 0.44, 0.14, "interfaz de terminal\n(orchestui)", fc=GREY, fs=5.8)
    _box(ax, (0.54, 0.88), 0.44, 0.14, "bindings de Python\n(orchestra-py)", fc=GREY, fs=5.8)
    orch = _box(ax, (0.24, 0.66), 0.52, 0.15, "orchestrator", fc=DARK_GREY, fs=7)
    _arrow(ax, (0.24, 0.88), (0.42, 0.81), lw=0.9)
    _arrow(ax, (0.76, 0.88), (0.58, 0.81), lw=0.9)
    comms = _box(ax, (0.32, 0.44), 0.36, 0.15, "capa de\ncomunicación", fc=GREY, fs=6.4)
    _link(ax, (0.5, 0.66), (0.5, 0.59), orch, comms, lw=0.9)
    node = _box(ax, (0.28, 0.22), 0.44, 0.15, "nodo", fc=DARK_GREY, fs=7)
    _link(ax, (0.5, 0.44), (0.5, 0.37), comms, node, lw=0.9)
    ml = _box(ax, (0.00, 0.00), 0.46, 0.17, "motor de redes\nneuronales", fc=GREY, fs=6.2)
    strat = _box(ax, (0.52, 0.00), 0.46, 0.17, "estrategias de\nsincronización", fc=GREY, fs=6.2)
    _link(ax, (0.40, 0.22), (0.23, 0.17), node, ml, lw=0.8, rad=-0.10)
    _link(ax, (0.60, 0.22), (0.77, 0.17), node, strat, lw=0.8, rad=0.10)
    ax.text(0.84, 0.615, "control\n(JSON)", ha="center", va="center", fontsize=5.4, color=ACCENT)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.04); ax.set_axis_off()
    fig.savefig(FIG / "components_overview.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    architecture()
    ps_vs_ar()
    ps_loop()
    ar_loop()
    strategy_switch_loop()
    connect_accept()
    components_overview()
    print("wrote all figures to", FIG)
