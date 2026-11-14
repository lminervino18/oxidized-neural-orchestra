"""Convert real `tmux capture-pane -e` dumps of orchestui into data/captures.js.

Usage (from presentation/):
    python3 tools/ansi_to_capture.py

Reads captures/raw/*.ans (single screenshots) and captures/rec/*.ans plus
captures/rec/times.txt (a frame-by-frame recording) and writes
data/captures.js, which defines `window.ONO_TUI` for js/tui.js.
Rows are deduplicated across every frame, so long recordings stay small.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "captures" / "raw"
REC = ROOT / "captures" / "rec"
OUT = ROOT / "data" / "captures.js"

COLS = 190
DEFAULT_FG = "#c8d0c8"
DEFAULT_BG = None
BASIC = ["#000000", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5"]
BRIGHT = ["#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#ffffff"]
SGR = re.compile(r"\x1b\[([0-9;]*)m")

SHOTS = {
    "menu": ("01_menu.ans", "Menú principal"),
    "config_model": ("02_config_model.ans", "Configuración · paso 1 (model.json)"),
    "config_example": ("03_config_example.ans", "Ejemplo inline de model.json ([?])"),
    "config_training": ("04_config_training.ans", "Configuración · paso 2 (training.json)"),
    "ar_dash_finished": ("05_ar_dash_finished.ans", "All-Reduce · dashboard al finalizar"),
    "ar_topo_finished": ("06_ar_topology_finished.ans", "All-Reduce · topología al finalizar"),
    "ar_dash_mid": ("10_ar_dash_mid.ans", "All-Reduce · dashboard en curso"),
    "ar_topo_mid": ("11_ar_topo_mid.ans", "All-Reduce · topología en curso"),
    "ps_dash_mid": ("20_ps_dash_mid.ans", "Parameter Server · dashboard en curso"),
    "ps_topo_mid": ("21_ps_topo_mid.ans", "Parameter Server · topología en curso"),
    "ss_topo_before": ("30_ss_topo_before.ans", "Strategy-Switch · antes del cambio"),
    "ss_topo_after": ("31_ss_topo_after.ans", "Strategy-Switch · durante el cambio"),
    "ss_dash_after": ("32_ss_dash_after.ans", "Strategy-Switch · dashboard tras el cambio"),
    "ss_dash_finished": ("33_ss_dash_finished.ans", "Strategy-Switch · dashboard al finalizar"),
    "ss_topo_finished": ("34_ss_topo_finished.ans", "Strategy-Switch · topología al finalizar"),
}


class Palette:
    def __init__(self):
        self.colors = []
        self.index = {}

    def id(self, color):
        if color is None:
            return -1
        if color not in self.index:
            self.index[color] = len(self.colors)
            self.colors.append(color)
        return self.index[color]


def apply_sgr(params, state):
    codes = [int(c) if c else 0 for c in params.split(";")] if params else [0]
    i = 0
    while i < len(codes):
        c = codes[i]
        if c == 0:
            state.update(fg=DEFAULT_FG, bg=DEFAULT_BG, bold=False, dim=False)
        elif c == 1:
            state["bold"] = True
        elif c == 2:
            state["dim"] = True
        elif c == 22:
            state.update(bold=False, dim=False)
        elif c in (38, 48) and i + 1 < len(codes):
            key = "fg" if c == 38 else "bg"
            if codes[i + 1] == 2 and i + 4 < len(codes):
                r, g, b = codes[i + 2:i + 5]
                state[key] = f"#{r:02x}{g:02x}{b:02x}"
                i += 4
            elif codes[i + 1] == 5 and i + 2 < len(codes):
                n = codes[i + 2]
                state[key] = (BASIC + BRIGHT)[n] if n < 16 else DEFAULT_FG
                i += 2
        elif c == 39:
            state["fg"] = DEFAULT_FG
        elif c == 49:
            state["bg"] = DEFAULT_BG
        elif 30 <= c <= 37:
            state["fg"] = BASIC[c - 30]
        elif 90 <= c <= 97:
            state["fg"] = BRIGHT[c - 90]
        elif 40 <= c <= 47:
            state["bg"] = BASIC[c - 40]
        i += 1


def parse_line(line, palette):
    state = {"fg": DEFAULT_FG, "bg": DEFAULT_BG, "bold": False, "dim": False}
    runs = []
    width = 0
    pos = 0
    for match in list(SGR.finditer(line)) + [None]:
        end = match.start() if match else len(line)
        text = line[pos:end]
        if text:
            flags = (1 if state["bold"] else 0) | (2 if state["dim"] else 0)
            key = (palette.id(state["fg"]), palette.id(state["bg"]), flags)
            if runs and tuple(runs[-1][1:]) == key:
                runs[-1][0] += text
            else:
                runs.append([text, *key])
            width += len(text)
        if match:
            apply_sgr(match.group(1), state)
            pos = match.end()
    if width < COLS:
        runs.append([" " * (COLS - width), palette.id(DEFAULT_FG), -1, 0])
    return runs


class RowTable:
    def __init__(self):
        self.rows = []
        self.index = {}

    def id(self, runs):
        key = json.dumps(runs, ensure_ascii=False)
        if key not in self.index:
            self.index[key] = len(self.rows)
            self.rows.append(runs)
        return self.index[key]


def read_frame(path, palette, table):
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [table.id(parse_line(line, palette)) for line in lines]


def main():
    palette = Palette()
    table = RowTable()

    shots = {}
    for name, (filename, title) in SHOTS.items():
        shots[name] = {"title": title, "rows": read_frame(RAW / filename, palette, table)}

    times = {}
    for line in (REC / "times.txt").read_text().split("\n"):
        if line.strip():
            frame, seconds = line.split()
            times[frame] = round(float(seconds), 2)
    frames = sorted(p for p in REC.glob("*.ans"))
    recording = {
        "title": "Strategy-Switch · vista de topología, 1 captura por segundo",
        "frames": [read_frame(p, palette, table) for p in frames],
        "times": [times.get(p.stem, 0.0) for p in frames],
    }

    data = {
        "cols": COLS,
        "captured": "28/10/2026",
        "palette": palette.colors,
        "rows": table.rows,
        "shots": shots,
        "recording": recording,
    }
    OUT.parent.mkdir(exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(f"window.ONO_TUI = {payload};\n", encoding="utf-8")
    print(f"{OUT}: {len(table.rows)} unique rows, {len(frames)} frames, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
