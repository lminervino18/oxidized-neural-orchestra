"""Export the speaker notes embedded in index.html to NOTAS_EXPOSITORES.md.

Usage (from presentation/):
    python3 tools/export_notes.py

index.html is the single source of truth: every <section class="slide">
carries data-speaker / data-time and an <aside class="notes">. The script
also checks that the main slides plus the demo add up to 45 minutes.
"""

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "NOTAS_EXPOSITORES.md"
DEMO_SECONDS = 10 * 60
TARGET_SECONDS = 45 * 60

SECTION = re.compile(r'<section class="slide[^"]*"(?P<attrs>[^>]*)>(?P<body>.*?)</section>', re.S)
ATTR = re.compile(r'data-([a-z]+)="([^"]*)"')
FIELD = {
    "n-idea": "Idea principal",
    "n-oral": "Desarrollo oral",
    "n-trans": "Transición",
    "n-src": "Respaldo",
}


def to_markdown(fragment):
    text = re.sub(r"</p>\s*<p>", "\n\n", fragment)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text)
    text = re.sub(r"<(strong|b)>(.*?)</\1>", r"**\2**", text)
    text = re.sub(r"<sub>(.*?)</sub>", r"_\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def seconds(t):
    minutes, secs = t.split(":")
    return int(minutes) * 60 + int(secs)


def fmt(total):
    return f"{total // 60:02d}:{total % 60:02d}"


def main():
    source = (ROOT / "index.html").read_text(encoding="utf-8")
    lines = ["# Notas de los expositores", "", "Generado desde `index.html` con `python3 tools/export_notes.py`. No editar a mano.", ""]
    total = 0
    blocks = {}
    main_count = 0
    for match in SECTION.finditer(source):
        attrs = dict(ATTR.findall(match.group("attrs")))
        appendix = "data-appendix" in match.group("attrs")
        if not appendix:
            main_count += 1
            label = f"{main_count:02d}"
            total += seconds(attrs.get("time", "0:00"))
            speaker = attrs.get("speaker", "—")
            blocks[speaker] = blocks.get(speaker, 0) + seconds(attrs.get("time", "0:00"))
        else:
            label = "Apéndice"
        lines.append(f"## {label} · {html.unescape(attrs.get('title', ''))}")
        lines.append("")
        lines.append(f"- Expositor sugerido: {attrs.get('speaker', '—')}")
        lines.append(f"- Tiempo estimado: {attrs.get('time', '—')}" + (f" (acumulado al terminar: {fmt(total)})" if not appendix else ""))
        notes = re.search(r'<aside class="notes">(.*?)</aside>', match.group("body"), re.S)
        if notes:
            for cls, title in FIELD.items():
                field = re.search(rf'<(p|div) class="{cls}">(.*?)</\1>', notes.group(1), re.S)
                if field:
                    lines.append(f"- {title}: {to_markdown(field.group(2))}" if cls != "n-oral" else f"- {title}:\n\n{to_markdown(field.group(2))}\n")
        lines.append("")

    grand = total + DEMO_SECONDS
    summary = [
        "## Resumen de tiempos",
        "",
        *[f"- {speaker}: {fmt(secs)}" for speaker, secs in blocks.items()],
        f"- Diapositivas principales: {fmt(total)}",
        f"- Demo: {fmt(DEMO_SECONDS)}",
        f"- Total: {fmt(grand)} (objetivo {fmt(TARGET_SECONDS)})",
        "",
    ]
    OUT.write_text("\n".join(lines[:4] + summary + lines[4:]), encoding="utf-8")
    print("\n".join(summary))
    if grand != TARGET_SECONDS:
        raise SystemExit(f"los tiempos suman {fmt(grand)}, no {fmt(TARGET_SECONDS)}")


if __name__ == "__main__":
    main()
