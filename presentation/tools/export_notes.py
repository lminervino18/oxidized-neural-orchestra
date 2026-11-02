"""Export the speaker notes embedded in index.html to GUION.md.

Usage (from presentation/):
    python3 tools/export_notes.py

index.html is the single source of truth: every <section class="slide">
carries data-speaker / data-time (and data-after for the end of the live
demo) and an <aside class="notes">. Slides without time are listed as backup.
The script also checks that everything adds up to 45 minutes.
"""

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "GUION.md"
TARGET_SECONDS = 45 * 60

SECTION = re.compile(r'<section class="slide[^"]*"(?P<attrs>[^>]*)>(?P<body>.*?)</section>', re.S)
ATTR = re.compile(r'data-([a-z]+)="([^"]*)"')
KEYWORD = re.compile(r'<span class="kw[^"]*"[^>]*>(.*?)</span>', re.S)


def to_markdown(fragment):
    text = re.sub(r"\s*<li>", "\n- ", fragment)
    text = re.sub(r"</p>\s*<p>", "\n\n", text)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text)
    text = re.sub(r"<(strong|b)>(.*?)</\1>", r"**\2**", text)
    text = re.sub(r"<em>(.*?)</em>", r"*\1*", text)
    text = re.sub(r"<sub>(.*?)</sub>", r"_\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def field(notes, cls):
    match = re.search(rf'<(p|div|ul) class="{cls}">(.*?)</\1>', notes, re.S)
    return to_markdown(match.group(2)) if match else ""


def plain(fragment):
    fragment = re.sub(r"<sub>(.*?)</sub>", r"_\1", fragment)
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def seconds(t):
    minutes, secs = (t or "0:00").split(":")
    return int(minutes) * 60 + int(secs)


def fmt(total):
    return f"{total // 60:02d}:{total % 60:02d}"


def parse(source):
    slides = []
    for match in SECTION.finditer(source):
        raw = match.group("attrs")
        attrs = {k: html.unescape(v) for k, v in ATTR.findall(raw)}
        body = match.group("body")
        notes = re.search(r'<aside class="notes">(.*?)</aside>', body, re.S)
        notes = notes.group(1) if notes else ""
        keywords = [plain(k) for k in KEYWORD.findall(body)]
        slides.append({
            "appendix": "data-appendix" in raw,
            "title": attrs.get("title", ""),
            "speaker": attrs.get("speaker", "—"),
            "time": seconds(attrs.get("time")),
            "after": seconds(attrs.get("after")),
            "keywords": " ".join(f"«{k}»" for k in keywords) if keywords else attrs.get("keywords", "—"),
            "idea": field(notes, "n-idea"),
            "oral": field(notes, "n-oral"),
            "trans": field(notes, "n-trans"),
            "src": field(notes, "n-src"),
        })
    return slides


def label(slides):
    main_n = appendix_n = 0
    for slide in slides:
        if slide["appendix"]:
            appendix_n += 1
            slide["label"] = f"A{appendix_n}"
        else:
            main_n += 1
            slide["label"] = f"{main_n:02d}"


def speaker_blocks(talk):
    blocks = []
    for slide in talk:
        if blocks and blocks[-1]["speaker"] == slide["speaker"]:
            blocks[-1]["last"] = slide["label"]
            blocks[-1]["time"] += slide["time"]
            blocks[-1]["after"] += slide["after"]
        else:
            blocks.append({"speaker": slide["speaker"], "first": slide["label"], "last": slide["label"], "time": slide["time"], "after": slide["after"]})
    return blocks


def main():
    slides = parse((ROOT / "index.html").read_text(encoding="utf-8"))
    label(slides)
    talk = [s for s in slides if s["time"] or s["after"]]
    backup = [s for s in slides if not (s["time"] or s["after"])]

    lines = [
        "# Guion de la defensa",
        "",
        "Generado desde `index.html` con `python3 tools/export_notes.py`. No editar a mano: los textos se cambian en las notas de cada diapositiva.",
        "",
        "Cada **→** marca que hay que avanzar una etapa (flecha derecha) antes de decir ese punto. «En pantalla» es lo único que ve el jurado; todo lo demás se dice. Las diapositivas A1–A8 del apéndice se cuentan mientras entrena la demo.",
        "",
        "## Reparto",
        "",
        "| Quién | Diapositivas | Tiempo |",
        "|---|---|---|",
    ]
    for block in speaker_blocks(talk):
        span = block["first"] if block["first"] == block["last"] else f"{block['first']}–{block['last']}"
        extra = f" + demo {fmt(block['after'])}" if block["after"] else ""
        lines.append(f"| {block['speaker']} | {span} | {fmt(block['time'])}{extra} |")
    total = sum(s["time"] + s["after"] for s in talk)
    lines += [f"| **Total** | | **{fmt(total)}** |", ""]

    lines += ["## Línea de tiempo", "", "| Desde | Hasta | # | Diapositiva | Quién | En pantalla |", "|---|---|---|---|---|---|"]
    clock = 0
    starts = []
    for slide in talk:
        starts.append(clock)
        lines.append(f"| {fmt(clock)} | {fmt(clock + slide['time'])} | {slide['label']} | {slide['title']} | {slide['speaker']} | {slide['keywords']} |")
        clock += slide["time"]
        if slide["after"]:
            lines.append(f"| {fmt(clock)} | {fmt(clock + slide['after'])} | — | Fin del entrenamiento y página de la demo | {slide['speaker']} | ver `GUION_DEMO.md` |")
            clock += slide["after"]
    lines.append("")

    lines += ["## Qué dice cada uno", ""]
    for slide, start in zip(talk, starts):
        lines += [
            f"### {fmt(start)} → {fmt(start + slide['time'])} · {slide['label']} · {slide['title']}",
            "",
            f"**{slide['speaker']}** · {fmt(slide['time'])} · en pantalla: {slide['keywords']}",
            "",
            f"> {slide['idea']}",
            "",
            slide["oral"],
            "",
            f"**Pase:** {slide['trans']}",
            "",
            f"Respaldo: {slide['src']}",
            "",
        ]

    lines += ["## Respaldo (si falla la demo o para preguntas)", ""]
    for slide in backup:
        lines += [f"### {slide['label']} · {slide['title']}", "", f"> {slide['idea']}", "", slide["oral"], "", f"Respaldo: {slide['src']}", ""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    for block in speaker_blocks(talk):
        print(f"{block['speaker']}: {fmt(block['time'] + block['after'])}")
    print(f"total: {fmt(total)}")
    if total != TARGET_SECONDS:
        raise SystemExit(f"los tiempos suman {fmt(total)}, no {fmt(TARGET_SECONDS)}")


if __name__ == "__main__":
    main()
