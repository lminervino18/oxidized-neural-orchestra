"""Bundle index.html and every local resource into dist/ono-defensa.html.

Usage (from presentation/):
    python3 tools/build_single.py

Inlines the stylesheet (fonts as data URIs), scripts and images, so the
result is a single file that opens offline with a double click.
"""

import base64
import mimetypes
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "ono-defensa.html"


def data_uri(path):
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if path.suffix == ".woff2":
        mime = "font/woff2"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def inline_css(match):
    css_path = ROOT / match.group(1)
    css = css_path.read_text(encoding="utf-8")
    css = re.sub(r'url\("([^"]+)"\)', lambda m: f'url("{data_uri((css_path.parent / m.group(1)).resolve())}")', css)
    return f"<style>\n{css}\n</style>"


def inline_script(match):
    js = (ROOT / match.group(1)).read_text(encoding="utf-8")
    return "<script>\n" + js.replace("</script", "<\\/script") + "\n</script>"


def inline_img(match):
    return f'src="{data_uri(ROOT / match.group(1))}"'


def main():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', inline_css, html)
    html = re.sub(r'<script src="([^"]+)"></script>', inline_script, html)
    html = re.sub(r'src="(assets/[^"]+)"', inline_img, html)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
