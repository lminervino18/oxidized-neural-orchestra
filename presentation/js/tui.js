/*
 * Renders real orchestui terminal buffers (data/captures.js) onto <canvas>.
 * Braille, box-drawing and block glyphs are drawn geometrically so the image
 * matches a terminal regardless of the fonts installed on the projector.
 */
(function () {
  const DATA = window.ONO_TUI;
  const CELL_RATIO = 2.05;
  const FONT = '"JetBrains Mono", "DejaVu Sans Mono", monospace';

  // [left, right, up, down] arm weights: 1 single, 2 double
  const BOX = {
    "─": [1, 1, 0, 0], "│": [0, 0, 1, 1], "┌": [0, 1, 0, 1], "┐": [1, 0, 0, 1],
    "└": [0, 1, 1, 0], "┘": [1, 0, 1, 0], "├": [0, 1, 1, 1], "┤": [1, 0, 1, 1],
    "┬": [1, 1, 0, 1], "┴": [1, 1, 1, 0], "┼": [1, 1, 1, 1],
    "═": [2, 2, 0, 0], "║": [0, 0, 2, 2], "╔": [0, 2, 0, 2], "╗": [2, 0, 0, 2],
    "╚": [0, 2, 2, 0], "╝": [2, 0, 2, 0],
  };

  const BLOCKS = {
    "█": [0, 0, 1, 1], "▀": [0, 0, 1, 0.5], "▄": [0, 0.5, 1, 0.5],
    "▌": [0, 0, 0.5, 1], "▐": [0.5, 0, 0.5, 1],
  };

  const BRAILLE_DOTS = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [0, 3], [1, 3]];

  function parseCrop(attr) {
    if (!attr) return [0, 0, DATA.cols, 46];
    return attr.split(",").map(Number);
  }

  function drawBox(ctx, arms, x, y, w, h, color) {
    const cx = x + w / 2;
    const cy = y + h / 2;
    const lw = Math.max(1, w * 0.12);
    const gap = Math.max(1.5, w * 0.16);
    ctx.fillStyle = color;
    const hseg = (x0, x1, weight) => {
      if (weight === 2) {
        ctx.fillRect(x0, cy - gap - lw / 2, x1 - x0, lw);
        ctx.fillRect(x0, cy + gap - lw / 2, x1 - x0, lw);
      } else ctx.fillRect(x0, cy - lw / 2, x1 - x0, lw);
    };
    const vseg = (y0, y1, weight) => {
      if (weight === 2) {
        ctx.fillRect(cx - gap - lw / 2, y0, lw, y1 - y0);
        ctx.fillRect(cx + gap - lw / 2, y0, lw, y1 - y0);
      } else ctx.fillRect(cx - lw / 2, y0, lw, y1 - y0);
    };
    const [l, r, u, d] = arms;
    const ext = arms.includes(2) ? gap + lw / 2 : lw / 2;
    if (l) hseg(x, cx + (r ? 0 : ext), l);
    if (r) hseg(cx - (l ? 0 : ext), x + w, r);
    if (u) vseg(y, cy + (d ? 0 : ext), u);
    if (d) vseg(cy - (u ? 0 : ext), y + h, d);
  }

  function drawBraille(ctx, code, x, y, w, h, color) {
    const bits = code - 0x2800;
    const r = Math.max(0.8, w * 0.13);
    ctx.fillStyle = color;
    for (let i = 0; i < 8; i++) {
      if (!(bits & (1 << i))) continue;
      const [dx, dy] = BRAILLE_DOTS[i];
      const px = x + w * (dx === 0 ? 0.3 : 0.7);
      const py = y + h * (0.125 + 0.25 * dy);
      ctx.fillRect(px - r, py - r, r * 2, r * 2);
    }
  }

  function paint(canvas, rowIds, crop, cssWidth) {
    const [cx, cy, cw, ch] = crop;
    const cellW = cssWidth / cw;
    const cellH = cellW * CELL_RATIO;
    const cssHeight = cellH * ch;
    const scale = (window.devicePixelRatio || 1) * (window.ONO_STAGE_SCALE || 1);

    canvas.style.height = cssHeight + "px";
    canvas.width = Math.round(cssWidth * scale);
    canvas.height = Math.round(cssHeight * scale);

    const ctx = canvas.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cssWidth, cssHeight);
    ctx.textBaseline = "middle";

    const fontPx = cellW / 0.6;
    for (let r = 0; r < ch; r++) {
      const runs = DATA.rows[rowIds[cy + r]];
      if (!runs) continue;
      let col = 0;
      for (const [text, fg, bg, flags] of runs) {
        const color = DATA.palette[fg] || "#c8d0c8";
        const chars = Array.from(text);
        ctx.font = `${flags & 1 ? 700 : 400} ${fontPx}px ${FONT}`;
        for (const chr of chars) {
          const c = col - cx;
          col++;
          if (c < 0 || c >= cw) continue;
          const x = c * cellW;
          const y = r * cellH;
          if (bg >= 0 && DATA.palette[bg] !== "#000000") {
            ctx.fillStyle = DATA.palette[bg];
            ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
          }
          if (chr === " ") continue;
          const code = chr.codePointAt(0);
          ctx.globalAlpha = flags & 2 ? 0.6 : 1;
          if (code >= 0x2800 && code <= 0x28ff) drawBraille(ctx, code, x, y, cellW, cellH, color);
          else if (BOX[chr]) drawBox(ctx, BOX[chr], x, y, cellW, cellH, color);
          else if (BLOCKS[chr]) {
            const [bx, by, bw, bh] = BLOCKS[chr];
            ctx.fillStyle = color;
            ctx.fillRect(x + bx * cellW, y + by * cellH, bw * cellW + 0.4, bh * cellH + 0.4);
          } else {
            ctx.fillStyle = color;
            ctx.fillText(chr === "⟳" ? "↻" : chr, x, y + cellH / 2);
          }
          ctx.globalAlpha = 1;
        }
      }
    }
  }

  function render(el, rowIds) {
    const canvas = el.querySelector("canvas");
    const crop = parseCrop(el.dataset.crop);
    const width = el.clientWidth || el.getBoundingClientRect().width / (window.ONO_STAGE_SCALE || 1);
    if (!width) return;
    paint(canvas, rowIds, crop, width);
  }

  function renderShot(el) {
    const shot = DATA.shots[el.dataset.tui];
    if (shot) render(el, shot.rows);
  }

  function createPlayer(el, controls) {
    const rec = DATA.recording;
    const total = rec.frames.length;
    let frame = 0;
    let playing = false;
    let speed = 1;
    let timer = null;

    const fmt = (s) => {
      const m = Math.floor(s / 60);
      return `${String(m).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    };

    function show(i) {
      frame = Math.max(0, Math.min(total - 1, i));
      render(el, rec.frames[frame]);
      controls.range.value = frame;
      controls.time.textContent = `${fmt(rec.times[frame])} / ${fmt(rec.times[total - 1])}`;
    }

    function tick() {
      if (!playing) return;
      if (frame >= total - 1) return stop();
      const wait = (rec.times[frame + 1] - rec.times[frame]) * 1000 / speed;
      timer = setTimeout(() => { show(frame + 1); tick(); }, Math.max(60, wait));
    }

    function play() {
      if (frame >= total - 1) show(0);
      playing = true;
      controls.toggle.textContent = "⏸ pausar";
      tick();
    }

    function stop() {
      playing = false;
      clearTimeout(timer);
      controls.toggle.textContent = "▶ reproducir";
    }

    controls.range.max = total - 1;
    controls.range.addEventListener("input", () => { stop(); show(Number(controls.range.value)); });
    controls.toggle.addEventListener("click", () => (playing ? stop() : play()));
    controls.speed.addEventListener("click", () => {
      speed = speed === 1 ? 4 : 1;
      controls.speed.textContent = `${speed}×`;
    });

    return { show: () => show(frame), stop, redraw: () => show(frame) };
  }

  window.ONO_TUI_VIEW = { renderShot, createPlayer, available: !!DATA };
})();
