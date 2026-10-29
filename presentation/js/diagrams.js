/*
 * SVG diagrams driven by paused GSAP timelines.
 * Each builder returns a controller; timeline labels "s0".."sN" mark the
 * static end state of every step, so stepping back is a seek and stepping
 * forward tweens between consecutive labels.
 */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  const C = {
    text: v("--text"), text2: v("--text-2"), muted: v("--muted"), line: v("--line"),
    lineStrong: v("--line-strong"), accent: v("--accent"), panel: v("--panel"),
    ar: v("--ar"), ps: v("--ps"), arHi: v("--ar-hi"), psHi: v("--ps-hi"), ss: v("--ss"),
    grad: v("--grad"), param: v("--param"), ctrl: v("--ctrl"), warn: v("--warn"),
  };
  const WORKER = ["#2fe38c", "#4fd8e8", "#d58cf0", "#f2e05a"];
  let svgCount = 0;


  function svgRoot(container, w, h) {
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.setAttribute("role", "img");
    container.appendChild(svg);
    const uid = `d${++svgCount}`;
    svg.ref = (name) => `url(#${uid}-${name})`;
    const defs = add(svg, "defs");
    arrowMarker(defs, `${uid}-arrow-line`, C.lineStrong);
    arrowMarker(defs, `${uid}-arrow-accent`, C.accent);
    arrowMarker(defs, `${uid}-arrow-muted`, C.muted);
    const hatch = add(defs, "pattern", { id: `${uid}-hatch`, width: 10, height: 10, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    add(hatch, "rect", { width: 10, height: 10, fill: "#0a120d" });
    add(hatch, "line", { x1: 0, y1: 0, x2: 0, y2: 10, stroke: C.muted, "stroke-width": 3, opacity: 0.55 });
    return svg;
  }

  function arrowMarker(defs, id, color) {
    const m = add(defs, "marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
    add(m, "path", { d: "M0,0 L10,5 L0,10 z", fill: color });
  }

  function add(parent, tag, attrs = {}) {
    const node = document.createElementNS(NS, tag);
    for (const [k, val] of Object.entries(attrs)) node.setAttribute(k, val);
    parent.appendChild(node);
    return node;
  }

  function label(parent, x, y, str, attrs = {}) {
    const t = add(parent, "text", { x, y, fill: C.text2, "font-size": 22, "text-anchor": "middle", "dominant-baseline": "middle", ...attrs });
    t.textContent = str;
    return t;
  }

  function group(parent, attrs = {}) { return add(parent, "g", attrs); }

  function richLabel(parent, x, y, parts, attrs = {}) {
    const t = add(parent, "text", { x, y, fill: C.text2, "font-size": 22, "text-anchor": "middle", "dominant-baseline": "middle", ...attrs });
    for (const [str, kind] of parts) {
      const span = add(t, "tspan", kind === "sub" ? { "baseline-shift": "sub", "font-size": "0.72em" } : {});
      span.textContent = str;
    }
    return t;
  }

  function gradPacket(parent, r = 10) {
    const g = group(parent);
    add(g, "circle", { r, fill: C.grad, stroke: "#000", "stroke-width": 2 });
    return g;
  }

  function paramPacket(parent, s = 18) {
    const g = group(parent);
    add(g, "rect", { x: -s / 2, y: -s / 2, width: s, height: s, rx: 2, fill: C.param, stroke: "#000", "stroke-width": 2 });
    return g;
  }

  function ctrlPacket(parent, s = 13) {
    const g = group(parent);
    add(g, "rect", { x: -s / 2, y: -s / 2, width: s, height: s, fill: C.ctrl, stroke: "#000", "stroke-width": 2, transform: "rotate(45)" });
    return g;
  }

  function legend(parent, x, y, items) {
    const g = group(parent);
    let cx = x;
    for (const [kind, text] of items) {
      const icon = kind === "grad" ? gradPacket(g, 8) : kind === "param" ? paramPacket(g, 15) : ctrlPacket(g, 11);
      icon.setAttribute("transform", `translate(${cx + 8},${y})`);
      label(g, cx + 26, y + 1, text, { "text-anchor": "start", "font-size": 19, fill: C.muted });
      cx += 26 + text.length * 11.5 + 34;
    }
    return g;
  }

  function arcPoints(cx, cy, r, a0, a1, n = 10) {
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const a = (a0 + (a1 - a0) * (i / n)) * Math.PI / 180;
      pts.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
    }
    return pts;
  }

  function arcPath(cx, cy, r, a0, a1) {
    const [p0, p1] = arcPoints(cx, cy, r, a0, a1, 1);
    const large = Math.abs(a1 - a0) > 180 ? 1 : 0;
    return `M${p0.x},${p0.y} A${r},${r} 0 ${large} 1 ${p1.x},${p1.y}`;
  }

  function lineLen(el) {
    try { return el.getTotalLength(); } catch (e) { return 1000; }
  }

  function draw(tl, el, at, dur = 0.8) {
    const len = lineLen(el);
    gsap.set(el, { attr: { "stroke-dasharray": len, "stroke-dashoffset": len } });
    tl.to(el, { attr: { "stroke-dashoffset": 0 }, duration: dur, ease: "power2.inOut" }, at);
  }

  function show(tl, targets, at, dur = 0.45, extra = {}) {
    gsap.set(targets, { autoAlpha: 0 });
    tl.to(targets, { autoAlpha: 1, duration: dur, ease: "power2.out", ...extra }, at);
  }

  function hide(tl, targets, at, dur = 0.35) {
    tl.to(targets, { autoAlpha: 0, duration: dur, ease: "power1.in" }, at);
  }

  function travel(tl, packet, from, to, at, dur = 0.8) {
    gsap.set(packet, { autoAlpha: 0, x: from.x, y: from.y });
    tl.set(packet, { autoAlpha: 1, x: from.x, y: from.y }, at);
    tl.to(packet, { x: to.x, y: to.y, duration: dur, ease: "power1.inOut" }, at);
    tl.set(packet, { autoAlpha: 0 }, at + dur + 0.02);
  }

  function travelPath(tl, packet, pts, at, dur = 0.9) {
    gsap.set(packet, { autoAlpha: 0, x: pts[0].x, y: pts[0].y });
    tl.set(packet, { autoAlpha: 1, x: pts[0].x, y: pts[0].y }, at);
    tl.to(packet, { keyframes: pts.slice(1).map((p) => ({ x: p.x, y: p.y })), duration: dur, ease: "none" }, at);
    tl.set(packet, { autoAlpha: 0 }, at + dur + 0.02);
  }


  function controller(tl, steps, ambientFor) {
    let current = 0;
    let running = null;
    let ambient = null;
    let ambientDelay = null;
    tl.pause(0);

    const stopAmbient = () => {
      if (ambientDelay) { ambientDelay.kill(); ambientDelay = null; }
      if (ambient) { ambient.kill(); (ambient.data || []).forEach((node) => node.remove()); ambient = null; }
    };

    const startAmbient = (delay) => {
      stopAmbient();
      if (!ambientFor || window.ONO_MOTION === "reduced") return;
      const make = () => { ambient = ambientFor(current) || null; };
      if (delay > 0) ambientDelay = gsap.delayedCall(delay, make);
      else make();
    };

    return {
      steps,
      go(step, animate) {
        if (running) { running.kill(); running = null; }
        const reduced = window.ONO_MOTION === "reduced";
        let delay = 0;
        if (animate && !reduced && step === current + 1) {
          tl.seek(`s${current}`);
          running = tl.tweenFromTo(`s${current}`, `s${step}`);
          delay = running.duration();
        } else if (animate && !reduced && step === 0 && current === 0) {
          tl.seek(0);
          running = tl.tweenFromTo(0, "s0");
          delay = running.duration();
        } else {
          tl.seek(`s${step}`);
        }
        current = step;
        startAmbient(delay);
      },
      replay() {
        if (current === 0) this.go(0, true);
        else { const target = current; current = target - 1; this.go(target, true); }
      },
      leave() {
        if (running) { running.progress(1); running.kill(); running = null; }
        stopAmbient();
      },
      motionChanged() { startAmbient(0); },
    };
  }


  function machines(container) {
    const svg = svgRoot(container, 560, 640);
    const tl = gsap.timeline();
    const big = group(svg);
    add(big, "rect", { x: 40, y: 70, width: 190, height: 330, rx: 6, fill: C.panel, stroke: C.muted, "stroke-width": 2, "stroke-dasharray": "8 7" });
    for (let i = 0; i < 6; i++) add(big, "rect", { x: 66, y: 102 + i * 46, width: 138, height: 24, rx: 3, fill: "none", stroke: C.muted, "stroke-width": 1.5 });
    label(big, 135, 440, "un equipo", { fill: C.muted });
    label(big, 135, 470, "de gran porte", { fill: C.muted });

    const small = group(svg);
    const pos = [[340, 90], [460, 90], [340, 250], [460, 250]];
    const net = group(small);
    add(net, "line", { x1: 340, y1: 380, x2: 460, y2: 380, stroke: C.accent, "stroke-width": 2 });
    const boxes = pos.map(([x, y]) => {
      const g = group(small);
      add(g, "rect", { x: x - 46, y, width: 92, height: 110, rx: 5, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2 });
      add(g, "rect", { x: x - 30, y: y + 22, width: 60, height: 14, rx: 2, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
      add(g, "rect", { x: x - 30, y: y + 48, width: 60, height: 14, rx: 2, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
      return g;
    });
    const links = pos.map(([x, y]) => add(net, "line", { x1: x, y1: y + 110, x2: x, y2: 380, stroke: C.accent, "stroke-width": 2 }));
    label(small, 400, 440, "varias máquinas", { fill: C.text2 });
    label(small, 400, 470, "modestas", { fill: C.text2 });
    const sum = label(svg, 280, 570, "Σ cómputo · Σ memoria", { fill: C.accent, "font-size": 26 });

    show(tl, big, 0.1, 0.5);
    tl.addLabel("s0");
    show(tl, [small, ...boxes], "s0", 0.5, { stagger: 0.08 });
    gsap.set(net, { autoAlpha: 0 });
    tl.addLabel("s1");
    tl.to(net, { autoAlpha: 1, duration: 0.2 }, "s1");
    links.forEach((l, i) => draw(tl, l, tl.duration() - 0.2 + i * 0.05, 0.5));
    show(tl, sum, ">-0.1", 0.4);
    tl.to(big, { opacity: 0.45, duration: 0.4 }, "<");
    tl.addLabel("s2");
    return controller(tl, 2);
  }


  function stack(container) {
    const svg = svgRoot(container, 1000, 620);
    const tl = gsap.timeline();
    const algos = [["Parameter Server", C.ps], ["Ring All-Reduce", C.ar], ["Strategy-Switch", C.ss]];
    const tops = algos.map(([name, color], i) => {
      const g = group(svg);
      const x = 120 + i * 290;
      add(g, "rect", { x, y: 40, width: 260, height: 110, rx: 6, fill: C.panel, stroke: color, "stroke-width": 2.5 });
      label(g, x + 130, 95, name, { fill: C.text, "font-size": 24 });
      return g;
    });
    const layers = [
      ["plano de control", "orchestrator"],
      ["motor de redes neuronales", "machine_learning"],
      ["capa de comunicación", "comms"],
    ].map(([name, crate], i) => {
      const g = group(svg);
      const y = 230 + i * 120;
      add(g, "rect", { x: 120, y, width: 840, height: 92, rx: 6, fill: "#07100b", stroke: C.lineStrong, "stroke-width": 2 });
      label(g, 150, y + 46, name, { "text-anchor": "start", fill: C.text, "font-size": 25 });
      label(g, 930, y + 46, crate, { "text-anchor": "end", fill: C.muted, "font-size": 21 });
      return g;
    });
    const varies = label(svg, 60, 95, "varía", { fill: C.accent, "font-size": 21, transform: "rotate(-90 60 95)" });
    const fixed = label(svg, 60, 400, "fijo", { fill: C.muted, "font-size": 21, transform: "rotate(-90 60 400)" });
    const brace = add(svg, "path", { d: "M100,40 L88,40 L88,150 L100,150", fill: "none", stroke: C.accent, "stroke-width": 2 });
    const brace2 = add(svg, "path", { d: "M100,230 L88,230 L88,552 L100,552", fill: "none", stroke: C.muted, "stroke-width": 2 });

    show(tl, layers.slice().reverse(), 0.05, 0.4, { stagger: 0.12 });
    show(tl, tops, 0.45, 0.4, { stagger: 0.1 });
    show(tl, [brace2, fixed], 0.8, 0.3);
    tl.addLabel("s0");
    show(tl, [brace, varies], "s0", 0.4);
    tl.addLabel("s1");
    return controller(tl, 1);
  }


  function dataparallel(container) {
    const svg = svgRoot(container, 1100, 720);
    const tl = gsap.timeline();
    const xs = [170, 423, 676, 929];

    const ds = group(svg);
    add(ds, "rect", { x: 70, y: 30, width: 960, height: 52, rx: 4, fill: "#0b1810", stroke: C.lineStrong, "stroke-width": 2 });
    label(ds, 550, 56, "dataset", { fill: C.text2 });

    const parts = xs.map((x, i) => {
      const g = group(svg);
      add(g, "rect", { x: -110, y: -18, width: 220, height: 36, rx: 3, fill: "#0b1810", stroke: WORKER[i], "stroke-width": 2 });
      label(g, 0, 1, `partición ${i}`, { "font-size": 18, fill: C.text2 });
      return g;
    });

    const workers = xs.map((x, i) => {
      const g = group(svg);
      add(g, "rect", { x: x - 105, y: 200, width: 210, height: 170, rx: 6, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2 });
      label(g, x, 228, `W${i}`, { fill: C.text, "font-size": 24 });
      const model = group(g);
      [0, 1, 2].forEach((k) => add(model, "rect", { x: x - 60, y: 258 + k * 26, width: 120, height: 16, rx: 2, fill: "none", stroke: C.accent, "stroke-width": 1.5 }));
      label(g, x, 352, "θ (réplica)", { "font-size": 17, fill: C.muted });
      return { g, model };
    });

    const grads = xs.map((x, i) => {
      const g = group(svg, { transform: `translate(${x},420)` });
      add(g, "circle", { r: 18, fill: C.grad, stroke: "#000", "stroke-width": 2 });
      label(g, 0, 1, `g${i}`, { fill: "#000", "font-size": 17, "font-weight": 700 });
      return g;
    });

    const agg = group(svg);
    add(agg, "rect", { x: 380, y: 560, width: 340, height: 70, rx: 6, fill: "#0b1810", stroke: C.grad, "stroke-width": 2 });
    label(agg, 550, 595, "ḡ = (1/N) Σ gᵢ", { fill: C.text, "font-size": 25 });
    const aggNote = label(svg, 550, 660, "agregación: Parameter Server o All-Reduce", { fill: C.muted, "font-size": 18 });

    const update = label(svg, 550, 470, "θ ← θ − η · ḡ   (la misma en todas las réplicas)", { fill: C.accent, "font-size": 23 });
    const down = xs.map(() => {
      const g = group(svg);
      add(g, "circle", { r: 13, fill: C.grad, stroke: "#000", "stroke-width": 2 });
      return g;
    });

    show(tl, ds, 0.05, 0.4);
    show(tl, workers.map((w) => w.g), 0.2, 0.4, { stagger: 0.08 });
    tl.addLabel("s0");

    parts.forEach((p, i) => {
      gsap.set(p, { autoAlpha: 0, x: 70 + 120 + i * 240, y: 56 });
      tl.to(p, { autoAlpha: 1, duration: 0.3 }, "s0");
      tl.to(p, { x: xs[i], y: 150 + 0, duration: 0.7, ease: "power2.inOut" }, "s0+=0.35");
    });
    tl.to(ds, { opacity: 0.25, duration: 0.4 }, "s0+=0.3");
    tl.addLabel("s1");
    gsap.set(update, { autoAlpha: 0 });

    show(tl, grads, "s1", 0.35, { stagger: 0.1, scale: 1, transformOrigin: "50% 50%" });
    tl.addLabel("s2");

    grads.forEach((g, i) => {
      tl.to(g, { x: 550 - xs[i], y: 140, scale: 0.6, duration: 0.7, ease: "power2.in" }, "s2+=" + i * 0.05);
    });
    show(tl, [agg, aggNote], "s2+=0.1", 0.4);
    tl.to(grads, { autoAlpha: 0, duration: 0.2 }, "s2+=0.8");
    tl.addLabel("s3");

    down.forEach((d, i) => travel(tl, d, { x: 550, y: 560 }, { x: xs[i], y: 372 }, tl.duration() - 0.02 + i * 0.04, 0.6));
    tl.to(workers.flatMap((w) => [...w.model.children]), { attr: { fill: "rgba(10,248,134,0.25)" }, duration: 0.35 }, ">-0.1");
    show(tl, update, "<", 0.4);
    tl.addLabel("s4");
    return controller(tl, 4);
  }


  function syncasync(container) {
    const svg = svgRoot(container, 1120, 660);
    const tl = gsap.timeline();
    const unit = 62;
    const x0 = 120;
    const rowH = 54;

    function lanes(y, title) {
      const g = group(svg);
      label(g, 0, y - 36, title, { "text-anchor": "start", fill: C.text, "font-size": 24 });
      for (let i = 0; i < 3; i++) label(g, 40, y + i * rowH + 20, `W${i}`, { "text-anchor": "start", fill: C.muted, "font-size": 20 });
      add(g, "line", { x1: x0, y1: y + 3 * rowH + 6, x2: 1100, y2: y + 3 * rowH + 6, stroke: C.line, "stroke-width": 1.5, "marker-end": svg.ref("arrow-muted") });
      label(g, 1100, y + 3 * rowH + 30, "tiempo", { "text-anchor": "end", fill: C.muted, "font-size": 17 });
      return g;
    }

    function bar(y, x, w, kind) {
      const fill = kind === "idle" ? svg.ref("hatch") : "rgba(17,173,98,0.35)";
      const stroke = kind === "idle" ? C.muted : C.ar;
      return add(svg, "rect", { x, y, width: w, height: 38, rx: 3, fill, stroke, "stroke-width": 1.5 });
    }

    const syncY = 60;
    const asyncY = 380;
    const syncLanes = lanes(syncY, "sincrónico");
    const asyncLanes = lanes(asyncY, "asincrónico");

    const work = [[3, 4.5, 2.5], [3.5, 2.8, 4.2]];
    const syncBars = [];
    const idle = [];
    const barriers = [];
    let t = 0;
    work.forEach((round) => {
      const max = Math.max(...round);
      round.forEach((d, w) => {
        syncBars.push(bar(syncY + w * rowH, x0 + t * unit, d * unit - 4, "work"));
        if (d < max) idle.push(bar(syncY + w * rowH, x0 + (t + d) * unit, (max - d) * unit - 4, "idle"));
      });
      t += max;
      const bx = x0 + t * unit - 2;
      const b = group(svg);
      add(b, "line", { x1: bx, y1: syncY - 10, x2: bx, y2: syncY + 3 * rowH - 6, stroke: C.accent, "stroke-width": 2.5 });
      label(b, bx, syncY - 22, "barrera", { fill: C.accent, "font-size": 16 });
      barriers.push(b);
      t += 0.35;
    });
    const idleNote = label(svg, x0 + 11.6 * unit, syncY + 1.5 * rowH, "espera", { fill: C.muted, "font-size": 19, "text-anchor": "start" });

    const asyncBars = [];
    const pushes = [];
    const seqs = [[3, 3.5, 3.1], [4.5, 2.8, 2.2], [2.5, 4.2, 3.4]];
    seqs.forEach((seq, w) => {
      let tt = 0;
      seq.forEach((d) => {
        asyncBars.push(bar(asyncY + w * rowH, x0 + tt * unit, d * unit - 4, "work"));
        const p = gradPacket(svg, 7);
        p.setAttribute("transform", `translate(${x0 + (tt + d) * unit - 4},${asyncY + w * rowH + 19})`);
        pushes.push(p);
        tt += d;
      });
    });
    const stale = group(svg);
    const sx = x0 + 4.5 * unit + 40;
    add(stale, "path", { d: `M${sx},${asyncY + rowH + 19} q20,-70 90,-80`, fill: "none", stroke: C.warn, "stroke-width": 1.5 });
    label(stale, sx + 100, asyncY + rowH - 66, "calculado sobre θ desactualizado", { fill: C.warn, "font-size": 18, "text-anchor": "start" });

    show(tl, syncLanes, 0.05, 0.3);
    syncBars.forEach((b, i) => { gsap.set(b, { scaleX: 0, transformOrigin: "0% 50%" }); tl.to(b, { scaleX: 1, duration: 0.45, ease: "power1.out" }, 0.2 + Math.floor(i / 3) * 0.9); });
    show(tl, idle.slice(0, 2), 0.7, 0.3);
    show(tl, barriers[0], 0.9, 0.25);
    show(tl, idle.slice(2), 1.6, 0.3);
    show(tl, barriers[1], 1.8, 0.25);
    show(tl, idleNote, 1.9, 0.3);
    gsap.set([asyncLanes, ...asyncBars, ...pushes, stale], { autoAlpha: 0 });
    tl.addLabel("s0");
    tl.to(asyncLanes, { autoAlpha: 1, duration: 0.3 }, "s0");
    asyncBars.forEach((b, i) => {
      gsap.set(b, { scaleX: 0, transformOrigin: "0% 50%" });
      tl.to(b, { autoAlpha: 1, scaleX: 1, duration: 0.35, ease: "power1.out" }, `s0+=${0.15 + (i % 3) * 0.32}`);
    });
    tl.to(pushes, { autoAlpha: 1, duration: 0.2, stagger: 0.03 }, "s0+=1");
    tl.to(stale, { autoAlpha: 1, duration: 0.3 }, "s0+=1.3");
    tl.addLabel("s1");
    return controller(tl, 1);
  }


  function arch(container) {
    const svg = svgRoot(container, 1100, 720);
    const tl = gsap.timeline();
    const ui = group(svg);
    add(ui, "rect", { x: 330, y: 20, width: 440, height: 56, rx: 5, fill: "#07100b", stroke: C.line, "stroke-width": 1.5 });
    label(ui, 550, 48, "orchestui  ·  orchestra-py", { fill: C.text2, "font-size": 21 });
    const orch = group(svg);
    add(orch, "rect", { x: 380, y: 96, width: 340, height: 82, rx: 6, fill: C.panel, stroke: C.accent, "stroke-width": 2.5 });
    label(orch, 550, 128, "orchestrator", { fill: C.text, "font-size": 26 });
    label(orch, 550, 158, "plano de control", { fill: C.muted, "font-size": 17 });

    const xs = [190, 430, 670, 910];
    const roles = ["worker", "worker", "worker", "server"];
    const nodes = xs.map((x, i) => {
      const g = group(svg);
      const box = add(g, "rect", { x: x - 95, y: 430, width: 190, height: 120, rx: 6, fill: C.panel, stroke: C.muted, "stroke-width": 2 });
      const generic = label(g, x, 478, "node", { fill: C.text2, "font-size": 24 });
      const role = label(g, x, 478, roles[i], { fill: roles[i] === "server" ? C.psHi : C.arHi, "font-size": 24 });
      label(g, x, 516, "mismo binario", { fill: C.muted, "font-size": 16 });
      return { g, box, generic, role, x };
    });

    const ctrl = xs.map((x) => add(svg, "line", { x1: 550, y1: 178, x2: x, y2: 430, stroke: C.ctrl, "stroke-width": 1.5, "stroke-dasharray": "6 6", opacity: 0.7 }));
    const handshake = label(svg, 550, 300, "Connect / Accept  ·  spec", { fill: C.ctrl, "font-size": 19 });
    const ctrlPk = xs.map(() => ctrlPacket(svg));

    const dataY = 620;
    const data = group(svg);
    xs.slice(0, 3).forEach((x) => add(data, "path", { d: `M${x},550 C${x},${dataY + 40} ${910},${dataY + 40} 910,550`, fill: "none", stroke: C.lineStrong, "stroke-width": 2 }));
    label(data, 550, 690, "plano de datos: gradientes y parámetros entre nodos", { fill: C.muted, "font-size": 18 });

    show(tl, [ui, orch], 0.05, 0.4);
    show(tl, nodes.map((n) => n.g), 0.2, 0.4, { stagger: 0.08 });
    gsap.set(nodes.map((n) => n.role), { autoAlpha: 0 });
    gsap.set(ctrl, { autoAlpha: 0 });
    gsap.set(handshake, { autoAlpha: 0 });
    gsap.set(data, { autoAlpha: 0 });
    tl.addLabel("s0");

    tl.to(ctrl, { autoAlpha: 0.7, duration: 0.3, stagger: 0.05 }, "s0");
    ctrlPk.forEach((p, i) => travel(tl, p, { x: 550, y: 180 }, { x: xs[i], y: 428 }, tl.duration() + 0.0 + i * 0.06 - 0.1, 0.6));
    tl.to(handshake, { autoAlpha: 1, duration: 0.3 }, "s0+=0.2");
    tl.addLabel("s1");

    nodes.forEach((n, i) => {
      tl.to(n.generic, { autoAlpha: 0, duration: 0.2 }, `s1+=${i * 0.08}`);
      tl.to(n.role, { autoAlpha: 1, duration: 0.3 }, `s1+=${0.15 + i * 0.08}`);
      tl.to(n.box, { attr: { stroke: roles[i] === "server" ? C.ps : C.ar }, duration: 0.3 }, `s1+=${0.15 + i * 0.08}`);
    });
    tl.addLabel("s2");
    tl.to(data, { autoAlpha: 1, duration: 0.4 }, "s2");
    tl.to(ctrl, { opacity: 0.25, duration: 0.4 }, "s2");
    tl.addLabel("s3");

    const ambientFor = (step) => {
      if (step !== 3) return null;
      const a = gsap.timeline({ repeat: -1, repeatDelay: 0.8 });
      a.data = [];
      xs.slice(0, 3).forEach((x, i) => {
        const g = gradPacket(svg, 8);
        const p = paramPacket(svg, 13);
        a.data.push(g, p);
        const curve = [];
        for (let k = 0; k <= 12; k++) {
          const t = k / 12;
          const bx = (1 - t) ** 3 * x + 3 * (1 - t) ** 2 * t * x + 3 * (1 - t) * t * t * 910 + t ** 3 * 910;
          const by = (1 - t) ** 3 * 550 + 3 * (1 - t) ** 2 * t * (dataY + 40) + 3 * (1 - t) * t * t * (dataY + 40) + t ** 3 * 550;
          curve.push({ x: bx, y: by });
        }
        travelPath(a, g, curve, 0.2 + i * 0.25, 1.4);
        travelPath(a, p, curve.slice().reverse(), 2.0 + i * 0.25, 1.4);
      });
      return a;
    };
    return controller(tl, 3, ambientFor);
  }


  function flatbuffer(container) {
    const svg = svgRoot(container, 1100, 620);
    const tl = gsap.timeline();
    const layers = [["capa 1", 0.3], ["capa 2", 0.5], ["capa 3", 0.2]];
    const bx = 60;
    const bw = 980;
    const by = 300;
    let acc = 0;
    const ranges = layers.map(([name, frac], i) => {
      const x = bx + acc * bw;
      const w = frac * bw;
      acc += frac;
      return { name, x, w, i };
    });

    const cards = ranges.map((r) => {
      const g = group(svg);
      const cx = r.x + r.w / 2;
      add(g, "rect", { x: cx - 110, y: 60, width: 220, height: 92, rx: 6, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2 });
      label(g, cx, 92, r.name, { fill: C.text, "font-size": 23 });
      label(g, cx, 124, "offset · len · forma", { fill: C.muted, "font-size": 16 });
      return g;
    });

    const buf = group(svg);
    const cells = ranges.map((r) => add(buf, "rect", { x: r.x + 2, y: by, width: r.w - 4, height: 70, fill: "rgba(10,248,134,0.10)", stroke: C.accent, "stroke-width": 1.5 }));
    for (let k = 0; k < 49; k++) add(buf, "line", { x1: bx + k * 20, y1: by + 52, x2: bx + k * 20, y2: by + 70, stroke: C.accent, "stroke-width": 1, opacity: 0.35 });
    label(buf, bx, by + 104, "buffer contiguo de f32 (parámetros de todo el modelo)", { "text-anchor": "start", fill: C.muted, "font-size": 18 });

    const views = ranges.map((r) => {
      const cx = r.x + r.w / 2;
      return add(svg, "path", { d: `M${cx},152 L${cx},${by - 4}`, stroke: C.accent, "stroke-width": 1.5, "stroke-dasharray": "5 5", fill: "none", "marker-end": svg.ref("arrow-accent") });
    });

    const s0 = group(svg);
    add(s0, "path", { d: `M${ranges[1].x + 4},${by + 130} L${ranges[1].x + 4},${by + 146} L${ranges[1].x + ranges[1].w - 4},${by + 146} L${ranges[1].x + ranges[1].w - 4},${by + 130}`, fill: "none", stroke: C.psHi, "stroke-width": 2 });
    label(s0, ranges[1].x + ranges[1].w / 2, by + 172, "servidor 0", { fill: C.psHi, "font-size": 20 });
    const s1 = group(svg);
    [ranges[0], ranges[2]].forEach((r) => add(s1, "path", { d: `M${r.x + 4},${by + 130} L${r.x + 4},${by + 146} L${r.x + r.w - 4},${by + 146} L${r.x + r.w - 4},${by + 130}`, fill: "none", stroke: C.ps, "stroke-width": 2 }));
    label(s1, ranges[0].x + ranges[0].w / 2, by + 172, "servidor 1", { fill: C.ps, "font-size": 20 });
    label(s1, ranges[2].x + ranges[2].w / 2, by + 172, "servidor 1", { fill: C.ps, "font-size": 20 });
    const greedy = label(svg, 550, by + 214, "capas enteras, de mayor a menor, al servidor menos cargado", { fill: C.muted, "font-size": 18 });

    const wire = group(svg);
    const pk = group(wire);
    add(pk, "rect", { x: -60, y: -16, width: 120, height: 32, rx: 3, fill: "rgba(10,248,134,0.18)", stroke: C.accent, "stroke-width": 1.5 });
    label(pk, 0, 1, "&[u8]", { fill: C.accent, "font-size": 17 });
    const wireText = label(svg, 550, 254, "bytes del tramo tal cual → red  (bytemuck, sin copias)", { fill: C.text2, "font-size": 19 });

    show(tl, cards, 0.05, 0.35, { stagger: 0.08 });
    show(tl, buf, 0.25, 0.4);
    views.forEach((vw, i) => draw(tl, vw, 0.5 + i * 0.1, 0.5));
    gsap.set([s0, s1, greedy, wire, wireText], { autoAlpha: 0 });
    tl.addLabel("s0");
    tl.to(cells[1], { attr: { fill: "rgba(111,176,255,0.20)", stroke: C.psHi }, duration: 0.35 }, "s0");
    tl.to(s0, { autoAlpha: 1, duration: 0.35 }, "s0");
    tl.to([cells[0], cells[2]], { attr: { fill: "rgba(61,139,230,0.16)", stroke: C.ps }, duration: 0.35 }, "s0+=0.25");
    tl.to(s1, { autoAlpha: 1, duration: 0.35 }, "s0+=0.25");
    tl.to(greedy, { autoAlpha: 1, duration: 0.3 }, "s0+=0.5");
    tl.addLabel("s1");
    tl.to([cards, views], { opacity: 0.25, duration: 0.3 }, "s1");
    travel(tl, pk, { x: ranges[1].x + ranges[1].w / 2, y: by + 35 }, { x: 990, y: 210 }, tl.duration() + 0.05, 0.9);
    tl.set(pk, { autoAlpha: 1 }, ">");
    tl.set(wire, { autoAlpha: 1 }, "s1");
    tl.to(wireText, { autoAlpha: 1, duration: 0.3 }, "s1+=0.3");
    tl.addLabel("s2");
    return controller(tl, 2);
  }


  function ps(container) {
    const svg = svgRoot(container, 1100, 720);
    const tl = gsap.timeline();
    const W = [{ x: 170, y: 150 }, { x: 170, y: 360 }, { x: 170, y: 570 }];
    const S0 = { x: 860, y: 250 };
    const S1 = { x: 860, y: 520 };

    const links0 = W.map((w) => add(svg, "line", { x1: w.x + 62, y1: w.y, x2: S0.x - 110, y2: S0.y, stroke: C.lineStrong, "stroke-width": 2 }));
    const links1 = W.map((w) => add(svg, "line", { x1: w.x + 62, y1: w.y, x2: S1.x - 110, y2: S1.y, stroke: C.lineStrong, "stroke-width": 2 }));

    const workers = W.map((w, i) => {
      const g = group(svg);
      const ring = add(g, "circle", { cx: w.x, cy: w.y, r: 62, fill: C.panel, stroke: C.ar, "stroke-width": 2.5 });
      label(g, w.x, w.y - 12, `W${i}`, { fill: C.text, "font-size": 25 });
      label(g, w.x, w.y + 20, "partición " + i, { fill: C.muted, "font-size": 15 });
      return { g, ring };
    });

    function server(pos, name, inner) {
      const g = group(svg);
      const box = add(g, "rect", { x: pos.x - 110, y: pos.y - 80, width: 220, height: 160, rx: 6, fill: C.panel, stroke: C.ps, "stroke-width": 2.5 });
      label(g, pos.x, pos.y - 50, name, { fill: C.text, "font-size": 24 });
      const theta = label(g, pos.x, pos.y - 8, inner, { fill: C.psHi, "font-size": 21 });
      const opt = label(g, pos.x, pos.y + 40, "optimizador (estado)", { fill: C.muted, "font-size": 15 });
      return { g, box, theta, opt };
    }

    const srv0 = server(S0, "servidor 0", "θ");
    const srv1 = server(S1, "servidor 1", "θ · capas 1, 3");
    const srv0split = label(svg, S0.x, S0.y - 8, "θ · capa 2", { fill: C.psHi, "font-size": 21 });
    const thetaNew = label(svg, S0.x, S0.y - 8, "θ′", { fill: C.accent, "font-size": 23 });

    const pulls = W.map(() => paramPacket(svg));
    const pushes = W.map(() => gradPacket(svg));
    const localG = W.map((w) => {
      const g = group(svg, { transform: `translate(${w.x + 48},${w.y - 52})` });
      add(g, "circle", { r: 14, fill: C.grad, stroke: "#000", "stroke-width": 2 });
      label(g, 0, 1, "g", { fill: "#000", "font-size": 15, "font-weight": 700 });
      return g;
    });

    const counter = group(svg);
    add(counter, "rect", { x: S0.x + 120, y: S0.y - 26, width: 100, height: 52, rx: 4, fill: "#0b1810", stroke: C.grad, "stroke-width": 1.5 });
    const cnt = [1, 2, 3].map((k) => label(counter, S0.x + 170, S0.y + 1, `${k}/3`, { fill: C.grad, "font-size": 22 }));
    const avg = richLabel(svg, S0.x, S0.y + 108, [["Σg / N"], ["W", "sub"], [" → Adam → θ′"]], { fill: C.accent, "font-size": 20 });
    const barrierNote = label(svg, S0.x + 170, S0.y - 48, "barrier", { fill: C.muted, "font-size": 16 });

    const leg = legend(svg, 40, 700, [["grad", "gradiente"], ["param", "parámetros"]]);

    show(tl, [...workers.map((w) => w.g), srv0.g, ...links0, leg], 0.05, 0.4, { stagger: 0.05 });
    gsap.set([srv1.g, ...links1, srv0split, thetaNew, counter, avg, barrierNote, ...localG, ...cnt], { autoAlpha: 0 });
    tl.addLabel("s0");

    pulls.forEach((p, i) => travel(tl, p, { x: S0.x - 110, y: S0.y }, { x: W[i].x + 62, y: W[i].y }, tl.duration() + i * 0.08, 0.8));
    tl.addLabel("s1");

    workers.forEach((w, i) => {
      tl.to(w.ring, { attr: { fill: "rgba(17,173,98,0.28)" }, duration: 0.25, yoyo: true, repeat: 3 }, `s1+=${i * 0.1}`);
    });
    tl.to(localG, { autoAlpha: 1, duration: 0.3, stagger: 0.1 }, "s1+=0.9");
    tl.addLabel("s2");

    tl.to([counter, barrierNote], { autoAlpha: 1, duration: 0.25 }, "s2");
    pushes.forEach((p, i) => {
      const at = tl.duration() + 0.05 + i * 0.35;
      tl.to(localG[i], { autoAlpha: 0, duration: 0.15 }, at);
      travel(tl, p, { x: W[i].x + 62, y: W[i].y }, { x: S0.x - 110, y: S0.y }, at, 0.8);
      tl.to(cnt[i], { autoAlpha: 1, duration: 0.1 }, at + 0.8);
      if (i > 0) tl.to(cnt[i - 1], { autoAlpha: 0, duration: 0.1 }, at + 0.8);
    });
    tl.addLabel("s3");

    tl.to(srv0.box, { attr: { stroke: C.accent }, duration: 0.3 }, "s3");
    tl.to(avg, { autoAlpha: 1, duration: 0.35 }, "s3+=0.1");
    tl.to(srv0.theta, { autoAlpha: 0, duration: 0.2 }, "s3+=0.6");
    tl.to(thetaNew, { autoAlpha: 1, duration: 0.3 }, "s3+=0.7");
    tl.to(srv0.box, { attr: { stroke: C.ps }, duration: 0.4 }, "s3+=1");
    tl.addLabel("s4");

    tl.to([avg, counter, barrierNote, thetaNew, ...cnt], { autoAlpha: 0, duration: 0.3 }, "s4");
    tl.to(srv0split, { autoAlpha: 1, duration: 0.3 }, "s4+=0.3");
    tl.to([srv1.g, ...links1], { autoAlpha: 1, duration: 0.4 }, "s4+=0.3");
    tl.addLabel("s5");

    const ambientFor = (step) => {
      if (step !== 5) return null;
      const a = gsap.timeline({ repeat: -1, repeatDelay: 1.2 });
      a.data = [];
      [S0, S1].forEach((S, k) => W.forEach((w, i) => {
        const g = gradPacket(svg, 8);
        const p = paramPacket(svg, 13);
        a.data.push(g, p);
        gsap.set([g, p], { opacity: 0.75 });
        travel(a, g, { x: w.x + 62, y: w.y }, { x: S.x - 110, y: S.y }, i * 0.2 + k * 0.1, 1.1);
        travel(a, p, { x: S.x - 110, y: S.y }, { x: w.x + 62, y: w.y }, 1.6 + i * 0.2 + k * 0.1, 1.1);
      }));
      return a;
    };
    return controller(tl, 5, ambientFor);
  }


  function ring(container) {
    const svg = svgRoot(container, 1100, 800);
    const tl = gsap.timeline();
    const n = 4;
    const cx = 550;
    const cy = 400;
    const R = 220;
    const angles = [-90, 0, 90, 180];
    const pos = angles.map((a) => ({ x: cx + R * Math.cos(a * Math.PI / 180), y: cy + R * Math.sin(a * Math.PI / 180) }));
    const cellW = 46;
    const cellH = 38;
    const rowW = n * cellW + (n - 1) * 6;
    const rowOrigin = [
      { x: pos[0].x - rowW / 2, y: pos[0].y - 40 - 22 - cellH - 24 },
      { x: pos[1].x + 80, y: pos[1].y - cellH / 2 },
      { x: pos[2].x - rowW / 2, y: pos[2].y + 40 + 30 },
      { x: pos[3].x - 80 - rowW, y: pos[3].y - cellH / 2 },
    ];

    const arcs = angles.map((a) => add(svg, "path", { d: arcPath(cx, cy, R, a + 17, a + 73), fill: "none", stroke: C.lineStrong, "stroke-width": 2.5, "marker-end": svg.ref("arrow-line") }));

    const nodes = pos.map((p, i) => {
      const g = group(svg);
      add(g, "rect", { x: p.x - 58, y: p.y - 30, width: 116, height: 60, rx: 6, fill: C.panel, stroke: WORKER[i], "stroke-width": 2.5 });
      label(g, p.x, p.y + 1, `W${i}`, { fill: C.text, "font-size": 25 });
      return g;
    });

    const contrib = [];
    for (let w = 0; w < n; w++) contrib.push(Array.from({ length: n }, () => new Set([w])));

    function cellGroup(parent, x, y, set) {
      const g = group(parent, { transform: `translate(${x},${y})` });
      add(g, "rect", { x: 0, y: 0, width: cellW, height: cellH, rx: 3, fill: "#050a07", stroke: C.lineStrong, "stroke-width": 1.5 });
      const slots = [];
      for (let k = 0; k < n; k++) {
        slots.push(add(g, "rect", { x: 4 + k * 9.5, y: 6, width: 7.5, height: cellH - 12, rx: 1.5, fill: WORKER[k], opacity: set.has(k) ? 1 : 0 }));
      }
      return { g, slots };
    }

    const cells = [];
    const rows = [];
    for (let w = 0; w < n; w++) {
      const rg = group(svg);
      rows.push(rg);
      const o = rowOrigin[w];
      cells.push([]);
      for (let c = 0; c < n; c++) {
        cells[w].push(cellGroup(rg, o.x + c * (cellW + 6), o.y, contrib[w][c]));
        label(rg, o.x + c * (cellW + 6) + cellW / 2, o.y + cellH + 16, `c${c}`, { fill: C.muted, "font-size": 14 });
      }
    }

    const phaseTexts = {};
    function phase(key, line1, line2) {
      const g = group(svg);
      label(g, cx, cy - 14, line1, { fill: C.text, "font-size": 23 });
      if (line2) label(g, cx, cy + 18, line2, { fill: C.muted, "font-size": 18 });
      phaseTexts[key] = g;
      return g;
    }
    const allPhases = [
      phase("split", "4 fragmentos", "por gradiente"),
      phase("rs1", "reduce-scatter", "iteración 1 / 3"), phase("rs2", "reduce-scatter", "iteración 2 / 3"), phase("rs3", "reduce-scatter", "iteración 3 / 3"),
      phase("rsd", "cada worker tiene", "un fragmento completo"),
      phase("ag1", "all-gather", "iteración 1 / 3"), phase("ag2", "all-gather", "iteración 2 / 3"), phase("ag3", "all-gather", "iteración 3 / 3"),
      phase("done", "÷ 4 → ḡ", "mismo gradiente en todos"),
    ];
    const optim = pos.map((p, i) => {
      const g = group(svg);
      const o = rowOrigin[i];
      label(g, o.x + rowW / 2, i === 2 ? o.y + cellH + 44 : o.y - 18, "÷4 · optimizador local", { fill: C.accent, "font-size": 16 });
      return g;
    });

    show(tl, [...arcs, ...nodes], 0.05, 0.4, { stagger: 0.05 });
    gsap.set([...rows, ...allPhases, ...optim], { autoAlpha: 0 });
    tl.addLabel("s0");
    tl.to(rows, { autoAlpha: 1, duration: 0.4, stagger: 0.08 }, "s0");
    tl.to(phaseTexts.split, { autoAlpha: 1, duration: 0.3 }, "s0");
    tl.addLabel("s1");

    const PK_DUR = 0.85;
    const ITER_GAP = 1.25;
    let currentPhase = phaseTexts.split;

    function runIteration(kind, k, at) {
      const snapshot = contrib.map((row) => row.map((s) => new Set(s)));
      for (let p = 0; p < n; p++) {
        const c = kind === "rs" ? ((p - k) % n + n) % n : ((p + 1 - k) % n + n) % n;
        const q = (p + 1) % n;
        const sent = snapshot[p][c];
        const packetCell = cellGroup(svg, 0, 0, sent);
        const pts = arcPoints(cx, cy, R, angles[p] + 17, angles[p] + 73, 10).map((pt) => ({ x: pt.x - cellW / 2, y: pt.y - cellH / 2 }));
        travelPath(tl, packetCell.g, pts, at, PK_DUR);
        tl.to(cells[p][c].g, { opacity: 0.55, duration: 0.15, yoyo: true, repeat: 1 }, at);
        const result = kind === "rs" ? new Set([...contrib[q][c], ...sent]) : new Set(sent);
        contrib[q][c] = result;
        cells[q][c].slots.forEach((slot, s) => {
          tl.to(slot, { attr: { opacity: result.has(s) ? 1 : 0 }, duration: 0.2 }, at + PK_DUR);
        });
      }
    }

    function switchPhase(key, at) {
      tl.to(currentPhase, { autoAlpha: 0, duration: 0.2 }, at);
      tl.to(phaseTexts[key], { autoAlpha: 1, duration: 0.2 }, at + 0.1);
      currentPhase = phaseTexts[key];
    }

    let at = tl.duration() + 0.05;
    for (let k = 0; k < n - 1; k++) {
      switchPhase(`rs${k + 1}`, at);
      runIteration("rs", k, at + 0.25);
      at += ITER_GAP;
    }
    switchPhase("rsd", at);
    for (let p = 0; p < n; p++) {
      const c = (p + 1) % n;
      tl.to(cells[p][c].g.querySelector("rect"), { attr: { stroke: C.accent }, duration: 0.3 }, at + 0.1);
    }
    tl.addLabel("s2", at + 0.5);

    at = tl.labels.s2 + 0.05;
    for (let k = 0; k < n - 1; k++) {
      switchPhase(`ag${k + 1}`, at);
      runIteration("ag", k, at + 0.25);
      at += ITER_GAP;
    }
    tl.addLabel("s3", at);

    switchPhase("done", tl.labels.s3 + 0.05);
    cells.flat().forEach((cell) => tl.to(cell.g.querySelector("rect"), { attr: { stroke: C.accent }, duration: 0.3 }, tl.labels.s3 + 0.15));
    tl.to(optim, { autoAlpha: 1, duration: 0.35, stagger: 0.06 }, tl.labels.s3 + 0.3);
    tl.addLabel("s4");
    return controller(tl, 4);
  }


  function dsync(container) {
    const svg = svgRoot(container, 1100, 700);
    const tl = gsap.timeline();
    const SR = 0.1;
    const X0 = 110, X1 = 1000, Y0 = 600, Y1 = 150;
    const xMax = 17, yMax = 3.4;
    const sx = (nw) => X0 + (nw / xMax) * (X1 - X0);
    const sy = (d) => Y0 - (d / yMax) * (Y0 - Y1);
    const psF = (nw) => 2 * nw * SR;
    const arF = (nw) => 2 * (1 - 1 / nw) * SR;

    const axes = group(svg);
    for (let d = 0; d <= 3; d += 0.5) {
      add(axes, "line", { x1: X0, y1: sy(d), x2: X1, y2: sy(d), stroke: C.line, "stroke-width": 1 });
      label(axes, X0 - 16, sy(d), d.toFixed(1).replace(".", ","), { "text-anchor": "end", fill: C.muted, "font-size": 18 });
    }
    [2, 4, 6, 8, 10, 12, 14, 16].forEach((nw) => label(axes, sx(nw), Y0 + 28, String(nw), { fill: C.muted, "font-size": 18 }));
    add(axes, "line", { x1: X0, y1: Y0, x2: X1, y2: Y0, stroke: C.lineStrong, "stroke-width": 1.5 });
    richLabel(axes, (X0 + X1) / 2, Y0 + 66, [["N"], ["W", "sub"], [" (workers)"]], { fill: C.text2, "font-size": 20 });
    richLabel(axes, 30, (Y0 + Y1) / 2, [["d"], ["sync", "sub"], [" por ronda (s)"]], { fill: C.text2, "font-size": 20, transform: `rotate(-90 30 ${(Y0 + Y1) / 2})` });

    function curve(f, color) {
      let d = "";
      for (let nw = 2; nw <= 16.001; nw += 0.25) d += `${d ? "L" : "M"}${sx(nw).toFixed(1)},${sy(f(nw)).toFixed(1)}`;
      return add(svg, "path", { d, fill: "none", stroke: color, "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round" });
    }

    function markers(f, color, name) {
      const g = group(svg);
      [2, 4, 6, 8, 12, 16].forEach((nw) => {
        const m = add(g, "circle", { cx: sx(nw), cy: sy(f(nw)), r: 6, fill: color, stroke: "#000", "stroke-width": 2 });
        const t = add(m, "title");
        t.textContent = `${name} · N_W = ${nw}: ${f(nw).toFixed(3).replace(".", ",")} s`;
      });
      return g;
    }

    const psLine = curve(psF, C.ps);
    const psMarks = markers(psF, C.ps, "PS, 1 servidor");
    const psLabel = label(svg, sx(16) - 10, sy(psF(16)) - 30, "PS · 1 servidor", { "text-anchor": "end", fill: C.text, "font-size": 21 });
    const arLine = curve(arF, C.ar);
    const arMarks = markers(arF, C.ar, "Ring All-Reduce");
    const asym = add(svg, "line", { x1: X0, y1: sy(2 * SR), x2: X1, y2: sy(2 * SR), stroke: C.muted, "stroke-width": 1.5, "stroke-dasharray": "4 6" });
    const asymLabel = label(svg, X1, sy(2 * SR) - 18, "2S/R = 0,2 s", { "text-anchor": "end", fill: C.muted, "font-size": 18 });
    const arLabel = label(svg, sx(16) - 10, sy(arF(16)) + 34, "Ring All-Reduce", { "text-anchor": "end", fill: C.text, "font-size": 21 });

    const leg = group(svg);
    add(leg, "line", { x1: X0, y1: 30, x2: X0 + 32, y2: 30, stroke: C.ps, "stroke-width": 3 });
    label(leg, X0 + 44, 31, "Parameter Server (1 servidor)", { "text-anchor": "start", fill: C.text2, "font-size": 19 });
    add(leg, "line", { x1: X0 + 420, y1: 30, x2: X0 + 452, y2: 30, stroke: C.ar, "stroke-width": 3 });
    label(leg, X0 + 464, 31, "Ring All-Reduce", { "text-anchor": "start", fill: C.text2, "font-size": 19 });
    const note = label(svg, X0, 72, "modelo analítico del informe · S/R ≈ 0,1 s (LeNet-5 en f16, 10 Mbps)", { "text-anchor": "start", fill: C.muted, "font-size": 17 });

    show(tl, [axes, leg, note], 0.05, 0.4);
    gsap.set([psMarks, psLabel, arMarks, arLabel, asym, asymLabel], { autoAlpha: 0 });
    gsap.set([psLine, arLine], { autoAlpha: 0 });
    tl.addLabel("s0");
    tl.set(psLine, { autoAlpha: 1 }, "s0");
    draw(tl, psLine, "s0", 1);
    tl.to(psMarks, { autoAlpha: 1, duration: 0.3 }, "s0+=0.8");
    tl.to(psLabel, { autoAlpha: 1, duration: 0.3 }, "s0+=0.9");
    tl.addLabel("s1");
    tl.set(arLine, { autoAlpha: 1 }, "s1");
    draw(tl, arLine, "s1", 1);
    tl.to(arMarks, { autoAlpha: 1, duration: 0.3 }, "s1+=0.8");
    tl.to([asym, asymLabel, arLabel], { autoAlpha: 1, duration: 0.3 }, "s1+=0.9");
    tl.addLabel("s2");
    return controller(tl, 2);
  }


  function sswindow(container) {
    const svg = svgRoot(container, 1100, 680);
    const tl = gsap.timeline();
    const E = 40;
    const W = 6;
    const loss = Array.from({ length: E }, (_, i) => {
      const e = i + 1;
      return 0.15 + 2.0 * Math.exp(-e / 4) + 0.0003 * Math.sin(e * 1.7);
    });
    const sOf = (k) => {
      let sum = 0;
      for (let i = k - W + 2; i <= k; i++) sum += Math.abs(loss[i] - loss[i - 1]) / loss[i - 1];
      return sum / (W - 1);
    };
    let trigger = E - 1;
    for (let k = W - 1; k < E; k++) if (sOf(k) <= 0.01) { trigger = k; break; }

    const X0 = 90, X1 = 1060, Y0 = 560, Y1 = 90;
    const yMax = 1.7;
    const sx = (i) => X0 + (i / (E - 1)) * (X1 - X0);
    const sy = (l) => Y0 - (Math.min(l, yMax) / yMax) * (Y0 - Y1);

    const axes = group(svg);
    add(axes, "line", { x1: X0, y1: Y0, x2: X1, y2: Y0, stroke: C.lineStrong, "stroke-width": 1.5 });
    add(axes, "line", { x1: X0, y1: Y0, x2: X0, y2: Y1, stroke: C.lineStrong, "stroke-width": 1.5 });
    label(axes, (X0 + X1) / 2, Y0 + 40, "rondas", { fill: C.muted, "font-size": 18 });
    label(axes, 50, (Y0 + Y1) / 2, "pérdida máx.", { fill: C.muted, "font-size": 18, transform: `rotate(-90 50 ${(Y0 + Y1) / 2})` });
    const badge = label(svg, X0, 40, "curva ilustrativa", { "text-anchor": "start", fill: C.warn, "font-size": 18 });

    const pathD = loss.map((l, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(l).toFixed(1)}`).join("");
    const curveEl = add(svg, "path", { d: pathD, fill: "none", stroke: C.text2, "stroke-width": 2.5 });
    const dots = group(svg);
    loss.forEach((l, i) => add(dots, "circle", { cx: sx(i), cy: sy(l), r: 4.5, fill: C.text2, stroke: "#000", "stroke-width": 1.5 }));

    const band = add(svg, "rect", { x: 0, y: Y1 - 10, width: sx(W - 1) - sx(0) + 24, height: Y0 - Y1 + 10, fill: C.accent, "fill-opacity": 0.08, stroke: C.accent, "stroke-width": 1.5, "stroke-opacity": 0.6 });
    const bandLabel = label(svg, 0, Y1 - 28, "ventana w = 6", { fill: C.accent, "font-size": 18 });
    const sTexts = [];
    for (let k = W - 1; k <= trigger; k++) {
      const val = (sOf(k) * 100).toFixed(1).replace(".", ",");
      const t = label(svg, X1 - 10, 150, `s = ${val} %`, { "text-anchor": "end", fill: k === trigger ? C.accent : C.text, "font-size": 30 });
      sTexts.push(t);
    }
    const sw = group(svg);
    add(sw, "line", { x1: sx(trigger) + 12, y1: Y1 - 20, x2: sx(trigger) + 12, y2: Y0, stroke: C.ss, "stroke-width": 2.5 });
    label(sw, sx(trigger) + 26, Y0 - 120, "switch", { "text-anchor": "start", fill: C.ss, "font-size": 22 });
    const phaseL = label(svg, (X0 + sx(trigger)) / 2, Y0 + 80, "All-Reduce · sincrónico", { fill: C.ar, "font-size": 21 });
    const phaseR = label(svg, (sx(trigger) + X1) / 2 + 10, Y0 + 80, "PS · asincrónico", { fill: C.ps, "font-size": 21 });
    const thresh = label(svg, X1 - 10, 190, "umbral: 1 %", { "text-anchor": "end", fill: C.muted, "font-size": 18 });

    show(tl, [axes, badge], 0.05, 0.3);
    draw(tl, curveEl, 0.15, 1.1);
    show(tl, dots, 0.9, 0.4);
    gsap.set([band, bandLabel, ...sTexts, sw, phaseL, phaseR, thresh], { autoAlpha: 0 });
    tl.addLabel("s0");

    const bx = (k) => sx(k - W + 1) - 12;
    gsap.set(band, { attr: { x: bx(W - 1) } });
    gsap.set(bandLabel, { attr: { x: bx(W - 1) + (sx(W - 1) - sx(0) + 24) / 2 } });
    tl.to([band, bandLabel, thresh], { autoAlpha: 1, duration: 0.3 }, "s0");
    tl.to(sTexts[0], { autoAlpha: 1, duration: 0.2 }, "s0");
    const stepDur = Math.min(0.22, 3 / sTexts.length);
    sTexts.forEach((t, j) => {
      if (j === 0) return;
      const at = `s0+=${0.3 + j * stepDur}`;
      const k = W - 1 + j;
      tl.to(band, { attr: { x: bx(k) }, duration: stepDur, ease: "none" }, at);
      tl.to(bandLabel, { attr: { x: bx(k) + (sx(W - 1) - sx(0) + 24) / 2 }, duration: stepDur, ease: "none" }, at);
      tl.set(sTexts[j - 1], { autoAlpha: 0 }, at);
      tl.set(t, { autoAlpha: 1 }, at);
    });
    tl.addLabel("s1");
    tl.to(sw, { autoAlpha: 1, duration: 0.35 }, "s1");
    tl.to([phaseL, phaseR], { autoAlpha: 1, duration: 0.35, stagger: 0.15 }, "s1+=0.2");
    tl.addLabel("s2");
    return controller(tl, 2);
  }


  function ssflow(container) {
    const svg = svgRoot(container, 1100, 780);
    const tl = gsap.timeline();
    const cx = 470, cy = 470, R = 175;
    const angles = [-90, 0, 90, 180];
    const ringPos = angles.map((a) => ({ x: cx + R * Math.cos(a * Math.PI / 180), y: cy + R * Math.sin(a * Math.PI / 180) }));
    const psPos = [{ x: 210, y: 300 }, { x: 210, y: 480 }, { x: 210, y: 660 }, { x: 830, y: 480 }];

    const orch = group(svg);
    add(orch, "rect", { x: 300, y: 40, width: 340, height: 100, rx: 6, fill: C.panel, stroke: C.accent, "stroke-width": 2.5 });
    label(orch, 470, 76, "orchestrator", { fill: C.text, "font-size": 25 });
    label(orch, 470, 110, "SwitchTracker", { fill: C.muted, "font-size": 18 });

    const win = group(svg);
    const slots = [];
    for (let k = 0; k < 6; k++) {
      add(win, "rect", { x: 690 + k * 44, y: 58, width: 36, height: 36, rx: 3, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
      slots.push(add(win, "rect", { x: 694 + k * 44, y: 62, width: 28, height: 28, rx: 2, fill: C.ctrl, opacity: 0 }));
    }
    label(win, 690, 122, "pérdida máx. por ronda · w = 6", { "text-anchor": "start", fill: C.muted, "font-size": 16 });
    const sVal = label(svg, 820, 160, "s ≤ 1 %  →  cambiar", { fill: C.accent, "font-size": 21 });

    const arcs = angles.map((a) => add(svg, "path", { d: arcPath(cx, cy, R, a + 22, a + 68), fill: "none", stroke: C.lineStrong, "stroke-width": 2.5, "marker-end": svg.ref("arrow-line") }));
    const ctrlLines = ringPos.map((p) => add(svg, "line", { x1: p.x, y1: p.y - 42, x2: 470, y2: 140, stroke: C.ctrl, "stroke-width": 1.5, "stroke-dasharray": "5 6", opacity: 0 }));

    const nodes = ringPos.map((p, i) => {
      const g = group(svg);
      gsap.set(g, { x: p.x, y: p.y });
      const circle = add(g, "circle", { cx: 0, cy: 0, r: 44, fill: C.panel, stroke: C.ar, "stroke-width": 2.5 });
      const name = label(g, 0, 1, `W${i}`, { fill: C.text, "font-size": 23 });
      return { g, circle, name };
    });
    const srvBox = add(nodes[3].g, "rect", { x: -90, y: -62, width: 180, height: 124, rx: 6, fill: C.panel, stroke: C.ps, "stroke-width": 2.5, opacity: 0 });
    const srvName = label(nodes[3].g, 0, -16, "S0", { fill: C.text, "font-size": 24, opacity: 0 });
    const srvSub = label(nodes[3].g, 0, 20, "θ de su tramo", { fill: C.psHi, "font-size": 16, opacity: 0 });
    nodes[3].g.appendChild(nodes[3].name);

    const upLosses = ringPos.map(() => ctrlPacket(svg));
    const downCmd = ringPos.map(() => ctrlPacket(svg, 15));
    const cmdLabels = ringPos.map((p, i) => label(svg, p.x + (i === 3 ? -70 : 70), p.y - 64, i === 3 ? "Upgrade" : "Switch", { fill: i === 3 ? C.psHi : C.ctrl, "font-size": 18 }));

    const psLinks = psPos.slice(0, 3).map((p) => add(svg, "line", { x1: p.x + 44, y1: p.y, x2: psPos[3].x - 90, y2: psPos[3].y, stroke: C.lineStrong, "stroke-width": 2 }));
    const phase1 = label(svg, 470, 740, "fase 1 · All-Reduce en anillo", { fill: C.ar, "font-size": 21 });
    const phase2 = label(svg, 520, 740, "fase 2 · Parameter Server, sin reiniciar el clúster", { fill: C.ps, "font-size": 21 });
    const noRestart = label(svg, 830, 600, "mismo proceso, nuevo rol", { fill: C.muted, "font-size": 17 });

    show(tl, [orch, ...arcs, ...nodes.map((nd) => nd.g), phase1], 0.05, 0.4, { stagger: 0.04 });
    gsap.set([win, sVal, ...cmdLabels, ...psLinks, phase2, noRestart], { autoAlpha: 0 });
    tl.addLabel("s0");

    tl.to(win, { autoAlpha: 1, duration: 0.3 }, "s0");
    tl.to(ctrlLines, { opacity: 0.7, duration: 0.3 }, "s0");
    for (let round = 0; round < 3; round++) {
      const at = tl.labels.s0 + 0.2 + round * 0.75;
      upLosses.forEach((pk, i) => {
        if (round === 0) gsap.set(pk, { autoAlpha: 0 });
        tl.set(pk, { autoAlpha: 1, x: ringPos[i].x, y: ringPos[i].y - 44 }, at);
        tl.to(pk, { x: 470, y: 142, duration: 0.55, ease: "power1.in" }, at);
        tl.set(pk, { autoAlpha: 0 }, at + 0.56);
      });
      [0, 1].forEach((j) => tl.to(slots[round * 2 + j], { attr: { opacity: 0.85 }, duration: 0.15 }, at + 0.58 + j * 0.08));
    }
    tl.addLabel("s1");

    tl.to(sVal, { autoAlpha: 1, duration: 0.3 }, "s1");
    downCmd.forEach((pk, i) => travel(tl, pk, { x: 470, y: 142 }, { x: ringPos[i].x, y: ringPos[i].y - 46 }, tl.labels.s1 + 0.35 + i * 0.06, 0.7));
    tl.to(cmdLabels, { autoAlpha: 1, duration: 0.3, stagger: 0.05 }, "s1+=0.9");
    tl.addLabel("s2");

    tl.to([...arcs, ...ctrlLines, ...cmdLabels, phase1], { autoAlpha: 0, duration: 0.35 }, "s2");
    tl.to(nodes[3].circle, { attr: { opacity: 0 }, duration: 0.3 }, "s2+=0.2");
    tl.to(nodes[3].name, { autoAlpha: 0, duration: 0.2 }, "s2+=0.2");
    tl.to([srvBox, srvName, srvSub], { attr: { opacity: 1 }, duration: 0.35 }, "s2+=0.35");
    nodes.forEach((nd, i) => tl.to(nd.g, { x: psPos[i].x, y: psPos[i].y, duration: 0.9, ease: "power2.inOut" }, "s2+=0.3"));
    tl.to(noRestart, { autoAlpha: 1, duration: 0.3 }, "s2+=1.1");
    tl.addLabel("s3");
    tl.to(psLinks, { autoAlpha: 1, duration: 0.4, stagger: 0.06 }, "s3");
    tl.to(phase2, { autoAlpha: 1, duration: 0.3 }, "s3+=0.2");
    tl.to(win, { opacity: 0.3, duration: 0.3 }, "s3");
    tl.addLabel("s4");

    const ambientFor = (step) => {
      if (step !== 4) return null;
      const a = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
      a.data = [];
      psPos.slice(0, 3).forEach((p, i) => {
        const g = gradPacket(svg, 8);
        const q = paramPacket(svg, 13);
        a.data.push(g, q);
        travel(a, g, { x: p.x + 44, y: p.y }, { x: psPos[3].x - 90, y: psPos[3].y }, i * 0.45, 1);
        travel(a, q, { x: psPos[3].x - 90, y: psPos[3].y }, { x: p.x + 44, y: p.y }, i * 0.45 + 1.15, 1);
      });
      return a;
    };
    return controller(tl, 4, ambientFor);
  }


  function compress(container) {
    const svg = svgRoot(container, 1100, 680);
    const tl = gsap.timeline();
    const g1 = [0.32, 0.9, 0.18, 0.55, 0.12, 0.7, 0.26, 0.44, 0.08, 0.95, 0.3, 0.2, 0.62, 0.14, 0.38, 0.5, 0.1, 0.78, 0.22, 0.34];
    const g2 = [0.3, 0.2, 0.36, 0.15, 0.3, 0.25, 0.34, 0.1, 0.12, 0.2, 0.32, 0.18, 0.1, 0.2, 0.3, 0.12, 0.08, 0.15, 0.36, 0.28];
    const thr = 0.6;
    const X0 = 70, Y0 = 470, H = 330, bw = 34, gap = 15;
    const sy = (m) => Y0 - m * H;

    const title = label(svg, X0, 40, "|g| por componente", { "text-anchor": "start", fill: C.muted, "font-size": 19 });
    const badge = label(svg, X0 + 260, 40, "ilustrativo · r = 0,25", { "text-anchor": "start", fill: C.warn, "font-size": 18 });
    add(svg, "line", { x1: X0 - 10, y1: Y0, x2: X0 + 20 * (bw + gap), y2: Y0, stroke: C.lineStrong, "stroke-width": 1.5 });

    const bars = g1.map((m, i) => add(svg, "rect", { x: X0 + i * (bw + gap), y: sy(m), width: bw, height: m * H, rx: 2, fill: C.muted, opacity: 0.8 }));
    const thrLine = group(svg);
    add(thrLine, "line", { x1: X0 - 20, y1: sy(thr), x2: X0 + 20 * (bw + gap), y2: sy(thr), stroke: C.accent, "stroke-width": 2, "stroke-dasharray": "8 6" });
    label(thrLine, X0 + 20 * (bw + gap) + 10, sy(thr) - 2, "umbral", { "text-anchor": "start", fill: C.accent, "font-size": 18 });
    const thrNote = label(svg, X0, 530, "umbral = percentil (1 − r) de |g|, estimado con una muestra ≤ 16384 y quickselect", { "text-anchor": "start", fill: C.text2, "font-size": 19 });

    const sentIdx = g1.map((m, i) => (m >= thr ? i : -1)).filter((i) => i >= 0);
    const restIdx = g1.map((m, i) => (m < thr ? i : -1)).filter((i) => i >= 0);
    const wire = label(svg, 990, 80, "→ red", { fill: C.grad, "font-size": 22 });
    const residualNote = label(svg, X0, 575, "lo que no se envía queda en el residual del worker", { "text-anchor": "start", fill: C.text2, "font-size": 19 });

    const newBars = g2.map((m, i) => add(svg, "rect", { x: X0 + i * (bw + gap), y: sy(g1[i] < thr ? g1[i] + m : m), width: bw, height: m * H, rx: 2, fill: C.param, opacity: 0.85 }));
    const crossIdx = g2.map((m, i) => (g1[i] < thr && g1[i] + m >= thr ? i : -1)).filter((i) => i >= 0);
    const crossNote = label(svg, X0, 620, "residual + gradiente nuevo: componentes chicos pero constantes cruzan el umbral", { "text-anchor": "start", fill: C.accent, "font-size": 19 });
    const legendG = group(svg);
    add(legendG, "rect", { x: X0, y: 70, width: 16, height: 16, fill: C.param });
    label(legendG, X0 + 24, 79, "gradiente de la ronda siguiente", { "text-anchor": "start", fill: C.muted, "font-size": 16 });

    show(tl, [title, badge], 0.05, 0.3);
    bars.forEach((b, i) => { gsap.set(b, { scaleY: 0, transformOrigin: "50% 100%" }); tl.to(b, { scaleY: 1, duration: 0.4, ease: "power2.out" }, 0.1 + i * 0.03); });
    gsap.set([thrLine, thrNote, wire, residualNote, ...newBars, crossNote, legendG], { autoAlpha: 0 });
    tl.addLabel("s0");
    tl.to([thrLine, thrNote], { autoAlpha: 1, duration: 0.35 }, "s0");
    tl.to(sentIdx.map((i) => bars[i]), { attr: { fill: C.grad }, opacity: 1, duration: 0.3 }, "s0+=0.3");
    tl.addLabel("s1");
    tl.to(sentIdx.map((i) => bars[i]), { y: -120, autoAlpha: 0, duration: 0.7, ease: "power2.in", stagger: 0.05 }, "s1");
    tl.to(wire, { autoAlpha: 1, duration: 0.3 }, "s1+=0.2");
    tl.to(restIdx.map((i) => bars[i]), { attr: { fill: C.text2 }, duration: 0.3 }, "s1+=0.6");
    tl.to(residualNote, { autoAlpha: 1, duration: 0.3 }, "s1+=0.6");
    tl.addLabel("s2");
    tl.to(wire, { autoAlpha: 0, duration: 0.2 }, "s2");
    tl.to(legendG, { autoAlpha: 1, duration: 0.3 }, "s2");
    newBars.forEach((b, i) => {
      gsap.set(b, { scaleY: 0, transformOrigin: "50% 100%" });
      tl.to(b, { autoAlpha: 1, scaleY: 1, duration: 0.35 }, `s2+=${0.1 + i * 0.03}`);
    });
    crossIdx.forEach((i) => {
      tl.to([bars[i], newBars[i]], { attr: { fill: C.grad }, duration: 0.3 }, "s2+=1");
    });
    tl.to(crossNote, { autoAlpha: 1, duration: 0.3 }, "s2+=1.1");
    tl.addLabel("s3");
    return controller(tl, 3);
  }


  function topo(container) {
    const svg = svgRoot(container, 1100, 640);
    const tl = gsap.timeline();
    const P = [{ x: 0, y: -170 }, { x: 170, y: -50 }, { x: 105, y: 150 }, { x: -105, y: 150 }, { x: -170, y: -50 }];
    const L = [
      [0, 2, 9, 7, 3],
      [2, 0, 3, 8, 6],
      [9, 3, 0, 2, 7],
      [7, 8, 2, 0, 3],
      [3, 6, 7, 3, 0],
    ];

    function permutations(arr) {
      if (arr.length <= 1) return [arr];
      return arr.flatMap((x, i) => permutations([...arr.slice(0, i), ...arr.slice(i + 1)]).map((p) => [x, ...p]));
    }
    let best = null;
    for (const perm of permutations([1, 2, 3, 4])) {
      const cyc = [0, ...perm];
      const cost = cyc.reduce((s, a, i) => s + L[a][cyc[(i + 1) % cyc.length]], 0);
      if (!best || cost < best.cost) best = { cyc, cost };
    }
    let center = 0;
    let centerCost = Infinity;
    for (let s = 0; s < 5; s++) {
      const worst = Math.max(...L[s].filter((_, j) => j !== s));
      if (worst < centerCost) { centerCost = worst; center = s; }
    }

    function graph(ox, oy, title) {
      const g = group(svg);
      label(g, ox, 40, title, { fill: C.text, "font-size": 23 });
      const edges = {};
      for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) {
        const pa = P[a], pb = P[b];
        const e = add(g, "line", { x1: ox + pa.x, y1: oy + pa.y, x2: ox + pb.x, y2: oy + pb.y, stroke: C.line, "stroke-width": 1.5 });
        const mx = ox + (pa.x + pb.x) / 2, my = oy + (pa.y + pb.y) / 2;
        const t = label(g, mx, my, String(L[a][b]), { fill: C.muted, "font-size": 15 });
        edges[`${a}-${b}`] = { e, t };
      }
      const nodes = P.map((p, i) => {
        const c = add(g, "circle", { cx: ox + p.x, cy: oy + p.y, r: 30, fill: C.panel, stroke: C.muted, "stroke-width": 2 });
        label(g, ox + p.x, oy + p.y + 1, `n${i}`, { fill: C.text2, "font-size": 19 });
        return c;
      });
      return { g, edges, nodes };
    }

    const left = graph(270, 330, "All-Reduce: anillo");
    const right = graph(830, 330, "Parameter Server: servidor");
    const edgeOf = (gph, a, b) => gph.edges[a < b ? `${a}-${b}` : `${b}-${a}`];
    const ringEdges = best.cyc.map((a, i) => edgeOf(left, a, best.cyc[(i + 1) % best.cyc.length]));
    const ringNote = label(svg, 270, 580, `viajante exacto (Held–Karp) · costo ${best.cost}`, { fill: C.accent, "font-size": 18 });
    const centerEdges = [0, 1, 2, 3, 4].filter((j) => j !== center).map((j) => edgeOf(right, center, j));
    const psNote = label(svg, 830, 580, `minimiza la mayor latencia worker–servidor (${centerCost})`, { fill: C.psHi, "font-size": 18 });
    const caption = label(svg, 550, 625, "latencias ilustrativas (ms) · en O.N.O.: 10 pings por par, se toma el mayor RTT", { fill: C.muted, "font-size": 16 });

    show(tl, [left.g, right.g, caption], 0.05, 0.4);
    gsap.set([ringNote, psNote], { autoAlpha: 0 });
    tl.addLabel("s0");
    ringEdges.forEach(({ e, t }, i) => {
      tl.to(e, { attr: { stroke: C.ar, "stroke-width": 4 }, duration: 0.25 }, `s0+=${i * 0.12}`);
      tl.to(t, { attr: { fill: C.text } , duration: 0.25 }, `s0+=${i * 0.12}`);
    });
    tl.to(left.nodes, { attr: { stroke: C.ar }, duration: 0.3 }, "s0+=0.4");
    tl.to(ringNote, { autoAlpha: 1, duration: 0.3 }, "s0+=0.6");
    tl.addLabel("s1");
    tl.to(right.nodes[center], { attr: { stroke: C.ps, fill: "rgba(61,139,230,0.2)" }, duration: 0.3 }, "s1");
    centerEdges.forEach(({ e, t }, i) => {
      tl.to(e, { attr: { stroke: C.ps, "stroke-width": 3.5 }, duration: 0.25 }, `s1+=${0.15 + i * 0.08}`);
      tl.to(t, { attr: { fill: C.text }, duration: 0.25 }, `s1+=${0.15 + i * 0.08}`);
    });
    tl.to(right.nodes.filter((_, i) => i !== center), { attr: { stroke: C.ar }, duration: 0.3 }, "s1+=0.4");
    tl.to(psNote, { autoAlpha: 1, duration: 0.3 }, "s1+=0.6");
    tl.addLabel("s2");
    return controller(tl, 2);
  }

  const BUILDERS = { machines, stack, dataparallel, syncasync, arch, flatbuffer, ps, ring, dsync, sswindow, ssflow, compress, topo };

  window.ONO_DIAGRAMS = {
    build(container) {
      const fn = BUILDERS[container.dataset.diagram];
      if (!fn) return null;
      const ctrl = fn(container);
      const btn = document.createElement("button");
      btn.className = "replay";
      btn.type = "button";
      btn.textContent = "↻ repetir";
      btn.setAttribute("aria-label", "Repetir la animación de la etapa actual");
      btn.addEventListener("click", (e) => { e.stopPropagation(); ctrl.replay(); });
      container.appendChild(btn);
      return ctrl;
    },
  };
})();
