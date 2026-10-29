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
    grad: v("--grad"), param: v("--param"), ctrl: v("--ctrl"), warn: v("--warn"), bad: v("--bad"),
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
    const idleNote = label(svg, x0 + t * unit, syncY + 1.5 * rowH, "espera", { fill: C.muted, "font-size": 19, "text-anchor": "start" });

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

  function starPath(cx, cy, R, r) {
    let d = "";
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      const rr = i % 2 ? r : R;
      d += `${i ? "L" : "M"}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`;
    }
    return `${d}Z`;
  }

  function drawLinear(tl, el, at, dur) {
    const len = lineLen(el);
    gsap.set(el, { attr: { "stroke-dasharray": len, "stroke-dashoffset": len } });
    tl.to(el, { attr: { "stroke-dashoffset": 0 }, duration: dur, ease: "none" }, at);
  }

  function pathOf(pts) {
    return pts.map((p, k) => `${k ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join("");
  }

  function alongPath(tl, el, pts, at, dur) {
    const seg = pts.slice(1).map((p, k) => Math.hypot(p.x - pts[k].x, p.y - pts[k].y));
    const total = seg.reduce((s, l) => s + l, 0) || 1;
    tl.to(el, { keyframes: pts.slice(1).map((p, k) => ({ x: p.x, y: p.y, duration: (dur * seg[k]) / total, ease: "none" })) }, at);
  }


  function cost(container) {
    const svg = svgRoot(container, 1100, 680);
    const tl = gsap.timeline();

    const cards = Array.from({ length: 12 }, (_, i) => add(svg, "rect", { x: 60, y: 548 - i * 36, width: 170, height: 26, rx: 3, fill: "#151105", stroke: C.grad, "stroke-width": 1.5 }));
    label(svg, 145, 616, "datos", { fill: C.text2 });

    function net(sizes, x0, x1, y0, y1) {
      const g = group(svg);
      const edges = group(g);
      const cols = sizes.map((n, li) => {
        const x = x0 + (li * (x1 - x0)) / (sizes.length - 1);
        const gap = Math.min(64, (y1 - y0) / n);
        return Array.from({ length: n }, (_, k) => ({ x, y: (y0 + y1) / 2 + (k - (n - 1) / 2) * gap }));
      });
      const lines = [];
      cols.slice(0, -1).forEach((col, li) => col.forEach((a) => cols[li + 1].forEach((b) => {
        lines.push(add(edges, "line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: C.lineStrong, "stroke-width": 1 }));
      })));
      const nodes = cols.flat().map((p) => add(g, "circle", { cx: p.x, cy: p.y, r: 11, fill: C.panel, stroke: C.accent, "stroke-width": 2 }));
      return { g, lines, nodes };
    }
    const small = net([3, 4, 2], 420, 640, 220, 470);
    const big = net([5, 8, 8, 8, 3], 320, 740, 110, 560);
    label(svg, 530, 616, "parámetros", { fill: C.text2 });

    const CX = 950;
    const CY = 160;
    const clock = group(svg);
    add(clock, "circle", { cx: CX, cy: CY, r: 74, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2 });
    for (let k = 0; k < 12; k++) {
      const a = (k * 30 * Math.PI) / 180;
      add(clock, "line", { x1: CX + 62 * Math.cos(a), y1: CY + 62 * Math.sin(a), x2: CX + 70 * Math.cos(a), y2: CY + 70 * Math.sin(a), stroke: C.muted, "stroke-width": 2 });
    }
    const hand = add(clock, "line", { x1: CX, y1: CY, x2: CX, y2: CY - 54, stroke: C.accent, "stroke-width": 3.5, "stroke-linecap": "round" });
    add(clock, "circle", { cx: CX, cy: CY, r: 5, fill: C.accent });
    const days = ["día 1", "día 2", "día 3"].map((t) => label(svg, CX, CY + 104, t, { fill: C.warn, "font-size": 24 }));

    const barG = group(svg);
    label(barG, 840, 318, "entrenamiento", { "text-anchor": "start", fill: C.muted, "font-size": 18 });
    add(barG, "rect", { x: 840, y: 334, width: 220, height: 24, rx: 3, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
    const barFill = add(barG, "rect", { x: 843, y: 337, width: 214, height: 18, rx: 2, fill: C.accent });

    const memG = group(svg);
    label(memG, 840, 438, "memoria", { "text-anchor": "start", fill: C.muted, "font-size": 18 });
    add(memG, "rect", { x: 840, y: 454, width: 220, height: 24, rx: 3, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
    const memFill = add(memG, "rect", { x: 843, y: 457, width: 214, height: 18, rx: 2, fill: C.param });
    const memOver = add(memG, "rect", { x: 1057, y: 457, width: 40, height: 18, rx: 2, fill: C.bad });
    const memNo = label(memG, 950, 514, "no entra", { fill: C.bad, "font-size": 24 });

    gsap.set(cards.slice(3), { autoAlpha: 0 });
    gsap.set([big.g, ...days, memNo], { autoAlpha: 0 });
    gsap.set(memOver, { autoAlpha: 0, scaleX: 0, transformOrigin: "0% 50%" });
    gsap.set(barFill, { scaleX: 0, transformOrigin: "0% 50%" });
    gsap.set(memFill, { scaleX: 0.25, transformOrigin: "0% 50%" });
    gsap.set(hand, { svgOrigin: `${CX} ${CY}` });
    gsap.set(big.nodes, { scale: 0, transformOrigin: "50% 50%" });
    gsap.set(big.lines, { autoAlpha: 0 });

    show(tl, cards.slice(0, 3), 0.05, 0.3, { stagger: 0.06 });
    show(tl, small.g, 0.15, 0.4);
    show(tl, [clock, barG, memG], 0.25, 0.4);
    tl.to(barFill, { scaleX: 1, duration: 0.8, ease: "power1.inOut" }, 0.5);
    tl.to(hand, { rotation: 60, duration: 0.8, ease: "power1.inOut" }, 0.5);
    tl.addLabel("s0");

    tl.to(cards.slice(3), { autoAlpha: 1, duration: 0.25, stagger: 0.06 }, "s0");
    tl.to(small.g, { autoAlpha: 0, duration: 0.3 }, "s0");
    tl.set(big.g, { autoAlpha: 1 }, "s0+=0.25");
    tl.to(big.nodes, { scale: 1, duration: 0.3, stagger: 0.012, ease: "back.out(2)" }, "s0+=0.25");
    tl.to(big.lines, { autoAlpha: 1, duration: 0.3, stagger: 0.003 }, "s0+=0.5");
    tl.to(memFill, { scaleX: 0.7, duration: 0.6 }, "s0+=0.4");
    tl.addLabel("s1");

    tl.set(barFill, { scaleX: 0 }, "s1+=0.05");
    tl.to(barFill, { scaleX: 0.16, duration: 2.6, ease: "none" }, "s1+=0.1");
    tl.to(hand, { rotation: 60 + 360 * 3, duration: 2.6, ease: "none" }, "s1+=0.1");
    days.forEach((d, i) => {
      const at = `s1+=${0.1 + i * 0.87}`;
      tl.set(d, { autoAlpha: 1 }, at);
      if (i > 0) tl.set(days[i - 1], { autoAlpha: 0 }, at);
    });
    tl.addLabel("s2");

    tl.to(memFill, { scaleX: 1, duration: 0.6, ease: "power2.in" }, "s2");
    tl.to(memOver, { autoAlpha: 1, scaleX: 1, duration: 0.3 }, "s2+=0.6");
    tl.to(memG, { x: 6, duration: 0.05, repeat: 5, yoyo: true }, "s2+=0.8");
    tl.to(memNo, { autoAlpha: 1, duration: 0.3 }, "s2+=0.9");
    tl.addLabel("s3");
    return controller(tl, 3);
  }


  function scaleup(container) {
    const svg = svgRoot(container, 1100, 680);
    const tl = gsap.timeline();
    const base = 600;
    const cx = 420;
    const w = 280;
    const ceilY = 120;

    const floor = add(svg, "line", { x1: 120, y1: base, x2: 980, y2: base, stroke: C.lineStrong, "stroke-width": 2 });
    const ceil = group(svg);
    const ceilLine = add(ceil, "line", { x1: 120, y1: ceilY, x2: 980, y2: ceilY, stroke: C.muted, "stroke-width": 2.5, "stroke-dasharray": "12 9" });
    const ceilText = label(ceil, 980, ceilY - 26, "límite de un solo equipo", { "text-anchor": "end", fill: C.muted, "font-size": 20 });

    const mach = group(svg);
    const box = add(mach, "rect", { x: cx - w / 2, y: base - 170, width: w, height: 170, rx: 6, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2.5 });
    const slots = Array.from({ length: 9 }, (_, i) => add(mach, "rect", { x: cx - w / 2 + 28, y: base - 52 - i * 48, width: w - 56, height: 26, rx: 3, fill: "rgba(10,248,134,0.10)", stroke: C.accent, "stroke-width": 1.5 }));

    const tag = group(svg);
    gsap.set(tag, { x: 650, y: 360 });
    add(tag, "path", { d: "M0,-44 L160,-44 L204,0 L160,44 L0,44 Z", fill: "#151105", stroke: C.grad, "stroke-width": 2.5 });
    add(tag, "circle", { cx: 166, cy: 0, r: 8, fill: "none", stroke: C.grad, "stroke-width": 2 });
    const prices = ["$", "$$", "$$$", "$$$$"].map((t) => label(tag, 76, 3, t, { fill: C.grad, "font-size": 38, "font-weight": 700 }));

    gsap.set([...slots.slice(3), ceil, tag, ...prices], { autoAlpha: 0 });
    show(tl, [floor, mach], 0.05, 0.4);
    tl.to(slots.slice(0, 3), { autoAlpha: 1, duration: 0.3, stagger: 0.08 }, 0.15);
    tl.to([tag, prices[0]], { autoAlpha: 1, duration: 0.35 }, 0.45);
    tl.addLabel("s0");

    tl.to(box, { attr: { y: base - 470, height: 470 }, duration: 1.8, ease: "power2.inOut" }, "s0");
    slots.slice(3).forEach((s, i) => tl.to(s, { autoAlpha: 1, duration: 0.2 }, `s0+=${0.35 + i * 0.22}`));
    prices.slice(1).forEach((p, i) => {
      const at = `s0+=${0.45 + i * 0.5}`;
      tl.set(prices[i], { autoAlpha: 0 }, at);
      tl.set(p, { autoAlpha: 1 }, at);
      tl.fromTo(tag, { scale: 1 }, { scale: 1.12 + i * 0.05, duration: 0.2, yoyo: true, repeat: 1, transformOrigin: "0% 50%" }, at);
    });
    tl.addLabel("s1");

    tl.to(ceil, { autoAlpha: 1, duration: 0.3 }, "s1");
    tl.to(box, { attr: { y: base - 492, height: 492 }, duration: 0.25, ease: "power2.in" }, "s1+=0.35");
    tl.to(box, { attr: { y: base - 480, height: 480 }, duration: 0.35, ease: "bounce.out" }, ">");
    tl.to(ceilLine, { attr: { stroke: C.bad }, duration: 0.15 }, "s1+=0.6");
    tl.to(ceilText, { attr: { fill: C.bad }, duration: 0.15 }, "s1+=0.6");
    tl.to(mach, { x: 5, duration: 0.05, repeat: 5, yoyo: true }, "s1+=0.6");
    tl.addLabel("s2");
    return controller(tl, 2);
  }


  function scaleout(container) {
    const svg = svgRoot(container, 1100, 680);
    const tl = gsap.timeline();
    const SW = { x: 550, y: 345 };
    const pos = [[190, 150], [420, 150], [680, 150], [910, 150], [190, 540], [420, 540], [680, 540], [910, 540]].map(([x, y]) => ({ x, y }));
    const ends = pos.map((p) => {
      const up = p.y < SW.y;
      return { from: { x: p.x, y: p.y + (up ? 50 : -50) }, to: { x: SW.x + (p.x - SW.x) * 0.25, y: SW.y + (up ? -28 : 28) } };
    });

    const links = ends.map((e) => add(svg, "line", { x1: e.from.x, y1: e.from.y, x2: e.to.x, y2: e.to.y, stroke: C.accent, "stroke-width": 2 }));
    const sw = group(svg);
    add(sw, "rect", { x: SW.x - 110, y: SW.y - 28, width: 220, height: 56, rx: 5, fill: C.panel, stroke: C.accent, "stroke-width": 2 });
    label(sw, SW.x, SW.y + 1, "red", { fill: C.accent, "font-size": 23 });

    const machines = pos.map(() => {
      const g = group(svg);
      add(g, "rect", { x: -70, y: -50, width: 140, height: 100, rx: 5, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2 });
      add(g, "rect", { x: -48, y: -28, width: 96, height: 16, rx: 2, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
      add(g, "rect", { x: -48, y: 0, width: 96, height: 16, rx: 2, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
      gsap.set(g, { x: SW.x, y: SW.y, autoAlpha: 0 });
      return g;
    });
    const one = label(svg, SW.x, SW.y + 92, "una máquina común", { fill: C.text2, "font-size": 23 });
    const sum = label(svg, SW.x, 644, "Σ cómputo  ·  Σ memoria", { fill: C.accent, "font-size": 27 });

    gsap.set([...links, sw, sum], { autoAlpha: 0 });
    tl.to(machines[0], { autoAlpha: 1, duration: 0.4 }, 0.05);
    show(tl, one, 0.2, 0.4);
    tl.addLabel("s0");

    tl.to(one, { autoAlpha: 0, duration: 0.2 }, "s0");
    machines.forEach((m, i) => {
      tl.to(m, { autoAlpha: 1, duration: 0.15 }, `s0+=${0.1 + i * 0.07}`);
      tl.to(m, { x: pos[i].x, y: pos[i].y, duration: 0.7, ease: "back.out(1.3)" }, `s0+=${0.1 + i * 0.07}`);
    });
    tl.addLabel("s1");

    tl.to(sw, { autoAlpha: 1, duration: 0.3 }, "s1");
    links.forEach((l, i) => {
      tl.set(l, { autoAlpha: 1 }, `s1+=${0.1 + i * 0.05}`);
      draw(tl, l, tl.labels.s1 + 0.1 + i * 0.05, 0.4);
    });
    tl.to(sum, { autoAlpha: 1, duration: 0.4 }, "s1+=0.6");
    tl.addLabel("s2");

    const ambientFor = (step) => {
      if (step !== 2) return null;
      const a = gsap.timeline({ repeat: -1, repeatDelay: 0.4 });
      a.data = [];
      ends.forEach((e, i) => {
        const g = gradPacket(svg, 7);
        const p = paramPacket(svg, 12);
        a.data.push(g, p);
        travel(a, g, e.from, e.to, (i % 4) * 0.22 + (i >= 4 ? 0.1 : 0), 0.8);
        travel(a, p, e.to, e.from, 1.4 + (i % 4) * 0.22, 0.8);
      });
      return a;
    };
    return controller(tl, 2, ambientFor);
  }


  function parallelism(container) {
    const svg = svgRoot(container, 1100, 700);
    const tl = gsap.timeline();
    const MC = WORKER.slice(0, 3);
    const lx = [120, 275, 430];
    const rx = [705, 850, 995];
    const MY = 460;

    function panel(x, title) {
      const g = group(svg);
      const frame = add(g, "rect", { x, y: 10, width: 520, height: 680, rx: 8, fill: "none", stroke: C.line, "stroke-width": 2 });
      label(g, x + 260, 52, title, { fill: C.text, "font-size": 26 });
      return { g, frame };
    }
    const left = panel(15, "paralelismo de modelo");
    const right = panel(565, "paralelismo de datos");

    function machine(x, i) {
      const g = group(svg);
      add(g, "rect", { x: x - 60, y: MY, width: 120, height: 150, rx: 6, fill: C.panel, stroke: MC[i], "stroke-width": 2.2 });
      label(g, x, MY + 172, `M${i}`, { fill: MC[i], "font-size": 20 });
      return g;
    }

    const layerEdges = [0, 1].map((i) => add(svg, "line", { x1: lx[i] + 34, y1: 190, x2: lx[i + 1] - 34, y2: 190, stroke: C.accent, "stroke-width": 2, "marker-end": svg.ref("arrow-accent") }));
    const layers = lx.map((x) => {
      const g = group(svg);
      add(g, "rect", { x: x - 32, y: 105, width: 64, height: 170, rx: 5, fill: "rgba(10,248,134,0.08)", stroke: C.accent, "stroke-width": 1.8 });
      [0, 1, 2, 3].forEach((k) => add(g, "circle", { cx: x, cy: 130 + k * 40, r: 10, fill: C.panel, stroke: C.accent, "stroke-width": 1.8 }));
      return g;
    });
    const netNote = label(svg, 275, 312, "una red de tres capas", { fill: C.muted, "font-size": 18 });
    const leftMachines = lx.map((x, i) => machine(x, i));
    const leftArrows = [0, 1].map((i) => add(svg, "line", { x1: lx[i] + 62, y1: MY + 75, x2: lx[i + 1] - 64, y2: MY + 75, stroke: C.lineStrong, "stroke-width": 2.5, "marker-end": svg.ref("arrow-line") }));
    const leftCap = label(svg, 275, 330, "se parte la red", { fill: C.text, "font-size": 23 });

    const shards = rx.map((x, i) => {
      const g = group(svg);
      const r = add(g, "rect", { x: 640 + i * 140, y: 120, width: 132, height: 46, rx: 3, fill: "#151105", stroke: C.grad, "stroke-width": 1.8 });
      label(g, 640 + i * 140 + 66, 144, `datos ${i}`, { fill: C.text2, "font-size": 17 });
      return { g, r, dx: x - (640 + i * 140 + 66) };
    });
    const dsNote = label(svg, 850, 200, "un dataset", { fill: C.muted, "font-size": 18 });
    const rightMachines = rx.map((x, i) => machine(x, i));
    const replicas = rx.map((x) => {
      const g = group(svg);
      [-30, 0, 30].forEach((dx) => add(g, "rect", { x: x + dx - 11, y: MY + 22, width: 22, height: 82, rx: 3, fill: "rgba(10,248,134,0.08)", stroke: C.accent, "stroke-width": 1.6 }));
      label(g, x, MY + 128, "modelo", { fill: C.muted, "font-size": 14 });
      return g;
    });
    const rightCap = label(svg, 850, 330, "se parte el dataset", { fill: C.text, "font-size": 23 });
    const badge = group(svg);
    add(badge, "rect", { x: 760, y: 150, width: 180, height: 64, rx: 6, fill: "rgba(10,248,134,0.14)", stroke: C.accent, "stroke-width": 2.5 });
    label(badge, 850, 183, "O.N.O.", { fill: C.accent, "font-size": 30 });

    show(tl, [left.g, right.g], 0.05, 0.35);
    show(tl, [...layers, ...layerEdges, netNote], 0.2, 0.4, { stagger: 0.05 });
    show(tl, [...shards.map((s) => s.g), dsNote], 0.3, 0.4, { stagger: 0.05 });
    show(tl, [...leftMachines, ...rightMachines], 0.45, 0.4, { stagger: 0.04 });
    gsap.set([...leftArrows, leftCap, ...replicas, rightCap, badge], { autoAlpha: 0 });
    tl.addLabel("s0");

    tl.to([...layerEdges, netNote], { autoAlpha: 0, duration: 0.25 }, "s0");
    layers.forEach((l, i) => tl.to(l, { y: MY + 75 - 190, scale: 0.6, transformOrigin: "50% 50%", duration: 0.8, ease: "power2.inOut" }, `s0+=${0.15 + i * 0.12}`));
    tl.to(leftArrows, { autoAlpha: 1, duration: 0.3, stagger: 0.1 }, "s0+=1");
    tl.to(leftCap, { autoAlpha: 1, duration: 0.3 }, "s0+=1.1");
    tl.addLabel("s1");

    tl.to(dsNote, { autoAlpha: 0, duration: 0.2 }, "s1");
    shards.forEach((s, i) => {
      tl.to(s.r, { attr: { stroke: MC[i] }, duration: 0.25 }, `s1+=${i * 0.08}`);
      tl.to(s.g, { x: s.dx, y: MY - 60 - 120, duration: 0.8, ease: "power2.inOut" }, `s1+=${0.2 + i * 0.1}`);
    });
    tl.to(replicas, { autoAlpha: 1, duration: 0.35, stagger: 0.1 }, "s1+=0.8");
    tl.to(rightCap, { autoAlpha: 1, duration: 0.3 }, "s1+=1.1");
    tl.addLabel("s2");

    tl.to(right.frame, { attr: { stroke: C.accent }, duration: 0.35 }, "s2");
    tl.to([left.g, ...layers, ...leftMachines, ...leftArrows, leftCap], { opacity: 0.35, duration: 0.4 }, "s2");
    tl.fromTo(badge, { autoAlpha: 0, scale: 0.6, transformOrigin: "50% 50%" }, { autoAlpha: 1, scale: 1, duration: 0.4, ease: "back.out(2)" }, "s2+=0.2");
    tl.addLabel("s3");

    const ambientFor = (step) => {
      if (step < 1) return null;
      const a = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
      a.data = [];
      const glow = (x, i, at) => {
        const r = add(svg, "rect", { x: x - 60, y: MY, width: 120, height: 150, rx: 6, fill: MC[i], opacity: 0 });
        a.data.push(r);
        a.to(r, { opacity: step === 3 && x < 550 ? 0.1 : 0.3, duration: 0.3, yoyo: true, repeat: 1 }, at);
      };
      lx.forEach((x, i) => {
        glow(x, i, i * 0.95);
        if (i < 2) {
          const pk = gradPacket(svg, 8);
          a.data.push(pk);
          travel(a, pk, { x: x + 60, y: MY + 75 }, { x: lx[i + 1] - 60, y: MY + 75 }, i * 0.95 + 0.55, 0.4);
        }
      });
      if (step >= 2) rx.forEach((x, i) => glow(x, i, 0));
      return a;
    };
    return controller(tl, 3, ambientFor);
  }


  function drift(container) {
    const svg = svgRoot(container, 1100, 700);
    const tl = gsap.timeline();
    const M = [{ x: 300, y: 500 }, { x: 800, y: 440 }, { x: 560, y: 200 }];
    const G = { x: (M[0].x + M[1].x + M[2].x) / 3, y: (M[0].y + M[1].y + M[2].y) / 3 };
    const start = { x: 120, y: 100 };
    const ka = 0.2;
    const kb = 0.32;
    const K = 5;
    const ROUNDS = 4;

    const rounds = [];
    let shared = { ...start };
    for (let r = 0; r < ROUNDS; r++) {
      const paths = M.map((m) => {
        const pts = [{ ...shared }];
        let p = { ...shared };
        for (let k = 0; k < K; k++) {
          p = { x: p.x - ka * (p.x - m.x), y: p.y - kb * (p.y - m.y) };
          pts.push(p);
        }
        return pts;
      });
      const avg = { x: paths.reduce((s, p) => s + p[K].x, 0) / 3, y: paths.reduce((s, p) => s + p[K].y, 0) / 3 };
      rounds.push({ paths, avg });
      shared = avg;
    }

    const contours = group(svg);
    const ratio = Math.sqrt(kb / ka);
    [45, 90, 135, 180, 225, 270, 315].forEach((r, i) => add(contours, "ellipse", { cx: G.x, cy: G.y, rx: r * ratio, ry: r, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5, opacity: 1 - i * 0.12 }));
    const badge = label(svg, 20, 30, "ilustrativo", { "text-anchor": "start", fill: C.warn, "font-size": 18 });
    const star = group(svg);
    add(star, "path", { d: starPath(G.x, G.y, 17, 7), fill: C.accent });
    label(star, G.x + 28, G.y - 24, "mínimo global", { "text-anchor": "start", fill: C.accent, "font-size": 19 });

    const minima = M.map((m, i) => {
      const g = group(svg);
      add(g, "path", { d: `M${m.x - 11},${m.y - 11} L${m.x + 11},${m.y + 11} M${m.x + 11},${m.y - 11} L${m.x - 11},${m.y + 11}`, stroke: WORKER[i], "stroke-width": 3.5 });
      label(g, m.x, m.y + (i === 2 ? -30 : 32), `mínimo de los datos ${i}`, { fill: WORKER[i], "font-size": 17 });
      return g;
    });

    const trails = rounds.map((rd, r) => rd.paths.map((pts, i) => add(svg, "path", { d: pathOf(pts), fill: "none", stroke: WORKER[i], "stroke-width": r === 0 ? 3 : 2, "stroke-linecap": "round", "stroke-linejoin": "round" })));
    const syncs = rounds.map((rd) => rd.paths.map((pts) => add(svg, "line", { x1: pts[K].x, y1: pts[K].y, x2: rd.avg.x, y2: rd.avg.y, stroke: C.ctrl, "stroke-width": 1.8, "stroke-dasharray": "5 6" })));
    const ends0 = rounds[0].paths.map((p) => p[K]);
    const avg0 = rounds[0].avg;
    const tri = add(svg, "path", { d: `${pathOf(ends0)}Z`, fill: "rgba(255,209,102,0.08)", stroke: C.warn, "stroke-width": 2, "stroke-dasharray": "8 6" });
    const triLabel = label(svg, avg0.x, avg0.y + 6, "client drift", { fill: C.warn, "font-size": 30 });

    const workers = WORKER.slice(0, 3).map((c) => {
      const g = group(svg);
      add(g, "circle", { r: 13, fill: c, stroke: "#000", "stroke-width": 2.5 });
      gsap.set(g, { x: start.x, y: start.y });
      return g;
    });
    const startLabel = label(svg, start.x + 34, start.y - 6, "misma réplica en los tres", { "text-anchor": "start", fill: C.text2, "font-size": 20 });
    const avgLabel = label(svg, avg0.x, avg0.y - 34, "promedio", { fill: C.ctrl, "font-size": 21 });

    function runLocal(r, at, dur) {
      rounds[r].paths.forEach((pts, i) => {
        tl.set(trails[r][i], { autoAlpha: 1 }, at);
        drawLinear(tl, trails[r][i], at, dur);
        alongPath(tl, workers[i], pts, at, dur);
      });
    }

    show(tl, [contours, star, badge], 0.05, 0.4);
    show(tl, [...workers, startLabel], 0.3, 0.4);
    gsap.set([...minima, ...trails.flat(), ...syncs.flat(), tri, triLabel, avgLabel], { autoAlpha: 0 });
    gsap.set(tri, { svgOrigin: `${avg0.x} ${avg0.y}` });
    gsap.set(star, { svgOrigin: `${G.x} ${G.y}` });
    tl.addLabel("s0");

    tl.to(startLabel, { autoAlpha: 0, duration: 0.2 }, "s0");
    tl.to(minima, { autoAlpha: 1, duration: 0.3, stagger: 0.12 }, "s0");
    runLocal(0, tl.labels.s0 + 0.5, 1.8);
    tl.to([tri, triLabel], { autoAlpha: 1, duration: 0.35 }, tl.labels.s0 + 2.3);
    tl.addLabel("s1");

    tl.to(triLabel, { autoAlpha: 0, duration: 0.2 }, "s1");
    tl.to(syncs[0], { autoAlpha: 1, duration: 0.3 }, "s1");
    tl.to(tri, { scale: 0, duration: 0.7, ease: "power2.in" }, "s1+=0.35");
    workers.forEach((wk) => tl.to(wk, { x: avg0.x, y: avg0.y, duration: 0.7, ease: "power2.in" }, "s1+=0.35"));
    tl.to(avgLabel, { autoAlpha: 1, duration: 0.3 }, "s1+=1");
    tl.addLabel("s2");

    tl.to([avgLabel, ...syncs[0]], { autoAlpha: 0, duration: 0.2 }, "s2");
    tl.to(trails[0], { opacity: 0.25, duration: 0.3 }, "s2");
    let at = tl.labels.s2 + 0.3;
    for (let r = 1; r < ROUNDS; r++) {
      runLocal(r, at, 0.75);
      at += 0.8;
      tl.to(syncs[r], { autoAlpha: 1, duration: 0.15 }, at);
      workers.forEach((wk) => tl.to(wk, { x: rounds[r].avg.x, y: rounds[r].avg.y, duration: 0.35, ease: "power2.in" }, at + 0.1));
      tl.to(syncs[r], { autoAlpha: 0, duration: 0.2 }, at + 0.5);
      tl.to(trails[r], { opacity: 0.25, duration: 0.2 }, at + 0.5);
      at += 0.6;
    }
    tl.fromTo(star, { scale: 1 }, { scale: 1.4, duration: 0.3, yoyo: true, repeat: 1 }, at);
    tl.addLabel("s3");
    return controller(tl, 3);
  }


  function alternatives(container) {
    const svg = svgRoot(container, 1680, 620);
    const tl = gsap.timeline();
    const specs = [
      { name: "Parameter Server", kw: "servidor central", color: C.ps },
      { name: "All-Reduce", kw: "anillo, sin centro", color: C.ar },
      { name: "PS asincrónico", kw: "nadie espera", color: C.psHi },
      { name: "Strategy-Switch", kw: "AR → PS asincrónico", color: C.ss },
    ];

    function starTopo(g, cx, cy, color, s, dashed) {
      const S = { x: cx, y: cy - 95 * s };
      const W = [-115, 0, 115].map((dx) => ({ x: cx + dx * s, y: cy + 85 * s }));
      const r = 26 * s;
      W.forEach((w) => add(g, "line", { x1: w.x, y1: w.y - r, x2: S.x, y2: S.y + 28 * s, stroke: C.lineStrong, "stroke-width": 2, ...(dashed ? { "stroke-dasharray": "6 6" } : {}) }));
      add(g, "rect", { x: S.x - 40 * s, y: S.y - 28 * s, width: 80 * s, height: 56 * s, rx: 4, fill: C.panel, stroke: color, "stroke-width": 2.5 });
      if (s === 1) label(g, S.x, S.y + 1, "S", { fill: C.text, "font-size": 22 });
      W.forEach((w, k) => {
        add(g, "circle", { cx: w.x, cy: w.y, r, fill: C.panel, stroke: C.ar, "stroke-width": 2 });
        if (s === 1) label(g, w.x, w.y + 1, `W${k}`, { fill: C.text2, "font-size": 17 });
      });
      return { s, S: { x: S.x, y: S.y + 28 * s }, W: W.map((w) => ({ x: w.x, y: w.y - r })) };
    }

    function ringTopo(g, cx, cy, R, r) {
      const angs = [-90, 0, 90, 180];
      angs.forEach((a) => add(g, "path", { d: arcPath(cx, cy, R, a + 18, a + 72), fill: "none", stroke: C.lineStrong, "stroke-width": 2, "marker-end": svg.ref("arrow-line") }));
      angs.forEach((a, k) => {
        const p = { x: cx + R * Math.cos((a * Math.PI) / 180), y: cy + R * Math.sin((a * Math.PI) / 180) };
        add(g, "circle", { cx: p.x, cy: p.y, r, fill: C.panel, stroke: C.ar, "stroke-width": 2 });
        if (r > 20) label(g, p.x, p.y + 1, `W${k}`, { fill: C.text2, "font-size": 17 });
      });
      return { cx, cy, R, angs, s: r / 26 };
    }

    const cards = specs.map((c, i) => {
      const x0 = 10 + i * 420;
      const cx = x0 + 200;
      const cy = 270;
      const g = group(svg);
      add(g, "rect", { x: x0, y: 10, width: 400, height: 520, rx: 8, fill: C.panel, stroke: c.color, "stroke-width": 2.2 });
      label(g, cx, 62, c.name, { fill: C.text, "font-size": 27 });
      label(g, cx, 482, c.kw, { fill: c.color, "font-size": 23 });
      return { g, cx, cy };
    });
    const topo = [
      starTopo(cards[0].g, cards[0].cx, cards[0].cy, C.ps, 1, false),
      ringTopo(cards[1].g, cards[1].cx, cards[1].cy, 105, 26),
      starTopo(cards[2].g, cards[2].cx, cards[2].cy, C.psHi, 1, true),
    ];
    const ssRing = ringTopo(cards[3].g, cards[3].cx - 95, cards[3].cy, 60, 15);
    add(cards[3].g, "line", { x1: cards[3].cx - 18, y1: cards[3].cy, x2: cards[3].cx + 22, y2: cards[3].cy, stroke: C.ss, "stroke-width": 2.5, "marker-end": svg.ref("arrow-line") });
    const ssStar = starTopo(cards[3].g, cards[3].cx + 105, cards[3].cy, C.ps, 0.5, true);

    const banner = group(svg);
    add(banner, "rect", { x: 10, y: 556, width: 1660, height: 58, rx: 6, fill: "rgba(10,248,134,0.10)", stroke: C.accent, "stroke-width": 2 });
    label(banner, 840, 586, "O.N.O.: las cuatro, sobre la misma base, para compararlas", { fill: C.accent, "font-size": 27 });

    gsap.set(cards.map((c) => c.g), { autoAlpha: 0, y: 24 });
    gsap.set(banner, { autoAlpha: 0 });
    tl.to(cards[0].g, { autoAlpha: 1, y: 0, duration: 0.45, ease: "power2.out" }, 0.05);
    tl.addLabel("s0");
    for (let i = 1; i < 4; i++) {
      tl.to(cards[i].g, { autoAlpha: 1, y: 0, duration: 0.45, ease: "power2.out" }, `s${i - 1}`);
      tl.addLabel(`s${i}`);
    }
    tl.fromTo(banner, { autoAlpha: 0, scaleX: 0.9, transformOrigin: "50% 50%" }, { autoAlpha: 1, scaleX: 1, duration: 0.45, ease: "power2.out" }, "s3");
    tl.addLabel("s4");

    function starLoop(a, t, async, t0) {
      t.W.forEach((w, k) => {
        const g = gradPacket(svg, 7 * Math.max(t.s, 0.7));
        const p = paramPacket(svg, 12 * Math.max(t.s, 0.7));
        a.data.push(g, p);
        const off = async ? [0, 0.75, 1.35][k] : 0;
        travel(a, g, w, t.S, t0 + off, 0.65);
        travel(a, p, t.S, w, t0 + (async ? off + 0.8 : 1.25), 0.65);
      });
    }

    function ringLoop(a, t, hops, t0) {
      for (let hop = 0; hop < hops; hop++) {
        t.angs.forEach((_, k) => {
          const pk = gradPacket(svg, 7 * Math.max(t.s, 0.7));
          a.data.push(pk);
          const a0 = t.angs[(k + hop) % 4];
          travelPath(a, pk, arcPoints(t.cx, t.cy, t.R, a0 + 18, a0 + 72, 8), t0 + hop * 0.72, 0.62);
        });
      }
    }

    const ambientFor = (step) => {
      const a = gsap.timeline({ repeat: -1, repeatDelay: 0.3 });
      a.data = [];
      starLoop(a, topo[0], false, 0);
      if (step >= 1) ringLoop(a, topo[1], 3, 0);
      if (step >= 2) starLoop(a, topo[2], true, 0);
      if (step >= 3) {
        ringLoop(a, ssRing, 2, 0);
        starLoop(a, ssStar, true, 1.5);
      }
      return a;
    };
    return controller(tl, 4, ambientFor);
  }


  function testbed(container) {
    const svg = svgRoot(container, 1100, 660);
    const tl = gsap.timeline();
    const busY = 480;
    const host = group(svg);
    add(host, "rect", { x: 30, y: 40, width: 1040, height: 520, rx: 10, fill: "none", stroke: C.lineStrong, "stroke-width": 2, "stroke-dasharray": "10 8" });
    label(host, 60, 74, "una máquina física", { "text-anchor": "start", fill: C.muted, "font-size": 20 });
    const bus = add(svg, "line", { x1: 70, y1: busY, x2: 1030, y2: busY, stroke: C.accent, "stroke-width": 3 });
    const busLabel = label(svg, 1030, busY + 32, "red bridge entre contenedores", { "text-anchor": "end", fill: C.muted, "font-size": 18 });

    const nodes = Array.from({ length: 16 }, (_, i) => {
      const row = i < 8 ? 0 : 1;
      const x = 105 + (i % 8) * 120 + row * 60;
      const y = row ? 280 : 120;
      const pipe = add(svg, "line", { x1: x, y1: y + 100, x2: x, y2: busY, stroke: C.lineStrong, "stroke-width": 7 });
      const g = group(svg);
      add(g, "rect", { x: x - 48, y, width: 96, height: 100, rx: 5, fill: C.panel, stroke: C.ar, "stroke-width": 2 });
      label(g, x, y + 36, "node", { fill: C.text2, "font-size": 18 });
      label(g, x, y + 68, "1 núcleo", { fill: C.muted, "font-size": 14 });
      return { g, pipe, x, y };
    });
    const counts = [2, 4, 6, 8, 12, 16];
    const ctr = counts.map((n) => label(svg, 1040, 74, `${n} workers`, { "text-anchor": "end", fill: C.accent, "font-size": 28 }));
    const limit = label(svg, 550, 612, "salida limitada a 10 Mbps por contenedor (Pumba)", { fill: C.warn, "font-size": 22 });

    show(tl, [host, bus, busLabel], 0.05, 0.4);
    gsap.set([...nodes.flatMap((n) => [n.g, n.pipe]), ...ctr, limit], { autoAlpha: 0 });
    tl.to(nodes.slice(0, 2).flatMap((n) => [n.pipe, n.g]), { autoAlpha: 1, duration: 0.35, stagger: 0.06 }, 0.3);
    tl.to(ctr[0], { autoAlpha: 1, duration: 0.3 }, 0.3);
    tl.addLabel("s0");

    counts.slice(1).forEach((n, j) => {
      const at = `s0+=${0.05 + j * 0.5}`;
      const fresh = nodes.slice(counts[j], n);
      tl.to(fresh.flatMap((nd) => [nd.pipe, nd.g]), { autoAlpha: 1, duration: 0.25, stagger: 0.03 }, at);
      tl.set(ctr[j], { autoAlpha: 0 }, at);
      tl.set(ctr[j + 1], { autoAlpha: 1 }, at);
    });
    tl.addLabel("s1");

    tl.to(nodes.map((n) => n.pipe), { attr: { "stroke-width": 2, stroke: C.warn }, duration: 0.6, stagger: 0.02 }, "s1");
    tl.to(limit, { autoAlpha: 1, duration: 0.35 }, "s1+=0.4");
    tl.addLabel("s2");

    const ambientFor = (step) => {
      if (step !== 2) return null;
      const a = gsap.timeline({ repeat: -1, repeatDelay: 0.2 });
      a.data = [];
      [0, 3, 5, 9, 12, 14].forEach((k, j) => {
        const n = nodes[k];
        const g = gradPacket(svg, 7);
        a.data.push(g);
        travel(a, g, { x: n.x, y: n.y + 100 }, { x: n.x, y: busY }, j * 0.35, 1.8);
      });
      return a;
    };
    return controller(tl, 2, ambientFor);
  }

  const BUILDERS = { syncasync, arch, flatbuffer, ps, ring, dsync, sswindow, ssflow, compress, topo, cost, scaleup, scaleout, parallelism, drift, alternatives, testbed };

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
