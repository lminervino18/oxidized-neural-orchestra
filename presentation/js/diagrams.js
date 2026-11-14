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


  function svgRoot(container, w, h, view = [0, 0, w, h]) {
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", view.join(" "));
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

  function legend(parent, x, y, items, size = 31) {
    const g = group(parent);
    let cx = x;
    for (const [kind, text] of items) {
      const icon = kind === "grad" ? gradPacket(g, 10) : kind === "param" ? paramPacket(g, 19) : ctrlPacket(g, 14);
      icon.setAttribute("transform", `translate(${cx + 10},${y})`);
      label(g, cx + 30, y + 1, text, { "text-anchor": "start", "font-size": size, fill: C.text2 });
      cx += 30 + text.length * 0.6 * size + 40;
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
      get step() { return current; },
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
    const svg = svgRoot(container, 1120, 660, [0, -8, 1120, 610]);
    const tl = gsap.timeline();
    const unit = 62;
    const x0 = 120;
    const rowH = 54;

    function lanes(y, title) {
      const g = group(svg);
      label(g, 0, y - 38, title, { "text-anchor": "start", fill: C.text, "font-size": 32 });
      for (let i = 0; i < 3; i++) label(g, 30, y + i * rowH + 20, `W${i}`, { "text-anchor": "start", fill: C.text2, "font-size": 31 });
      add(g, "line", { x1: x0, y1: y + 3 * rowH + 6, x2: 1100, y2: y + 3 * rowH + 6, stroke: C.line, "stroke-width": 1.5, "marker-end": svg.ref("arrow-muted") });
      label(g, 1100, y + 3 * rowH + 34, "tiempo", { "text-anchor": "end", fill: C.text2, "font-size": 31 });
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
      label(b, bx, syncY - 26, "barrera", { fill: C.accent, "font-size": 31 });
      barriers.push(b);
      t += 0.35;
    });
    const idleNote = label(svg, x0 + t * unit, syncY + 1.5 * rowH, "espera", { fill: C.text2, "font-size": 31, "text-anchor": "start" });

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
    add(stale, "path", { d: `M${sx},${asyncY + rowH + 19} Q${sx + 6},${asyncY - 16} ${sx + 64},${asyncY - 34}`, fill: "none", stroke: C.warn, "stroke-width": 2 });
    label(stale, sx + 74, asyncY - 34, "calculado sobre θ desactualizado", { fill: C.warn, "font-size": 31, "text-anchor": "start" });

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
    const svg = svgRoot(container, 1100, 720, [64, -12, 972, 719]);
    const tl = gsap.timeline();
    const ui = group(svg);
    add(ui, "rect", { x: 285, y: 0, width: 530, height: 64, rx: 5, fill: "#07100b", stroke: C.line, "stroke-width": 1.5 });
    label(ui, 550, 33, "orchestui  ·  orchestra-py", { fill: C.text2, "font-size": 31 });
    const orch = group(svg);
    add(orch, "rect", { x: 370, y: 92, width: 360, height: 92, rx: 6, fill: C.panel, stroke: C.accent, "stroke-width": 2.5 });
    label(orch, 550, 122, "orchestrator", { fill: C.text, "font-size": 32 });
    label(orch, 550, 160, "plano de control", { fill: C.text2, "font-size": 31 });

    const xs = [190, 430, 670, 910];
    const roles = ["worker", "worker", "worker", "server"];
    const nodes = xs.map((x, i) => {
      const g = group(svg);
      const box = add(g, "rect", { x: x - 95, y: 430, width: 190, height: 120, rx: 6, fill: C.panel, stroke: C.muted, "stroke-width": 2 });
      const generic = label(g, x, 476, "node", { fill: C.text2, "font-size": 32 });
      const role = label(g, x, 490, roles[i], { fill: roles[i] === "server" ? C.psHi : C.arHi, "font-size": 32 });
      const noRole = label(g, x, 516, "sin rol", { fill: C.text2, "font-size": 31 });
      return { g, box, generic, role, noRole, x };
    });

    const ctrl = xs.map((x) => add(svg, "line", { x1: 550, y1: 184, x2: x, y2: 430, stroke: C.ctrl, "stroke-width": 1.5, "stroke-dasharray": "6 6", opacity: 0.7 }));
    const handshake = label(svg, 550, 300, "Connect / Accept  ·  spec", { fill: C.ctrl, "font-size": 31, stroke: "#000", "stroke-width": 8, "paint-order": "stroke" });
    const ctrlPk = xs.map(() => ctrlPacket(svg));

    const dataY = 620;
    const data = group(svg);
    xs.slice(0, 3).forEach((x) => add(data, "path", { d: `M${x},550 C${x},${dataY + 40} ${910},${dataY + 40} 910,550`, fill: "none", stroke: C.lineStrong, "stroke-width": 2 }));
    label(data, 550, 680, "plano de datos: gradientes y parámetros entre nodos", { fill: C.text2, "font-size": 31 });

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
      tl.to([n.generic, n.noRole], { autoAlpha: 0, duration: 0.2 }, `s1+=${i * 0.08}`);
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
    const svg = svgRoot(container, 1100, 620, [48, 64, 1014, 529]);
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
      add(g, "rect", { x: cx - 90, y: 76, width: 180, height: 76, rx: 6, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2 });
      label(g, cx, 115, r.name, { fill: C.text, "font-size": 32 });
      return g;
    });

    const buf = group(svg);
    const cells = ranges.map((r) => add(buf, "rect", { x: r.x + 2, y: by, width: r.w - 4, height: 70, fill: "rgba(10,248,134,0.10)", stroke: C.accent, "stroke-width": 1.5 }));
    for (let k = 0; k < 49; k++) add(buf, "line", { x1: bx + k * 20, y1: by + 52, x2: bx + k * 20, y2: by + 70, stroke: C.accent, "stroke-width": 1, opacity: 0.35 });
    label(buf, bx, by + 104, "buffer contiguo de f32 (parámetros de todo el modelo)", { "text-anchor": "start", fill: C.text2, "font-size": 31 });

    const views = ranges.map((r) => {
      const cx = r.x + r.w / 2;
      return add(svg, "path", { d: `M${cx},152 L${cx},${by - 4}`, stroke: C.accent, "stroke-width": 1.5, "stroke-dasharray": "5 5", fill: "none", "marker-end": svg.ref("arrow-accent") });
    });

    const s0 = group(svg);
    add(s0, "path", { d: `M${ranges[1].x + 4},${by + 130} L${ranges[1].x + 4},${by + 146} L${ranges[1].x + ranges[1].w - 4},${by + 146} L${ranges[1].x + ranges[1].w - 4},${by + 130}`, fill: "none", stroke: C.psHi, "stroke-width": 2 });
    label(s0, ranges[1].x + ranges[1].w / 2, by + 176, "servidor 0", { fill: C.psHi, "font-size": 31 });
    const s1 = group(svg);
    [ranges[0], ranges[2]].forEach((r) => add(s1, "path", { d: `M${r.x + 4},${by + 130} L${r.x + 4},${by + 146} L${r.x + r.w - 4},${by + 146} L${r.x + r.w - 4},${by + 130}`, fill: "none", stroke: C.ps, "stroke-width": 2 }));
    label(s1, ranges[0].x + ranges[0].w / 2, by + 176, "servidor 1", { fill: C.ps, "font-size": 31 });
    label(s1, ranges[2].x + ranges[2].w / 2, by + 176, "servidor 1", { fill: C.ps, "font-size": 31 });
    const greedy = group(svg);
    label(greedy, 550, by + 224, "capas enteras, de mayor a menor,", { fill: C.text2, "font-size": 31 });
    label(greedy, 550, by + 264, "al servidor menos cargado", { fill: C.text2, "font-size": 31 });

    const wire = group(svg);
    const pk = group(wire);
    add(pk, "rect", { x: -70, y: -24, width: 140, height: 48, rx: 3, fill: "rgba(10,248,134,0.18)", stroke: C.accent, "stroke-width": 1.5 });
    label(pk, 0, 1, "&[u8]", { fill: C.accent, "font-size": 31 });
    const wireText = group(svg);
    add(wireText, "rect", { x: 50, y: 194, width: 580, height: 90, fill: "#000" });
    const halo = { stroke: "#000", "stroke-width": 8, "paint-order": "stroke" };
    label(wireText, 60, 220, "bytes del tramo tal cual → red", { "text-anchor": "start", fill: C.text2, "font-size": 31, ...halo });
    label(wireText, 60, 258, "(bytemuck, sin copias)", { "text-anchor": "start", fill: C.text2, "font-size": 31, ...halo });

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
    travel(tl, pk, { x: ranges[1].x + ranges[1].w / 2, y: by + 35 }, { x: 980, y: 236 }, tl.duration() + 0.05, 0.9);
    tl.set(pk, { autoAlpha: 1 }, ">");
    tl.set(wire, { autoAlpha: 1 }, "s1");
    tl.to(wireText, { autoAlpha: 1, duration: 0.3 }, "s1+=0.3");
    tl.addLabel("s2");
    return controller(tl, 2);
  }


  function ps(container) {
    const svg = svgRoot(container, 1100, 720, [56, 30, 946, 690]);
    const tl = gsap.timeline();
    const FS = 31;
    const W = [{ x: 170, y: 120 }, { x: 170, y: 330 }, { x: 170, y: 540 }];
    const S0 = { x: 810, y: 230 };
    const S1 = { x: 810, y: 520 };
    const SH = 150;

    const links0 = W.map((w) => add(svg, "line", { x1: w.x + 62, y1: w.y, x2: S0.x - SH, y2: S0.y, stroke: C.lineStrong, "stroke-width": 2 }));
    const links1 = W.map((w) => add(svg, "line", { x1: w.x + 62, y1: w.y, x2: S1.x - SH, y2: S1.y, stroke: C.lineStrong, "stroke-width": 2 }));

    const workers = W.map((w, i) => {
      const g = group(svg);
      const ring = add(g, "circle", { cx: w.x, cy: w.y, r: 62, fill: C.panel, stroke: C.ar, "stroke-width": 2.5 });
      label(g, w.x, w.y + 1, `W${i}`, { fill: C.text, "font-size": 32 });
      label(g, w.x, w.y + 88, `partición ${i}`, { fill: C.text2, "font-size": FS });
      return { g, ring };
    });

    function server(pos, name, inner) {
      const g = group(svg);
      const box = add(g, "rect", { x: pos.x - SH, y: pos.y - 86, width: 2 * SH, height: 172, rx: 6, fill: C.panel, stroke: C.ps, "stroke-width": 2.5 });
      label(g, pos.x, pos.y - 46, name, { fill: C.text, "font-size": FS });
      const theta = label(g, pos.x, pos.y + 1, inner, { fill: C.psHi, "font-size": FS });
      const opt = label(g, pos.x, pos.y + 48, "optimizador", { fill: C.text2, "font-size": FS });
      return { g, box, theta, opt };
    }

    const srv0 = server(S0, "servidor 0", "θ");
    const srv1 = server(S1, "servidor 1", "θ · capas 1, 3");
    const srv0split = label(svg, S0.x, S0.y + 1, "θ · capa 2", { fill: C.psHi, "font-size": FS });
    const thetaNew = label(svg, S0.x, S0.y + 1, "θ′", { fill: C.accent, "font-size": 34 });

    const pulls = W.map(() => paramPacket(svg));
    const pushes = W.map(() => gradPacket(svg));
    const localG = W.map((w) => {
      const g = group(svg, { transform: `translate(${w.x + 52},${w.y - 56})` });
      add(g, "circle", { r: 22, fill: C.grad, stroke: "#000", "stroke-width": 2 });
      label(g, 0, 1, "g", { fill: "#000", "font-size": 31, "font-weight": 700 });
      return g;
    });

    const counter = group(svg);
    add(counter, "rect", { x: S0.x + 10, y: S0.y - 160, width: 100, height: 56, rx: 4, fill: "#0b1810", stroke: C.grad, "stroke-width": 1.5 });
    const cnt = [1, 2, 3].map((k) => label(counter, S0.x + 60, S0.y - 131, `${k}/3`, { fill: C.grad, "font-size": FS }));
    const avg = richLabel(svg, S0.x, S0.y + 124, [["Σg / N"], ["W", "sub"], [" → Adam → θ′"]], { fill: C.accent, "font-size": FS });
    const barrierNote = label(svg, S0.x - 70, S0.y - 131, "barrier", { fill: C.text2, "font-size": FS });

    const leg = legend(svg, 520, 690, [["grad", "gradiente"], ["param", "parámetros"]]);

    show(tl, [...workers.map((w) => w.g), srv0.g, ...links0, leg], 0.05, 0.4, { stagger: 0.05 });
    gsap.set([srv1.g, ...links1, srv0split, thetaNew, counter, avg, barrierNote, ...localG, ...cnt], { autoAlpha: 0 });
    tl.addLabel("s0");

    pulls.forEach((p, i) => travel(tl, p, { x: S0.x - SH, y: S0.y }, { x: W[i].x + 62, y: W[i].y }, tl.duration() + i * 0.08, 0.8));
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
      travel(tl, p, { x: W[i].x + 62, y: W[i].y }, { x: S0.x - SH, y: S0.y }, at, 0.8);
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
        travel(a, g, { x: w.x + 62, y: w.y }, { x: S.x - SH, y: S.y }, i * 0.2 + k * 0.1, 1.1);
        travel(a, p, { x: S.x - SH, y: S.y }, { x: w.x + 62, y: w.y }, 1.6 + i * 0.2 + k * 0.1, 1.1);
      }));
      return a;
    };
    return controller(tl, 5, ambientFor);
  }


  function ring(container) {
    const svg = svgRoot(container, 1100, 800, [36, 44, 1028, 736]);
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
      label(g, p.x, p.y + 1, `W${i}`, { fill: C.text, "font-size": 32 });
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
        label(rg, o.x + c * (cellW + 6) + cellW / 2, o.y + cellH + 22, `c${c}`, { fill: C.text2, "font-size": 32 });
      }
    }

    const phaseTexts = {};
    function phase(key, line1, line2) {
      const g = group(svg);
      label(g, cx, cy - 50, line1, { fill: C.text, "font-size": 34 });
      if (line2) label(g, cx, cy + 50, line2, { fill: C.text2, "font-size": 32 });
      phaseTexts[key] = g;
      return g;
    }
    const allPhases = [
      phase("split", "4 fragmentos", "por gradiente"),
      phase("rs1", "reduce-scatter", "iteración 1 / 3"), phase("rs2", "reduce-scatter", "iteración 2 / 3"), phase("rs3", "reduce-scatter", "iteración 3 / 3"),
      phase("rsd", "cada worker tiene", "un fragmento completo"),
      phase("ag1", "all-gather", "iteración 1 / 3"), phase("ag2", "all-gather", "iteración 2 / 3"), phase("ag3", "all-gather", "iteración 3 / 3"),
      phase("done", "÷ 4 → ḡ", "optimizador local"),
    ];

    show(tl, [...arcs, ...nodes], 0.05, 0.4, { stagger: 0.05 });
    gsap.set([...rows, ...allPhases], { autoAlpha: 0 });
    tl.addLabel("s0");
    tl.to(rows, { autoAlpha: 1, duration: 0.4, stagger: 0.08 }, "s0");
    tl.to(phaseTexts.split, { autoAlpha: 1, duration: 0.3 }, "s0");
    tl.addLabel("s1");

    const PK_DUR = 0.85;
    const ITER_GAP = 1.35;
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

    let step = 2;
    let at = tl.duration() + 0.05;
    for (let k = 0; k < n - 1; k++) {
      switchPhase(`rs${k + 1}`, at);
      runIteration("rs", k, at + 0.25);
      at += ITER_GAP;
      if (k < n - 2) tl.addLabel(`s${step++}`, at);
      at += 0.05;
    }
    switchPhase("rsd", at);
    for (let p = 0; p < n; p++) {
      const c = (p + 1) % n;
      tl.to(cells[p][c].g.querySelector("rect"), { attr: { stroke: C.accent }, duration: 0.3 }, at + 0.1);
    }
    tl.addLabel(`s${step++}`, at + 0.5);

    at += 0.55;
    for (let k = 0; k < n - 1; k++) {
      switchPhase(`ag${k + 1}`, at);
      runIteration("ag", k, at + 0.25);
      at += ITER_GAP;
      tl.addLabel(`s${step++}`, at);
      at += 0.05;
    }

    switchPhase("done", at);
    cells.flat().forEach((cell) => tl.to(cell.g.querySelector("rect"), { attr: { stroke: C.accent }, duration: 0.3 }, at + 0.1));
    tl.addLabel(`s${step}`);
    return controller(tl, step);
  }


  function dsync(container) {
    const VW = 1000;
    const svg = svgRoot(container, VW, 700, [-9, -6, 991, 698]);
    const tl = gsap.timeline();
    const FS = 31;
    const SR = 0.1;
    const X0 = 150, X1 = VW - 30, Y0 = 570, Y1 = 214;
    const xMax = 17, yMax = 3.4;
    const sx = (nw) => X0 + ((nw - 1) / (xMax - 1)) * (X1 - X0);
    const sy = (d) => Y0 - (d / yMax) * (Y0 - Y1);
    const psF = (nw) => 2 * nw * SR;
    const arF = (nw) => 2 * (1 - 1 / nw) * SR;

    const head = group(svg);
    richLabel(head, VW / 2, 32, [["Modelo analítico: d"], ["sync", "sub"], [" por ronda"]], { fill: C.text, "font-size": 34 });
    const entries = [["PS · 1 servidor", C.ps], ["Ring All-Reduce", C.ar]];
    const entryW = (name) => 62 + name.length * 0.6 * FS;
    let lx = VW / 2 - (entries.reduce((s, [n]) => s + entryW(n), 0) + 44) / 2;
    entries.forEach(([name, color]) => {
      add(head, "line", { x1: lx, y1: 92, x2: lx + 48, y2: 92, stroke: color, "stroke-width": 5 });
      add(head, "circle", { cx: lx + 24, cy: 92, r: 8, fill: color, stroke: "#000", "stroke-width": 2 });
      label(head, lx + 62, 93, name, { "text-anchor": "start", fill: C.text, "font-size": FS });
      lx += entryW(name) + 44;
    });
    label(head, VW / 2, 142, "S/R ≈ 0,1 s (LeNet-5 en f16, 10 Mbps)", { fill: C.text2, "font-size": FS });

    const axes = group(svg);
    for (let d = 0; d <= 3; d += 1) {
      add(axes, "line", { x1: X0, y1: sy(d), x2: X1, y2: sy(d), stroke: C.line, "stroke-width": 1.5 });
      label(axes, X0 - 16, sy(d), String(d), { "text-anchor": "end", fill: C.text2, "font-size": FS });
    }
    [2, 4, 6, 8, 12, 16].forEach((nw) => label(axes, sx(nw), Y0 + 36, String(nw), { fill: C.text2, "font-size": FS }));
    add(axes, "line", { x1: X0, y1: Y0, x2: X1, y2: Y0, stroke: C.lineStrong, "stroke-width": 2 });
    add(axes, "line", { x1: X0, y1: Y0, x2: X0, y2: Y1 - 14, stroke: C.lineStrong, "stroke-width": 2 });
    label(axes, (X0 + X1) / 2, Y0 + 92, "cantidad de workers", { fill: C.text2, "font-size": FS });
    richLabel(axes, 26, (Y0 + Y1) / 2, [["d"], ["sync", "sub"], [" (s)"]], { fill: C.text2, "font-size": FS, transform: `rotate(-90 26 ${(Y0 + Y1) / 2})` });

    function curve(f, color) {
      let d = "";
      for (let nw = 2; nw <= 16.001; nw += 0.25) d += `${d ? "L" : "M"}${sx(nw).toFixed(1)},${sy(f(nw)).toFixed(1)}`;
      return add(svg, "path", { d, fill: "none", stroke: color, "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round" });
    }

    function markers(f, color, name) {
      const g = group(svg);
      [2, 4, 6, 8, 12, 16].forEach((nw) => {
        const m = add(g, "circle", { cx: sx(nw), cy: sy(f(nw)), r: 8, fill: color, stroke: "#000", "stroke-width": 2 });
        const t = add(m, "title");
        t.textContent = `${name} · N_W = ${nw}: ${f(nw).toFixed(3).replace(".", ",")} s`;
      });
      return g;
    }

    const psLine = curve(psF, C.ps);
    const psMarks = markers(psF, C.ps, "PS, 1 servidor");
    const arLine = curve(arF, C.ar);
    const arMarks = markers(arF, C.ar, "Ring All-Reduce");
    const asym = add(svg, "line", { x1: X0, y1: sy(2 * SR), x2: X1, y2: sy(2 * SR), stroke: C.text2, "stroke-width": 2, "stroke-dasharray": "6 8" });
    const asymLabel = label(svg, X1, sy(2 * SR) - 30, "2S/R = 0,2 s", { "text-anchor": "end", fill: C.text, "font-size": FS });

    show(tl, [head, axes], 0.05, 0.4);
    gsap.set([psMarks, arMarks, asym, asymLabel], { autoAlpha: 0 });
    gsap.set([psLine, arLine], { autoAlpha: 0 });
    tl.addLabel("s0");
    tl.set(psLine, { autoAlpha: 1 }, "s0");
    draw(tl, psLine, "s0", 1);
    tl.to(psMarks, { autoAlpha: 1, duration: 0.3 }, "s0+=0.8");
    tl.addLabel("s1");
    tl.set(arLine, { autoAlpha: 1 }, "s1");
    draw(tl, arLine, "s1", 1);
    tl.to(arMarks, { autoAlpha: 1, duration: 0.3 }, "s1+=0.8");
    tl.to([asym, asymLabel], { autoAlpha: 1, duration: 0.3 }, "s1+=0.9");
    tl.addLabel("s2");
    return controller(tl, 2);
  }


  function sswindow(container) {
    const svg = svgRoot(container, 1100, 680, [11, 23, 1065, 651]);
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
    label(axes, (X0 + X1) / 2, Y0 + 40, "rondas", { fill: C.text2, "font-size": 31 });
    label(axes, 46, (Y0 + Y1) / 2, "pérdida máx.", { fill: C.text2, "font-size": 31, transform: `rotate(-90 46 ${(Y0 + Y1) / 2})` });
    const badge = label(svg, X1, 300, "curva ilustrativa", { "text-anchor": "end", fill: C.warn, "font-size": 31 });

    const pathD = loss.map((l, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(l).toFixed(1)}`).join("");
    const curveEl = add(svg, "path", { d: pathD, fill: "none", stroke: C.text2, "stroke-width": 2.5 });
    const dots = group(svg);
    loss.forEach((l, i) => add(dots, "circle", { cx: sx(i), cy: sy(l), r: 4.5, fill: C.text2, stroke: "#000", "stroke-width": 1.5 }));

    const band = add(svg, "rect", { x: 0, y: Y1 - 10, width: sx(W - 1) - sx(0) + 24, height: Y0 - Y1 + 10, fill: C.accent, "fill-opacity": 0.08, stroke: C.accent, "stroke-width": 1.5, "stroke-opacity": 0.6 });
    const bandLabel = label(svg, 0, Y1 - 32, "ventana w = 6", { fill: C.accent, "font-size": 31 });
    const sTexts = [];
    for (let k = W - 1; k <= trigger; k++) {
      const val = (sOf(k) * 100).toFixed(1).replace(".", ",");
      const t = label(svg, X1 - 10, 150, `s = ${val} %`, { "text-anchor": "end", fill: k === trigger ? C.accent : C.text, "font-size": 34 });
      sTexts.push(t);
    }
    const sw = group(svg);
    add(sw, "line", { x1: sx(trigger) + 12, y1: Y1 - 20, x2: sx(trigger) + 12, y2: Y0, stroke: C.ss, "stroke-width": 2.5 });
    label(sw, sx(trigger) + 26, Y0 - 120, "switch", { "text-anchor": "start", fill: C.ss, "font-size": 31 });
    const phaseL = label(svg, (X0 + sx(trigger)) / 2, Y0 + 84, "All-Reduce · sincrónico", { fill: C.ar, "font-size": 31 });
    const phaseR = label(svg, (sx(trigger) + X1) / 2 + 10, Y0 + 84, "PS · asincrónico", { fill: C.ps, "font-size": 31 });
    const thresh = label(svg, X1 - 10, 194, "umbral: 1 %", { "text-anchor": "end", fill: C.text2, "font-size": 31 });

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
    const svg = svgRoot(container, 1100, 780, [54, 28, 1057, 750]);
    const tl = gsap.timeline();
    const cx = 470, cy = 470, R = 175;
    const angles = [-90, 0, 90, 180];
    const ringPos = angles.map((a) => ({ x: cx + R * Math.cos(a * Math.PI / 180), y: cy + R * Math.sin(a * Math.PI / 180) }));
    const psPos = [{ x: 210, y: 300 }, { x: 210, y: 480 }, { x: 210, y: 660 }, { x: 830, y: 480 }];

    const orch = group(svg);
    add(orch, "rect", { x: 300, y: 40, width: 340, height: 100, rx: 6, fill: C.panel, stroke: C.accent, "stroke-width": 2.5 });
    label(orch, 470, 74, "orchestrator", { fill: C.text, "font-size": 32 });
    label(orch, 470, 112, "SwitchTracker", { fill: C.text2, "font-size": 31 });

    const win = group(svg);
    const slots = [];
    for (let k = 0; k < 6; k++) {
      add(win, "rect", { x: 690 + k * 44, y: 58, width: 36, height: 36, rx: 3, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
      slots.push(add(win, "rect", { x: 694 + k * 44, y: 62, width: 28, height: 28, rx: 2, fill: C.ctrl, opacity: 0 }));
    }
    label(win, 690, 124, "pérdida máx. por ronda", { "text-anchor": "start", fill: C.text2, "font-size": 31 });
    const sVal = label(svg, 690, 168, "s ≤ 1 %  →  cambiar", { "text-anchor": "start", fill: C.accent, "font-size": 31 });

    const arcs = angles.map((a) => add(svg, "path", { d: arcPath(cx, cy, R, a + 22, a + 68), fill: "none", stroke: C.lineStrong, "stroke-width": 2.5, "marker-end": svg.ref("arrow-line") }));
    const ctrlLines = ringPos.map((p) => add(svg, "line", { x1: p.x, y1: p.y - 42, x2: 470, y2: 140, stroke: C.ctrl, "stroke-width": 1.5, "stroke-dasharray": "5 6", opacity: 0 }));

    const nodes = ringPos.map((p, i) => {
      const g = group(svg);
      gsap.set(g, { x: p.x, y: p.y });
      const circle = add(g, "circle", { cx: 0, cy: 0, r: 44, fill: C.panel, stroke: C.ar, "stroke-width": 2.5 });
      const name = label(g, 0, 1, `W${i}`, { fill: C.text, "font-size": 31 });
      return { g, circle, name };
    });
    const srvBox = add(nodes[3].g, "rect", { x: -135, y: -62, width: 270, height: 124, rx: 6, fill: C.panel, stroke: C.ps, "stroke-width": 2.5, opacity: 0 });
    const srvName = label(nodes[3].g, 0, -20, "S0", { fill: C.text, "font-size": 32, opacity: 0 });
    const srvSub = label(nodes[3].g, 0, 22, "θ de su tramo", { fill: C.psHi, "font-size": 31, opacity: 0 });
    nodes[3].g.appendChild(nodes[3].name);

    const upLosses = ringPos.map(() => ctrlPacket(svg));
    const downCmd = ringPos.map(() => ctrlPacket(svg, 15));
    const cmdLabels = ringPos.map((p, i) => label(svg, p.x + (i === 3 ? -74 : 74), p.y - 66, i === 3 ? "Upgrade" : "Switch", { fill: i === 3 ? C.psHi : C.ctrl, "font-size": 31 }));

    const psLinks = psPos.slice(0, 3).map((p) => add(svg, "line", { x1: p.x + 44, y1: p.y, x2: psPos[3].x - 135, y2: psPos[3].y, stroke: C.lineStrong, "stroke-width": 2 }));
    const phase1 = label(svg, 470, 748, "fase 1 · All-Reduce en anillo", { fill: C.ar, "font-size": 31 });
    const phase2 = label(svg, 540, 748, "fase 2 · Parameter Server, sin reiniciar el clúster", { fill: C.ps, "font-size": 31 });
    const noRestart = label(svg, 830, 600, "mismo proceso, nuevo rol", { fill: C.text2, "font-size": 31 });

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
    tl.to(nodes[3].circle, { autoAlpha: 0, duration: 0.3 }, "s2+=0.2");
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
        travel(a, g, { x: p.x + 44, y: p.y }, { x: psPos[3].x - 135, y: psPos[3].y }, i * 0.45, 1);
        travel(a, q, { x: psPos[3].x - 135, y: psPos[3].y }, { x: p.x + 44, y: p.y }, i * 0.45 + 1.15, 1);
      });
      return a;
    };
    return controller(tl, 4, ambientFor);
  }


  function compress(container) {
    const svg = svgRoot(container, 1100, 760, [28, -1, 1074, 753]);
    const tl = gsap.timeline();
    const FS = 31;
    const g1 = [0.32, 0.9, 0.18, 0.55, 0.12, 0.7, 0.26, 0.44, 0.08, 0.95, 0.3, 0.2, 0.62, 0.14, 0.38, 0.5, 0.1, 0.78, 0.22, 0.34];
    const g2 = [0.3, 0.2, 0.36, 0.15, 0.3, 0.25, 0.34, 0.1, 0.12, 0.2, 0.32, 0.18, 0.1, 0.2, 0.3, 0.12, 0.08, 0.15, 0.36, 0.28];
    const thr = 0.6;
    const X0 = 60, Y0 = 480, H = 320, bw = 30, gap = 14;
    const XE = X0 + 20 * (bw + gap);
    const sy = (m) => Y0 - m * H;

    const title = label(svg, X0, 34, "|g| por componente", { "text-anchor": "start", fill: C.text2, "font-size": FS });
    const badge = label(svg, 1090, 34, "ilustrativo · r = 0,25", { "text-anchor": "end", fill: C.warn, "font-size": FS });
    add(svg, "line", { x1: X0 - 10, y1: Y0, x2: XE, y2: Y0, stroke: C.lineStrong, "stroke-width": 1.5 });
    add(svg, "line", { x1: X0 - 10, y1: Y0, x2: X0 - 10, y2: Y0 - H, stroke: C.lineStrong, "stroke-width": 1.5 });

    const bars = g1.map((m, i) => add(svg, "rect", { x: X0 + i * (bw + gap), y: sy(m), width: bw, height: m * H, rx: 2, fill: C.muted, opacity: 0.8 }));
    const thrLine = group(svg);
    add(thrLine, "line", { x1: X0 - 20, y1: sy(thr), x2: XE, y2: sy(thr), stroke: C.accent, "stroke-width": 2, "stroke-dasharray": "8 6" });
    label(thrLine, XE + 10, sy(thr), "umbral", { "text-anchor": "start", fill: C.accent, "font-size": FS });
    const thrNote = group(svg);
    label(thrNote, X0, 530, "umbral = percentil (1 − r) de |g|,", { "text-anchor": "start", fill: C.text2, "font-size": FS });
    label(thrNote, X0, 570, "estimado con una muestra ≤ 16384 (quickselect)", { "text-anchor": "start", fill: C.text2, "font-size": FS });

    const sentIdx = g1.map((m, i) => (m >= thr ? i : -1)).filter((i) => i >= 0);
    const restIdx = g1.map((m, i) => (m < thr ? i : -1)).filter((i) => i >= 0);
    const wire = label(svg, 1000, 96, "→ red", { fill: C.grad, "font-size": 34 });
    const residualNote = label(svg, X0, 626, "lo que no se envía queda en el residual del worker", { "text-anchor": "start", fill: C.text2, "font-size": FS });

    const newBars = g2.map((m, i) => add(svg, "rect", { x: X0 + i * (bw + gap), y: sy(g1[i] < thr ? g1[i] + m : m), width: bw, height: m * H, rx: 2, fill: C.param, opacity: 0.85 }));
    const crossIdx = g2.map((m, i) => (g1[i] < thr && g1[i] + m >= thr ? i : -1)).filter((i) => i >= 0);
    const crossNote = group(svg);
    label(crossNote, X0, 682, "residual + gradiente nuevo: los componentes", { "text-anchor": "start", fill: C.accent, "font-size": FS });
    label(crossNote, X0, 722, "chicos pero constantes cruzan el umbral", { "text-anchor": "start", fill: C.accent, "font-size": FS });
    const legendG = group(svg);
    add(legendG, "rect", { x: X0, y: 72, width: 22, height: 22, fill: C.param });
    label(legendG, X0 + 34, 84, "gradiente de la ronda siguiente", { "text-anchor": "start", fill: C.text2, "font-size": FS });

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
    const svg = svgRoot(container, 1100, 670, [7, 4, 1085, 665]);
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
      label(g, ox, 40, title, { fill: C.text, "font-size": 32 });
      const edges = {};
      for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) {
        const pa = P[a], pb = P[b];
        const e = add(g, "line", { x1: ox + pa.x, y1: oy + pa.y, x2: ox + pb.x, y2: oy + pb.y, stroke: C.line, "stroke-width": 1.5 });
        const mx = ox + (pa.x + pb.x) / 2, my = oy + (pa.y + pb.y) / 2;
        const t = label(g, mx, my, String(L[a][b]), { fill: C.text2, "font-size": 30, stroke: "#000", "stroke-width": 7, "paint-order": "stroke" });
        edges[`${a}-${b}`] = { e, t };
      }
      const nodes = P.map((p, i) => {
        const c = add(g, "circle", { cx: ox + p.x, cy: oy + p.y, r: 34, fill: C.panel, stroke: C.muted, "stroke-width": 2 });
        label(g, ox + p.x, oy + p.y + 1, `n${i}`, { fill: C.text, "font-size": 31 });
        return c;
      });
      return { g, edges, nodes };
    }

    const left = graph(270, 300, "All-Reduce: anillo");
    const right = graph(830, 300, "Parameter Server: servidor");
    const edgeOf = (gph, a, b) => gph.edges[a < b ? `${a}-${b}` : `${b}-${a}`];
    const ringEdges = best.cyc.map((a, i) => edgeOf(left, a, best.cyc[(i + 1) % best.cyc.length]));
    const ringNote = group(svg);
    label(ringNote, 270, 540, "viajante exacto (Held–Karp)", { fill: C.accent, "font-size": 31 });
    label(ringNote, 270, 580, `costo total: ${best.cost}`, { fill: C.accent, "font-size": 31 });
    const centerEdges = [0, 1, 2, 3, 4].filter((j) => j !== center).map((j) => edgeOf(right, center, j));
    const psNote = group(svg);
    label(psNote, 830, 540, "minimiza la mayor latencia", { fill: C.psHi, "font-size": 31 });
    label(psNote, 830, 580, `worker–servidor: ${centerCost}`, { fill: C.psHi, "font-size": 31 });
    const caption = label(svg, 550, 640, "latencias ilustrativas (ms) · mayor RTT de 10 pings", { fill: C.text2, "font-size": 31 });

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
    tl.to(right.nodes[center], { attr: { stroke: C.ps, fill: "#102434" }, duration: 0.3 }, "s1");
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
    const svg = svgRoot(container, 1100, 680, [48, 74, 1061, 570]);
    const tl = gsap.timeline();

    const cards = Array.from({ length: 12 }, (_, i) => add(svg, "rect", { x: 60, y: 548 - i * 36, width: 170, height: 26, rx: 3, fill: "#151105", stroke: C.grad, "stroke-width": 1.5 }));
    label(svg, 145, 616, "datos", { fill: C.text2, "font-size": 28 });

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
    label(svg, 530, 616, "parámetros", { fill: C.text2, "font-size": 28 });

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
    const days = ["día 1", "día 2", "día 3"].map((t) => label(svg, CX, CY + 106, t, { fill: C.warn, "font-size": 28 }));

    const barG = group(svg);
    label(barG, 840, 314, "entrenamiento", { "text-anchor": "start", fill: C.text2, "font-size": 28 });
    add(barG, "rect", { x: 840, y: 334, width: 220, height: 24, rx: 3, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
    const barFill = add(barG, "rect", { x: 843, y: 337, width: 214, height: 18, rx: 2, fill: C.accent });

    const memG = group(svg);
    label(memG, 840, 434, "memoria", { "text-anchor": "start", fill: C.text2, "font-size": 28 });
    add(memG, "rect", { x: 840, y: 454, width: 220, height: 24, rx: 3, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
    const memFill = add(memG, "rect", { x: 843, y: 457, width: 214, height: 18, rx: 2, fill: C.param });
    const memOver = add(memG, "rect", { x: 1057, y: 457, width: 40, height: 18, rx: 2, fill: C.bad });
    const memNo = label(memG, 950, 516, "excede", { fill: C.bad, "font-size": 28 });

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
    const svg = svgRoot(container, 1100, 680, [108, 54, 884, 558]);
    const tl = gsap.timeline();
    const base = 600;
    const cx = 420;
    const w = 280;
    const ceilY = 120;

    const floor = add(svg, "line", { x1: 120, y1: base, x2: 980, y2: base, stroke: C.lineStrong, "stroke-width": 2 });
    const ceil = group(svg);
    const ceilLine = add(ceil, "line", { x1: 120, y1: ceilY, x2: 980, y2: ceilY, stroke: C.muted, "stroke-width": 2.5, "stroke-dasharray": "12 9" });
    const ceilText = label(ceil, 980, ceilY - 30, "límite de un solo equipo", { "text-anchor": "end", fill: C.text2, "font-size": 31 });

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
    const svg = svgRoot(container, 1100, 680, [108, 88, 884, 586]);
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
    label(sw, SW.x, SW.y + 1, "red", { fill: C.accent, "font-size": 30 });

    const machines = pos.map(() => {
      const g = group(svg);
      add(g, "rect", { x: -70, y: -50, width: 140, height: 100, rx: 5, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2 });
      add(g, "rect", { x: -48, y: -28, width: 96, height: 16, rx: 2, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
      add(g, "rect", { x: -48, y: 0, width: 96, height: 16, rx: 2, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
      gsap.set(g, { x: SW.x, y: SW.y, autoAlpha: 0 });
      return g;
    });
    const one = label(svg, SW.x, SW.y + 100, "una máquina común", { fill: C.text, "font-size": 36 });
    const sum = label(svg, SW.x, 644, "Σ cómputo  ·  Σ memoria", { fill: C.accent, "font-size": 31 });

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
    const rx = [661, 825, 989];
    const MY = 460;
    const FS = 31;

    function panel(x, title) {
      const g = group(svg);
      const frame = add(g, "rect", { x, y: 10, width: 530, height: 680, rx: 8, fill: "none", stroke: C.line, "stroke-width": 2 });
      label(g, x + 265, 54, title, { fill: C.text, "font-size": 32 });
      return { g, frame };
    }
    const left = panel(10, "paralelismo de modelo");
    const right = panel(560, "paralelismo de datos");

    function machine(x, i) {
      const g = group(svg);
      add(g, "rect", { x: x - 68, y: MY, width: 136, height: 150, rx: 6, fill: C.panel, stroke: MC[i], "stroke-width": 2.2 });
      label(g, x, MY + 180, `M${i}`, { fill: MC[i], "font-size": FS });
      return g;
    }

    const layerEdges = [0, 1].map((i) => add(svg, "line", { x1: lx[i] + 34, y1: 190, x2: lx[i + 1] - 34, y2: 190, stroke: C.accent, "stroke-width": 2, "marker-end": svg.ref("arrow-accent") }));
    const layers = lx.map((x) => {
      const g = group(svg);
      add(g, "rect", { x: x - 32, y: 105, width: 64, height: 170, rx: 5, fill: "rgba(10,248,134,0.08)", stroke: C.accent, "stroke-width": 1.8 });
      [0, 1, 2, 3].forEach((k) => add(g, "circle", { cx: x, cy: 130 + k * 40, r: 10, fill: C.panel, stroke: C.accent, "stroke-width": 1.8 }));
      return g;
    });
    const netNote = label(svg, 275, 318, "una red de tres capas", { fill: C.text2, "font-size": FS });
    const leftMachines = lx.map((x, i) => machine(x, i));
    const leftArrows = [0, 1].map((i) => add(svg, "line", { x1: lx[i] + 70, y1: MY + 75, x2: lx[i + 1] - 72, y2: MY + 75, stroke: C.lineStrong, "stroke-width": 2.5, "marker-end": svg.ref("arrow-line") }));
    const leftCap = label(svg, 275, 330, "se parte la red", { fill: C.text, "font-size": FS });

    const SW = 150, SP = 164, SX = 560 + (530 - 3 * SP + (SP - SW)) / 2;
    const shards = rx.map((x, i) => {
      const g = group(svg);
      const r = add(g, "rect", { x: SX + i * SP, y: 118, width: SW, height: 50, rx: 3, fill: "#151105", stroke: C.grad, "stroke-width": 1.8 });
      label(g, SX + i * SP + SW / 2, 144, `datos ${i}`, { fill: C.text2, "font-size": FS });
      return { g, r, dx: x - (SX + i * SP + SW / 2) };
    });
    const dsNote = label(svg, 825, 206, "un dataset", { fill: C.text2, "font-size": FS });
    const rightMachines = rx.map((x, i) => machine(x, i));
    const replicas = rx.map((x) => {
      const g = group(svg);
      [-30, 0, 30].forEach((dx) => add(g, "rect", { x: x + dx - 11, y: MY + 18, width: 22, height: 80, rx: 3, fill: "rgba(10,248,134,0.08)", stroke: C.accent, "stroke-width": 1.6 }));
      label(g, x, MY + 126, "modelo", { fill: C.text2, "font-size": FS });
      return g;
    });
    const rightCap = label(svg, 825, 330, "se parte el dataset", { fill: C.text, "font-size": FS });
    const badge = group(svg);
    add(badge, "rect", { x: 725, y: 150, width: 200, height: 68, rx: 6, fill: "rgba(10,248,134,0.14)", stroke: C.accent, "stroke-width": 2.5 });
    label(badge, 825, 185, "O.N.O.", { fill: C.accent, "font-size": 34 });

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
      tl.to(s.g, { x: s.dx, y: MY - 60 - 140, duration: 0.8, ease: "power2.inOut" }, `s1+=${0.2 + i * 0.1}`);
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
        const r = add(svg, "rect", { x: x - 68, y: MY, width: 136, height: 150, rx: 6, fill: MC[i], opacity: 0 });
        a.data.push(r);
        a.to(r, { opacity: step === 3 && x < 550 ? 0.1 : 0.3, duration: 0.3, yoyo: true, repeat: 1 }, at);
      };
      lx.forEach((x, i) => {
        glow(x, i, i * 0.95);
        if (i < 2) {
          const pk = gradPacket(svg, 8);
          a.data.push(pk);
          travel(a, pk, { x: x + 68, y: MY + 75 }, { x: lx[i + 1] - 68, y: MY + 75 }, i * 0.95 + 0.55, 0.4);
        }
      });
      if (step >= 2) rx.forEach((x, i) => glow(x, i, 0));
      return a;
    };
    return controller(tl, 3, ambientFor);
  }


  function drift(container) {
    const svg = svgRoot(container, 1100, 700, [8, -6, 999, 713]);
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
    const badge = label(svg, 20, 30, "ilustrativo", { "text-anchor": "start", fill: C.warn, "font-size": 31 });
    const star = group(svg);
    add(star, "path", { d: starPath(G.x, G.y, 17, 7), fill: C.accent });
    const halo = { stroke: "#000", "stroke-width": 8, "paint-order": "stroke" };
    label(star, G.x + 28, G.y - 26, "mínimo global", { "text-anchor": "start", fill: C.accent, "font-size": 31, ...halo });

    const minima = M.map((m, i) => {
      const g = group(svg);
      add(g, "path", { d: `M${m.x - 11},${m.y - 11} L${m.x + 11},${m.y + 11} M${m.x + 11},${m.y - 11} L${m.x - 11},${m.y + 11}`, stroke: WORKER[i], "stroke-width": 3.5 });
      const beside = i === 2;
      label(g, beside ? m.x + 26 : m.x, beside ? m.y + 2 : m.y + 36, `mínimo de los datos ${i}`, { "text-anchor": beside ? "start" : "middle", fill: WORKER[i], "font-size": 31, ...halo });
      return g;
    });

    const trails = rounds.map((rd, r) => rd.paths.map((pts, i) => add(svg, "path", { d: pathOf(pts), fill: "none", stroke: WORKER[i], "stroke-width": r === 0 ? 3 : 2, "stroke-linecap": "round", "stroke-linejoin": "round" })));
    const syncs = rounds.map((rd) => rd.paths.map((pts) => add(svg, "line", { x1: pts[K].x, y1: pts[K].y, x2: rd.avg.x, y2: rd.avg.y, stroke: C.ctrl, "stroke-width": 1.8, "stroke-dasharray": "5 6" })));
    [...minima, star].forEach((g) => svg.appendChild(g));
    const ends0 = rounds[0].paths.map((p) => p[K]);
    const avg0 = rounds[0].avg;
    const tri = add(svg, "path", { d: `${pathOf(ends0)}Z`, fill: "rgba(255,209,102,0.08)", stroke: C.warn, "stroke-width": 2, "stroke-dasharray": "8 6" });
    const triLabel = label(svg, avg0.x, avg0.y + 6, "client drift", { fill: C.warn, "font-size": 34, ...halo });

    const workers = WORKER.slice(0, 3).map((c) => {
      const g = group(svg);
      add(g, "circle", { r: 13, fill: c, stroke: "#000", "stroke-width": 2.5 });
      gsap.set(g, { x: start.x, y: start.y });
      return g;
    });
    const startLabel = label(svg, start.x + 34, start.y - 6, "misma réplica en los tres", { "text-anchor": "start", fill: C.text2, "font-size": 31 });
    const avgLabel = label(svg, avg0.x, avg0.y - 38, "promedio", { fill: C.ctrl, "font-size": 31 });

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
      if (s === 1) label(g, S.x, S.y + 1, "S", { fill: C.text, "font-size": 28 });
      W.forEach((w, k) => {
        add(g, "circle", { cx: w.x, cy: w.y, r, fill: C.panel, stroke: C.ar, "stroke-width": 2 });
        if (s === 1) label(g, w.x, w.y + 1, `W${k}`, { fill: C.text2, "font-size": 27 });
      });
      return { s, S: { x: S.x, y: S.y + 28 * s }, W: W.map((w) => ({ x: w.x, y: w.y - r })) };
    }

    function ringTopo(g, cx, cy, R, r) {
      const angs = [-90, 0, 90, 180];
      angs.forEach((a) => add(g, "path", { d: arcPath(cx, cy, R, a + 18, a + 72), fill: "none", stroke: C.lineStrong, "stroke-width": 2, "marker-end": svg.ref("arrow-line") }));
      angs.forEach((a, k) => {
        const p = { x: cx + R * Math.cos((a * Math.PI) / 180), y: cy + R * Math.sin((a * Math.PI) / 180) };
        add(g, "circle", { cx: p.x, cy: p.y, r, fill: C.panel, stroke: C.ar, "stroke-width": 2 });
        if (r > 20) label(g, p.x, p.y + 1, `W${k}`, { fill: C.text2, "font-size": 27 });
      });
      return { cx, cy, R, angs, s: r / 26 };
    }

    const cards = specs.map((c, i) => {
      const x0 = 10 + i * 420;
      const cx = x0 + 200;
      const cy = 270;
      const g = group(svg);
      add(g, "rect", { x: x0, y: 10, width: 400, height: 520, rx: 8, fill: C.panel, stroke: c.color, "stroke-width": 2.2 });
      label(g, cx, 62, c.name, { fill: C.text, "font-size": 30 });
      label(g, cx, 482, c.kw, { fill: c.color, "font-size": 28 });
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
    label(banner, 840, 586, "O.N.O.: las cuatro, sobre la misma base, para compararlas", { fill: C.accent, "font-size": 30 });

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
    const svg = svgRoot(container, 1100, 680, [18, 28, 1064, 625]);
    const tl = gsap.timeline();
    const FS = 31;
    const busY = 486;
    const host = group(svg);
    add(host, "rect", { x: 30, y: 40, width: 1040, height: 520, rx: 10, fill: "none", stroke: C.lineStrong, "stroke-width": 2, "stroke-dasharray": "10 8" });
    label(host, 60, 80, "una máquina física", { "text-anchor": "start", fill: C.text2, "font-size": FS });
    const bus = add(svg, "line", { x1: 70, y1: busY, x2: 1030, y2: busY, stroke: C.accent, "stroke-width": 3 });
    const busLabel = label(svg, 1030, busY + 38, "red bridge de Docker", { "text-anchor": "end", fill: C.text2, "font-size": FS });
    const coreLabel = label(svg, 70, busY + 38, "1 núcleo por contenedor", { "text-anchor": "start", fill: C.text2, "font-size": FS });

    const nodes = Array.from({ length: 16 }, (_, i) => {
      const row = i < 8 ? 0 : 1;
      const x = 100 + (i % 8) * 120 + row * 60;
      const y = row ? 290 : 130;
      const pipe = add(svg, "line", { x1: x, y1: y + 90, x2: x, y2: busY, stroke: C.lineStrong, "stroke-width": 7 });
      const g = group(svg);
      add(g, "rect", { x: x - 50, y, width: 100, height: 90, rx: 5, fill: C.panel, stroke: C.ar, "stroke-width": 2 });
      label(g, x, y + 46, "node", { fill: C.text, "font-size": FS });
      return { g, pipe, x, y };
    });
    const counts = [2, 4, 6, 8, 12, 16];
    const ctr = counts.map((n) => label(svg, 1040, 80, `${n} workers`, { "text-anchor": "end", fill: C.accent, "font-size": 34 }));
    const limit = label(svg, 550, 624, "salida limitada a 10 Mbps por contenedor (Pumba)", { fill: C.warn, "font-size": FS });

    show(tl, [host, bus, busLabel, coreLabel], 0.05, 0.4);
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
        travel(a, g, { x: n.x, y: n.y + 90 }, { x: n.x, y: busY }, j * 0.35, 1.8);
      });
      return a;
    };
    return controller(tl, 2, ambientFor);
  }

  function tcost(container) {
    const svg = svgRoot(container, 1100, 600, [9, -1, 1079, 612]);
    const tl = gsap.timeline();
    const X0 = 210, W = 170, SYNC = 80, INIT = 60, H = 46;
    const lanes = [214, 286, 358, 430];
    const dim = C.lineStrong;

    const formula = add(svg, "text", { x: 550, y: 46, "text-anchor": "middle", "dominant-baseline": "middle", "font-size": 48, fill: C.text });
    const term = (parts) => {
      const t = add(formula, "tspan");
      for (const [str, kind] of parts) {
        const s = add(t, "tspan", kind === "sub" ? { "baseline-shift": "sub", "font-size": "0.62em" } : {});
        s.textContent = str;
      }
      return t;
    };
    term([["T"], ["total", "sub"], [" = "]]);
    const epochsT = term([["N"], ["epochs", "sub"], [" · ("]]);
    const splitT = term([["T"], ["epoch", "sub"], ["/N"], ["W", "sub"]]);
    const syncT = term([[" + d"], ["sync", "sub"]]);
    const closeT = term([[")"]]);
    const initT = term([[" + d"], ["init", "sub"]]);
    gsap.set([epochsT, splitT, syncT, closeT, initT], { fill: dim });

    const machine = label(svg, X0 - 22, 322, "1 máquina", { "text-anchor": "end", fill: C.text2, "font-size": 31 });
    const laneLabels = lanes.map((y, k) => label(svg, X0 - 22, y, `W${k + 1}`, { "text-anchor": "end", fill: WORKER[k], "font-size": 31 }));

    const round = group(svg);
    const segs = lanes.map((_, k) => add(round, "rect", { x: X0 + k * W, y: 322 - H / 2, width: W, height: H, rx: 3, fill: C.accent, stroke: "#000", "stroke-width": 2 }));
    const epochLabel = richLabel(svg, X0 + 2 * W, 322 - H - 4, [["T"], ["epoch", "sub"]], { fill: C.accent, "font-size": 34 });
    const splitLabel = richLabel(round, X0 + W / 2, 156, [["T"], ["epoch", "sub"], ["/N"], ["W", "sub"]], { fill: C.accent, "font-size": 32 });
    const barrier = add(round, "line", { x1: X0 + W, y1: 178, x2: X0 + W, y2: 460, stroke: C.warn, "stroke-width": 2, "stroke-dasharray": "6 6" });
    const syncs = lanes.map((y) => add(round, "rect", { x: X0 + W, y: y - H / 2, width: SYNC, height: H, rx: 3, fill: svg.ref("hatch"), stroke: C.warn, "stroke-width": 2 }));
    const syncLabel = richLabel(round, X0 + W + SYNC / 2 + 14, 156, [["d"], ["sync", "sub"]], { fill: C.warn, "font-size": 32 });

    const init = add(svg, "rect", { x: X0, y: lanes[0] - H / 2, width: INIT - 8, height: lanes[3] - lanes[0] + H, rx: 3, fill: C.panel, stroke: C.muted, "stroke-width": 2 });
    const initLabel = richLabel(svg, X0 + (INIT - 8) / 2 - 14, 156, [["d"], ["init", "sub"]], { fill: C.text2, "font-size": 32 });
    const ghosts = [1, 2].map((r) => {
      const g = group(svg);
      const x = X0 + INIT + r * (W + SYNC);
      lanes.forEach((y) => {
        add(g, "rect", { x, y: y - H / 2, width: W, height: H, rx: 3, fill: C.accent, opacity: 0.75 });
        add(g, "rect", { x: x + W, y: y - H / 2, width: SYNC, height: H, rx: 3, fill: svg.ref("hatch"), stroke: C.warn, "stroke-width": 2 });
      });
      return g;
    });
    const epochs = [0, 1, 2].map((r) => label(svg, X0 + INIT + r * (W + SYNC) + (W + SYNC) / 2, 488, `epoch ${r + 1}`, { fill: C.text2, "font-size": 31 }));
    const totalEnd = X0 + INIT + 3 * (W + SYNC);
    const totalLine = add(svg, "line", { x1: X0, y1: 528, x2: totalEnd, y2: 528, stroke: C.text2, "stroke-width": 2, "marker-end": svg.ref("arrow-line") });
    const totalLabel = richLabel(svg, (X0 + totalEnd) / 2, 568, [["T"], ["total", "sub"]], { fill: C.text, "font-size": 34 });

    show(tl, formula, 0.05, 0.4);
    gsap.set(segs, { scaleX: 0, transformOrigin: "0% 50%" });
    gsap.set([...laneLabels, splitLabel, barrier, ...syncs, syncLabel, init, initLabel, ...ghosts, ...epochs, totalLine, totalLabel], { autoAlpha: 0 });
    show(tl, [machine, epochLabel], 0.2, 0.35);
    tl.to(segs, { scaleX: 1, duration: 0.22, stagger: 0.22, ease: "none" }, 0.3);
    tl.addLabel("s0");

    tl.to([machine, epochLabel], { autoAlpha: 0, duration: 0.25 }, "s0");
    segs.forEach((s, k) => tl.to(s, { attr: { x: X0, y: lanes[k] - H / 2 }, duration: 0.7, ease: "power2.inOut" }, `s0+=${0.1 + k * 0.08}`));
    tl.to(laneLabels, { autoAlpha: 1, duration: 0.3, stagger: 0.05 }, "s0+=0.5");
    tl.to(splitLabel, { autoAlpha: 1, duration: 0.3 }, "s0+=0.8");
    tl.to(splitT, { fill: C.accent, duration: 0.4 }, "s0+=0.8");
    tl.addLabel("s1");

    tl.set(barrier, { autoAlpha: 1 }, "s1");
    draw(tl, barrier, "s1", 0.4);
    gsap.set(syncs, { scaleX: 0, transformOrigin: "0% 50%" });
    tl.set(syncs, { autoAlpha: 1 }, "s1+=0.3");
    tl.to(syncs, { scaleX: 1, duration: 0.5, stagger: 0.05, ease: "power1.out" }, "s1+=0.3");
    tl.to(syncLabel, { autoAlpha: 1, duration: 0.3 }, "s1+=0.6");
    tl.to(syncT, { fill: C.warn, duration: 0.4 }, "s1+=0.6");
    tl.addLabel("s2");

    tl.to(round, { x: INIT, duration: 0.5, ease: "power2.inOut" }, "s2");
    gsap.set(init, { scaleY: 0, transformOrigin: "50% 50%" });
    tl.set(init, { autoAlpha: 1 }, "s2+=0.35");
    tl.to(init, { scaleY: 1, duration: 0.35 }, "s2+=0.35");
    tl.to(initLabel, { autoAlpha: 1, duration: 0.3 }, "s2+=0.5");
    tl.to(initT, { fill: C.text2, duration: 0.4 }, "s2+=0.5");
    ghosts.forEach((g, k) => tl.fromTo(g, { autoAlpha: 0, x: -30 }, { autoAlpha: 1, x: 0, duration: 0.4, immediateRender: false }, `s2+=${0.8 + k * 0.35}`));
    tl.to(epochs, { autoAlpha: 1, duration: 0.3, stagger: 0.2 }, "s2+=0.8");
    tl.to([epochsT, closeT], { fill: C.text, duration: 0.4 }, "s2+=1.3");
    tl.set(totalLine, { autoAlpha: 1 }, "s2+=1.4");
    draw(tl, totalLine, "s2+=1.4", 0.6);
    tl.to(totalLabel, { autoAlpha: 1, duration: 0.3 }, "s2+=1.8");
    tl.addLabel("s3");
    return controller(tl, 3);
  }


  function dsyncwhy(container) {
    const svg = svgRoot(container, 1100, 760, [-2, -1, 1104, 753]);
    const tl = gsap.timeline();
    const N = 4;
    const FS = 31;

    add(svg, "line", { x1: 550, y1: 20, x2: 550, y2: 568, stroke: C.line, "stroke-width": 1.5 });
    const psTitle = label(svg, 275, 36, "Parameter Server", { fill: C.text, "font-size": 34 });
    const arTitle = label(svg, 825, 36, "Ring All-Reduce", { fill: C.text, "font-size": 34 });

    const psG = group(svg);
    const nic = { x: 275, y: 196 };
    const wk = [0, 1, 2, 3].map((k) => ({ x: 275 + (k - 1.5) * 118, y: 440 }));
    wk.forEach((w) => add(psG, "line", { x1: w.x, y1: w.y - 30, x2: nic.x, y2: nic.y + 16, stroke: C.lineStrong, "stroke-width": 1.5 }));
    add(psG, "rect", { x: 175, y: 88, width: 200, height: 70, rx: 6, fill: C.panel, stroke: C.ps, "stroke-width": 2.5 });
    label(psG, 275, 124, "servidor", { fill: C.text, "font-size": FS });
    const nicBox = add(psG, "rect", { x: nic.x - 40, y: nic.y - 16, width: 80, height: 32, rx: 4, fill: "#1d1a08", stroke: C.warn, "stroke-width": 2 });
    label(psG, nic.x - 54, nic.y + 1, "placa · R", { "text-anchor": "end", fill: C.text2, "font-size": FS });
    wk.forEach((w, k) => {
      add(psG, "rect", { x: w.x - 44, y: w.y - 30, width: 88, height: 60, rx: 6, fill: C.panel, stroke: WORKER[k], "stroke-width": 2 });
      label(psG, w.x, w.y + 1, `W${k + 1}`, { fill: C.text, "font-size": FS });
    });

    const arG = group(svg);
    const RC = { x: 825, y: 280 }, RR = 150;
    const ang = [-90, 0, 90, 180];
    const pos = ang.map((a) => ({ x: RC.x + RR * Math.cos((a * Math.PI) / 180), y: RC.y + RR * Math.sin((a * Math.PI) / 180) }));
    add(arG, "circle", { cx: RC.x, cy: RC.y, r: RR, fill: "none", stroke: C.lineStrong, "stroke-width": 1.5 });
    pos.forEach((p, k) => {
      add(arG, "circle", { cx: p.x, cy: p.y, r: 36, fill: C.panel, stroke: WORKER[k], "stroke-width": 2 });
      label(arG, p.x, p.y + 1, `W${k + 1}`, { fill: C.text, "font-size": FS });
    });

    const psCount = Array.from({ length: 2 * N }, (_, k) => label(svg, 275, 540, `${k + 1} × S/R`, { fill: C.warn, "font-size": 36 }));
    const arCount = Array.from({ length: 2 * (N - 1) }, (_, k) => label(svg, 825, 540, `${k + 1} × S/(4R)`, { fill: C.ar, "font-size": 36 }));

    function formula(cx, parts) {
      const g = group(svg);
      add(g, "rect", { x: cx - 265, y: 588, width: 530, height: 84, fill: C.panel, stroke: C.lineStrong, "stroke-width": 1.5 });
      add(g, "rect", { x: cx - 265, y: 588, width: 6, height: 84, fill: C.accent });
      richLabel(g, cx + 3, 632, parts, { fill: C.text, "font-size": 42 });
      return g;
    }
    const psFormula = formula(275, [["d"], ["sync", "sub"], [" = 2·N"], ["W", "sub"], ["·S/R"]]);
    const arFormula = formula(825, [["d"], ["sync", "sub"], [" = 2(1−1/N"], ["W", "sub"], [")·S/R"]]);
    const psGrow = richLabel(svg, 275, 712, [["crece con N"], ["W", "sub"]], { fill: C.warn, "font-size": FS });
    const arBound = label(svg, 825, 712, "nunca supera 2S/R", { fill: C.accent, "font-size": FS });

    show(tl, [psTitle, psG, arTitle, arG], 0.05, 0.4, { stagger: 0.08 });
    gsap.set([...psCount, psFormula, psGrow, ...arCount, arFormula, arBound], { autoAlpha: 0 });
    tl.addLabel("s0");

    const T1 = tl.duration();
    const slots = [0, 1, 2, 3].map((k) => ({ x: nic.x, y: nic.y + 48 + k * 32 }));
    const ups = wk.map((w, k) => {
      const g = gradPacket(svg, 11);
      add(g, "circle", { r: 5, fill: WORKER[k] });
      gsap.set(g, { autoAlpha: 0, x: w.x, y: w.y - 36 });
      tl.set(g, { autoAlpha: 1, x: w.x, y: w.y - 36 }, T1 + 0.05);
      tl.to(g, { x: slots[k].x, y: slots[k].y, duration: 0.6, ease: "power1.inOut" }, T1 + 0.05);
      return g;
    });
    const tick = (labels, i, at) => {
      tl.set(labels[i], { autoAlpha: 1 }, at);
      if (i > 0) tl.set(labels[i - 1], { autoAlpha: 0 }, at);
    };
    ups.forEach((g, i) => {
      const at = T1 + 0.85 + i * 0.4;
      tl.to(g, { y: 124, duration: 0.3, ease: "power1.in" }, at);
      tl.set(g, { autoAlpha: 0 }, at + 0.32);
      ups.slice(i + 1).forEach((h, j) => tl.to(h, { y: slots[j].y, duration: 0.25 }, at + 0.05));
      tick(psCount, i, at + 0.15);
      tl.to(nicBox, { attr: { fill: C.warn }, duration: 0.1, yoyo: true, repeat: 1 }, at + 0.1);
    });
    const T2 = T1 + 0.85 + N * 0.4 + 0.1;
    wk.forEach((w, i) => {
      const at = T2 + i * 0.4;
      const p = paramPacket(svg, 18);
      travelPath(tl, p, [{ x: 275, y: 124 }, nic, { x: w.x, y: w.y - 36 }], at, 0.55);
      tick(psCount, N + i, at + 0.15);
      tl.to(nicBox, { attr: { fill: C.warn }, duration: 0.1, yoyo: true, repeat: 1 }, at + 0.1);
    });
    tl.to(psFormula, { autoAlpha: 1, duration: 0.4 }, T2 + N * 0.4 + 0.4);
    tl.to(psGrow, { autoAlpha: 1, duration: 0.4 }, T2 + N * 0.4 + 0.7);
    tl.addLabel("s1");

    const T3 = tl.duration();
    for (let hop = 0; hop < 2 * (N - 1); hop++) {
      const at = T3 + 0.1 + hop * 0.55;
      ang.forEach((a) => {
        const pkt = hop < N - 1 ? gradPacket(svg, 8) : paramPacket(svg, 14);
        travelPath(tl, pkt, arcPoints(RC.x, RC.y, RR, a, a + 90, 8), at, 0.45);
      });
      tick(arCount, hop, at + 0.3);
    }
    const T4 = T3 + 0.1 + 2 * (N - 1) * 0.55;
    tl.to(arFormula, { autoAlpha: 1, duration: 0.4 }, T4 + 0.2);
    tl.to(arBound, { autoAlpha: 1, duration: 0.4 }, T4 + 0.5);
    tl.addLabel("s2");
    return controller(tl, 2);
  }


  function im2col(container) {
    const svg = svgRoot(container, 1100, 760, [48, 2, 1061, 738]);
    const tl = gsap.timeline();
    const FS = 31;
    const IN = 60, CELL = 44;
    const IX = 60, IY = 196, KX = 60, KY = 498;
    const BX = 720, BY = 180, AX = 304, RY = 626, OX = 960, OY = 640;

    const cell = (parent, x, y, s, stroke, text, size) => {
      add(parent, "rect", { x, y, width: s, height: s, fill: C.panel, stroke, "stroke-width": 1.5 });
      return text ? label(parent, x + s / 2, y + s / 2, text, { fill: C.text2, "font-size": size }) : null;
    };
    const center = (x, y, s) => ({ x: x + s / 2, y: y + s / 2 });

    const input = group(svg);
    label(input, IX, IY - 30, "entrada 4×4", { "text-anchor": "start", fill: C.text2, "font-size": FS });
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) cell(input, IX + c * IN, IY + r * IN, IN, C.lineStrong, String(r * 4 + c + 1), 26);

    const kernel = group(svg);
    label(kernel, KX, KY - 30, "kernel 3×3", { "text-anchor": "start", fill: C.text2, "font-size": FS });
    const kCells = [];
    for (let i = 0; i < 9; i++) {
      const x = KX + (i % 3) * IN, y = KY + Math.floor(i / 3) * IN;
      add(kernel, "rect", { x, y, width: IN, height: IN, fill: C.panel, stroke: C.param, "stroke-width": 1.5 });
      richLabel(kernel, x + IN / 2, y + IN / 2, [["w"], [String(i + 1), "sub"]], { fill: C.param, "font-size": 26 });
      kCells.push(center(x, y, IN));
    }

    const fft = group(svg);
    add(fft, "rect", { x: 390, y: 14, width: 320, height: 64, rx: 6, fill: C.panel, stroke: C.muted, "stroke-width": 1.5 });
    label(fft, 550, 47, "1º intento: FFT", { fill: C.text2, "font-size": FS });
    const strike = add(svg, "line", { x1: 404, y1: 47, x2: 696, y2: 47, stroke: C.bad, "stroke-width": 4 });
    const fftNo = group(svg);
    label(fftNo, 550, 112, "más overhead que ganancia", { fill: C.bad, "font-size": FS });

    const colG = group(svg);
    label(colG, BX + 4 * CELL + 18, BY + 22, "im2col", { "text-anchor": "start", fill: C.text2, "font-size": FS });
    label(colG, BX + 4 * CELL + 18, BY + 62, "9 × 4", { "text-anchor": "start", fill: C.text2, "font-size": FS });
    const colVals = [];
    for (let p = 0; p < 4; p++) {
      const r0 = p >> 1, c0 = p & 1;
      colVals.push(Array.from({ length: 9 }, (_, i) => {
        const r = r0 + Math.floor(i / 3), c = c0 + (i % 3);
        const text = String(r * 4 + c + 1);
        add(colG, "rect", { x: BX + p * CELL, y: BY + i * CELL, width: CELL, height: CELL, fill: C.panel, stroke: C.line, "stroke-width": 1.5 });
        const t = label(colG, BX + p * CELL + CELL / 2, BY + i * CELL + CELL / 2, text, { fill: C.text2, "font-size": 22 });
        return { t, text, from: center(IX + c * IN, IY + r * IN, IN), to: { x: BX + p * CELL + CELL / 2, y: BY + i * CELL + CELL / 2 } };
      }));
    }
    const win = add(svg, "rect", { x: IX, y: IY, width: 3 * IN, height: 3 * IN, rx: 4, fill: "none", stroke: C.accent, "stroke-width": 4 });

    const rowG = group(svg);
    for (let i = 0; i < 9; i++) cell(rowG, AX + i * CELL, RY, CELL, C.param);
    const rowVals = Array.from({ length: 9 }, (_, i) => richLabel(svg, AX + i * CELL + CELL / 2, RY + CELL / 2, [["w"], [String(i + 1), "sub"]], { fill: C.param, "font-size": 22 }));
    const rowCap = label(svg, AX + 4.5 * CELL, RY + CELL + 34, "kernel como fila 1×9", { fill: C.text2, "font-size": FS });
    const colHi = add(svg, "rect", { x: BX, y: BY, width: CELL, height: 9 * CELL, fill: "none", stroke: C.accent, "stroke-width": 3 });
    const rowHi = add(svg, "rect", { x: AX, y: RY, width: 9 * CELL, height: CELL, fill: "none", stroke: C.accent, "stroke-width": 3 });
    const res = [0, 1, 2, 3].map((j) => {
      const g = group(svg);
      add(g, "rect", { x: BX + j * CELL, y: RY, width: CELL, height: CELL, fill: "#0f2219", stroke: C.accent, "stroke-width": 1.5 });
      richLabel(g, BX + j * CELL + CELL / 2, RY + CELL / 2, [["y"], [String(j + 1), "sub"]], { fill: C.accent, "font-size": 22 });
      return g;
    });
    const out = group(svg);
    label(out, OX + CELL, OY - 28, "salida 2×2", { fill: C.text2, "font-size": FS });
    [0, 1, 2, 3].forEach((j) => {
      const x = OX + (j & 1) * CELL, y = OY + (j >> 1) * CELL;
      add(out, "rect", { x, y, width: CELL, height: CELL, fill: "#0f2219", stroke: C.accent, "stroke-width": 1.5 });
      richLabel(out, x + CELL / 2, y + CELL / 2, [["y"], [String(j + 1), "sub"]], { fill: C.accent, "font-size": 22 });
    });

    show(tl, [input, kernel], 0.05, 0.4, { stagger: 0.1 });
    show(tl, fft, 0.3, 0.4);
    gsap.set([strike, fftNo, colG, win, rowG, ...rowVals, rowCap, colHi, rowHi, ...res, out], { autoAlpha: 0 });
    gsap.set(colVals.flat().map((cv) => cv.t), { autoAlpha: 0 });
    tl.addLabel("s0");

    tl.set(strike, { autoAlpha: 1 }, "s0");
    draw(tl, strike, "s0", 0.4);
    tl.to(fftNo, { autoAlpha: 1, duration: 0.3 }, "s0+=0.35");
    tl.to(fft, { autoAlpha: 0.45, duration: 0.3 }, "s0+=0.35");
    tl.addLabel("s1");

    const T1 = tl.duration();
    tl.to(colG, { autoAlpha: 1, duration: 0.3 }, T1);
    colVals.forEach((col, p) => {
      const at = T1 + 0.3 + p * 0.9;
      if (p === 0) tl.set(win, { autoAlpha: 1 }, at);
      tl.to(win, { attr: { x: IX + (p & 1) * IN, y: IY + (p >> 1) * IN }, duration: 0.25 }, at);
      col.forEach((cv, i) => {
        const ghost = label(svg, 0, 0, cv.text, { fill: C.accent, "font-size": 26 });
        travel(tl, ghost, cv.from, cv.to, at + 0.25 + i * 0.04, 0.45);
        tl.set(cv.t, { autoAlpha: 1 }, at + 0.72 + i * 0.04);
      });
    });
    tl.to(win, { autoAlpha: 0, duration: 0.25 }, T1 + 0.3 + 4 * 0.9);
    tl.addLabel("s2");

    const T2 = tl.duration();
    tl.to(rowG, { autoAlpha: 1, duration: 0.3 }, T2);
    rowVals.forEach((rv, i) => {
      const ghost = richLabel(svg, 0, 0, [["w"], [String(i + 1), "sub"]], { fill: C.param, "font-size": 24 });
      travel(tl, ghost, kCells[i], { x: AX + i * CELL + CELL / 2, y: RY + CELL / 2 }, T2 + 0.1 + i * 0.05, 0.6);
      tl.set(rv, { autoAlpha: 1 }, T2 + 0.72 + i * 0.05);
    });
    tl.to(rowCap, { autoAlpha: 1, duration: 0.3 }, T2 + 1.1);
    tl.set(rowHi, { autoAlpha: 1 }, T2 + 1.3);
    res.forEach((g, j) => {
      const at = T2 + 1.3 + j * 0.4;
      tl.set(colHi, { autoAlpha: 1, attr: { x: BX + j * CELL } }, at);
      tl.fromTo(g, { autoAlpha: 0, scale: 0.6, transformOrigin: "50% 50%" }, { autoAlpha: 1, scale: 1, duration: 0.3, immediateRender: false }, at + 0.1);
    });
    tl.to([colHi, rowHi], { autoAlpha: 0, duration: 0.25 }, T2 + 1.3 + 4 * 0.4);
    tl.fromTo(out, { autoAlpha: 0, x: -24 }, { autoAlpha: 1, x: 0, duration: 0.4, immediateRender: false }, T2 + 1.3 + 4 * 0.4 + 0.1);
    tl.addLabel("s3");
    return controller(tl, 3);
  }

  function agenda(container) {
    const svg = svgRoot(container, 1640, 460, [42, 63, 1575, 329]);
    const tl = gsap.timeline();
    const Y = 230;
    const items = [
      ["motivación", "Alejo"],
      ["sincronización", "Marcos"],
      ["arquitectura", "Marcos y Alejo"],
      ["costo de sincronización", "Lorenzo"],
      ["resultados", "Lorenzo"],
      ["demostración", "los tres"],
    ];
    const xs = items.map((_, k) => 150 + k * 268);
    const line = add(svg, "line", { x1: 80, y1: Y, x2: 1560, y2: Y, stroke: C.lineStrong, "stroke-width": 3 });
    const stops = items.map(([topic, who], k) => {
      const g = group(svg);
      const side = k % 2 === 0 ? -1 : 1;
      add(g, "line", { x1: xs[k], y1: Y, x2: xs[k], y2: Y + side * 58, stroke: C.line, "stroke-width": 2 });
      add(g, "circle", { cx: xs[k], cy: Y, r: 31, fill: C.panel, stroke: C.accent, "stroke-width": 3 });
      label(g, xs[k], Y + 1, String(k + 1).padStart(2, "0"), { fill: C.accent, "font-size": 28 });
      label(g, xs[k], Y + side * 88, topic, { fill: C.text, "font-size": 32 });
      label(g, xs[k], Y + side * 134, who, { fill: C.text2, "font-size": 28 });
      return g;
    });
    const dot = add(svg, "circle", { cx: 0, cy: 0, r: 9, fill: C.accent });

    tl.set(line, { autoAlpha: 1 }, 0);
    drawLinear(tl, line, 0.05, 1.6);
    gsap.set(stops, { autoAlpha: 0, y: 12 });
    gsap.set(dot, { autoAlpha: 0, x: 80, y: Y });
    tl.set(dot, { autoAlpha: 1 }, 0.05);
    tl.to(dot, { x: 1560, duration: 1.6, ease: "none" }, 0.05);
    stops.forEach((g, k) => tl.to(g, { autoAlpha: 1, y: 0, duration: 0.35, ease: "back.out(2)" }, 0.05 + ((xs[k] - 80) / 1480) * 1.6));
    tl.to(dot, { autoAlpha: 0, duration: 0.2 }, 1.7);
    tl.addLabel("s0");
    return controller(tl, 0);
  }


  function blackbox(container) {
    const svg = svgRoot(container, 1100, 640, [-1, -57, 1102, 669]);
    const tl = gsap.timeline();
    const BX = 380, BY = 120, BW = 340, BH = 240, MID = BY + BH / 2;
    const FS = 31;

    const input = group(svg);
    add(input, "rect", { x: 60, y: MID - 90, width: 180, height: 180, rx: 8, fill: C.panel, stroke: C.lineStrong, "stroke-width": 2 });
    add(input, "path", { d: `M100,${MID - 50} L200,${MID - 50} L138,${MID + 62}`, fill: "none", stroke: C.text, "stroke-width": 14, "stroke-linecap": "round", "stroke-linejoin": "round" });
    label(input, 150, MID + 122, "entrada", { fill: C.text2, "font-size": FS });
    const arrows = [
      add(svg, "line", { x1: 254, y1: MID, x2: BX - 14, y2: MID, stroke: C.lineStrong, "stroke-width": 3, "marker-end": svg.ref("arrow-line") }),
      add(svg, "line", { x1: BX + BW + 14, y1: MID, x2: 860, y2: MID, stroke: C.lineStrong, "stroke-width": 3, "marker-end": svg.ref("arrow-line") }),
    ];

    const box = add(svg, "rect", { x: BX, y: BY, width: BW, height: BH, rx: 10, fill: "#030504", stroke: C.lineStrong, "stroke-width": 3 });
    const boxLabel = label(svg, BX + BW / 2, BY - 28, "modelo", { fill: C.text2, "font-size": FS });
    const question = label(svg, BX + BW / 2, MID + 4, "?", { fill: C.muted, "font-size": 110 });

    const knobs = [];
    const knobG = group(svg);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 8; c++) {
        const x = BX + 40 + c * 37, y0 = BY + 30 + r * 66, y1 = y0 + 48;
        add(knobG, "line", { x1: x, y1: y0, x2: x, y2: y1, stroke: C.line, "stroke-width": 3, "stroke-linecap": "round" });
        const k = add(knobG, "circle", { cx: x, cy: y0 + 24, r: 7, fill: C.accent });
        knobs.push({ k, y0 });
      }
    }

    const outRight = label(svg, 960, MID + 6, "7", { fill: C.accent, "font-size": 120 });
    const outWrong = label(svg, 960, MID + 6, "3", { fill: C.bad, "font-size": 120 });
    label(svg, 960, MID + 122, "salida", { fill: C.text2, "font-size": FS });
    const expected = label(svg, 960, BY - 28, "esperado: 7", { fill: C.text2, "font-size": FS });
    const error = add(svg, "path", { d: `M960,${BY - 54} C960,${BY - 124} ${BX + BW + 30},${BY - 124} ${BX + BW - 30},${BY - 6}`, fill: "none", stroke: C.warn, "stroke-width": 3, "stroke-dasharray": "8 7", "marker-end": svg.ref("arrow-muted") });
    const errorLabel = label(svg, 875, BY - 142, "error", { fill: C.warn, "font-size": FS });

    const llm = group(svg);
    const LY = 556;
    const ask = "«¿qué es un modelo?»";
    const reply = "«una función que…»";
    const wAsk = ask.length * 0.6 * FS + 28;
    const wReply = reply.length * 0.6 * FS + 28;
    const wGpt = 7 * 0.6 * 32 + 44;
    const arrowW = 44, gap = 12;
    let x = 550 - (wAsk + wGpt + wReply + 2 * (arrowW + 2 * gap)) / 2;
    const arrow = (x0) => add(llm, "line", { x1: x0, y1: LY, x2: x0 + arrowW, y2: LY, stroke: C.lineStrong, "stroke-width": 2.5, "marker-end": svg.ref("arrow-line") });
    add(llm, "rect", { x, y: LY - 38, width: wAsk, height: 76, rx: 38, fill: C.panel, stroke: C.lineStrong, "stroke-width": 1.5 });
    label(llm, x + wAsk / 2, LY + 1, ask, { fill: C.text2, "font-size": FS });
    x += wAsk + gap;
    arrow(x);
    x += arrowW + gap;
    add(llm, "rect", { x, y: LY - 44, width: wGpt, height: 88, rx: 8, fill: "#030504", stroke: C.lineStrong, "stroke-width": 2.5 });
    label(llm, x + wGpt / 2, LY + 1, "ChatGPT", { fill: C.text, "font-size": 32 });
    x += wGpt + gap;
    arrow(x);
    x += arrowW + gap;
    add(llm, "rect", { x, y: LY - 38, width: wReply, height: 76, rx: 38, fill: C.panel, stroke: C.lineStrong, "stroke-width": 1.5 });
    label(llm, x + wReply / 2, LY + 1, reply, { fill: C.text2, "font-size": FS });

    show(tl, input, 0.05, 0.4);
    show(tl, [box, boxLabel, question], 0.25, 0.4);
    gsap.set(arrows, { autoAlpha: 0 });
    tl.set(arrows[0], { autoAlpha: 1 }, 0.4);
    draw(tl, arrows[0], 0.4, 0.4);
    tl.set(arrows[1], { autoAlpha: 1 }, 0.8);
    draw(tl, arrows[1], 0.8, 0.4);
    show(tl, outRight, 1.1, 0.4);
    gsap.set([knobG, outWrong, expected, error, errorLabel, llm], { autoAlpha: 0 });
    tl.addLabel("s0");

    tl.to(question, { autoAlpha: 0, duration: 0.3 }, "s0");
    tl.to(box, { attr: { stroke: C.accent }, duration: 0.4 }, "s0");
    tl.to(knobG, { autoAlpha: 1, duration: 0.4 }, "s0+=0.2");
    tl.addLabel("s1");

    tl.to(outRight, { autoAlpha: 0, duration: 0.2 }, "s1");
    knobs.forEach(({ k, y0 }, i) => tl.to(k, { attr: { cy: y0 + ((i * 37) % 48) }, duration: 0.3 }, "s1"));
    tl.to(outWrong, { autoAlpha: 1, duration: 0.3 }, "s1+=0.3");
    tl.to(expected, { autoAlpha: 1, duration: 0.3 }, "s1+=0.6");
    tl.set(error, { autoAlpha: 1 }, "s1+=0.9");
    draw(tl, error, "s1+=0.9", 0.6);
    tl.to(errorLabel, { autoAlpha: 1, duration: 0.3 }, "s1+=1.1");
    knobs.forEach(({ k, y0 }, i) => tl.to(k, { attr: { cy: y0 + ((i * 53 + 11) % 48) }, duration: 0.5, ease: "power2.inOut" }, `s1+=${1.5 + (i % 8) * 0.04}`));
    tl.to(outWrong, { autoAlpha: 0, duration: 0.2 }, "s1+=2.2");
    tl.to(outRight, { autoAlpha: 1, duration: 0.3 }, "s1+=2.3");
    tl.addLabel("s2");

    tl.fromTo(llm, { autoAlpha: 0, y: 20 }, { autoAlpha: 1, y: 0, duration: 0.5, immediateRender: false }, "s2");
    tl.addLabel("s3");
    return controller(tl, 3);
  }

  function lineChart(container, spec) {
    const VW = 1000;
    const svg = svgRoot(container, VW, 700, [-9, -6, 991, 698]);
    const tl = gsap.timeline();
    const FS = 31;
    const entryW = (s) => 48 + 14 + s.name.length * 0.6 * FS;
    const GAP = 36;
    const rowW = (row) => row.reduce((sum, s) => sum + entryW(s) + GAP, -GAP);
    const perRow = rowW(spec.series) > VW - 20 ? Math.ceil(spec.series.length / 2) : spec.series.length;
    const rows = [];
    for (let i = 0; i < spec.series.length; i += perRow) rows.push(spec.series.slice(i, i + perRow));
    const X0 = 150, X1 = VW - 30, Y0 = 570, Y1 = 118 + rows.length * 48;
    const [xMin, xMax] = spec.x.domain;
    const [yMin, yMax] = spec.y.domain;
    const sx = (x) => X0 + ((x - xMin) / (xMax - xMin)) * (X1 - X0);
    const sy = (y) => Y0 - ((y - yMin) / (yMax - yMin)) * (Y0 - Y1);

    const head = group(svg);
    label(head, VW / 2, 32, spec.title, { fill: C.text, "font-size": 34 });
    rows.forEach((row, r) => {
      let x = VW / 2 - rowW(row) / 2;
      const y = 92 + r * 48;
      row.forEach((s) => {
        add(head, "line", { x1: x, y1: y, x2: x + 48, y2: y, stroke: s.color, "stroke-width": 5, ...(s.dash ? { "stroke-dasharray": "10 6" } : {}) });
        add(head, "circle", { cx: x + 24, cy: y, r: 8, fill: s.color, stroke: "#000", "stroke-width": 2 });
        label(head, x + 62, y + 1, s.name, { "text-anchor": "start", fill: C.text, "font-size": FS });
        x += entryW(s) + GAP;
      });
    });

    const axes = group(svg);
    spec.y.ticks.forEach((v) => {
      add(axes, "line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), stroke: C.line, "stroke-width": 1.5 });
      label(axes, X0 - 16, sy(v), spec.y.fmt(v), { "text-anchor": "end", fill: C.text2, "font-size": FS });
    });
    spec.x.ticks.forEach((v) => label(axes, sx(v), Y0 + 36, String(v), { fill: C.text2, "font-size": FS }));
    add(axes, "line", { x1: X0, y1: Y0, x2: X1, y2: Y0, stroke: C.lineStrong, "stroke-width": 2 });
    add(axes, "line", { x1: X0, y1: Y0, x2: X0, y2: Y1 - 14, stroke: C.lineStrong, "stroke-width": 2 });
    label(axes, (X0 + X1) / 2, Y0 + 92, spec.x.title, { fill: C.text2, "font-size": FS });
    label(axes, 26, (Y0 + Y1) / 2, spec.y.title, { fill: C.text2, "font-size": FS, transform: `rotate(-90 26 ${(Y0 + Y1) / 2})` });

    const curves = spec.series.map((s) => {
      const d = s.points.map(([x, y], k) => `${k ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
      const line = add(svg, "path", { d, fill: "none", stroke: s.color, "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round", ...(s.dash ? { "stroke-dasharray": "14 9" } : {}) });
      const marks = group(svg);
      s.points.forEach(([x, y]) => add(marks, "circle", { cx: sx(x), cy: sy(y), r: 8, fill: s.color, stroke: "#000", "stroke-width": 2 }));
      return { line, marks, dash: s.dash };
    });
    const note = spec.note ? group(svg) : null;
    if (note) spec.note.lines.forEach((text, k) => label(note, sx(spec.note.x), sy(spec.note.y) + k * 42, text, { "text-anchor": "end", fill: C.text2, "font-size": FS }));

    show(tl, [head, axes], 0.05, 0.4);
    curves.forEach((c, i) => {
      const at = 0.3 + i * 0.25;
      if (c.dash) show(tl, c.line, at, 0.6);
      else { tl.set(c.line, { autoAlpha: 1 }, at); draw(tl, c.line, at, 0.9); }
      show(tl, c.marks, at + 0.6, 0.3);
    });
    if (note) show(tl, note, 0.9 + curves.length * 0.25, 0.3);
    tl.addLabel("s0");
    return controller(tl, 0);
  }

  const WORKER_AXIS = { domain: [1, 17], ticks: [2, 4, 6, 8, 12, 16], title: "cantidad de workers" };
  const TIME_AXIS = { domain: [200, 700], ticks: [200, 300, 400, 500, 600, 700], fmt: String, title: "tiempo (s)" };
  const percent = (v) => `${Math.round(v * 100)} %`;

  function synctime(container) {
    return lineChart(container, {
      title: "Tiempo total de entrenamiento",
      x: WORKER_AXIS,
      y: TIME_AXIS,
      series: [
        { name: "All-Reduce", color: C.ar, points: [[2, 653.8], [4, 374.5], [6, 291.5], [8, 295.3], [12, 290.9], [16, 288.4]] },
        { name: "Parameter Server", color: C.ps, points: [[2, 642.8], [4, 399.6], [6, 341.2], [8, 366.3], [12, 382.9], [16, 405.3]] },
      ],
    });
  }

  function asynctime(container) {
    return lineChart(container, {
      title: "Tiempo total de entrenamiento",
      x: WORKER_AXIS,
      y: TIME_AXIS,
      series: [
        { name: "PS asincrónico", color: C.psHi, dash: true, points: [[2, 437.7], [4, 298.6], [6, 276.2], [8, 256.6], [12, 247.8], [16, 229.0]] },
        { name: "Strategy-Switch", color: C.ss, points: [[2, 417.6], [4, 294.7], [6, 271.0], [8, 266.5], [12, 259.2], [16, 240.8]] },
      ],
    });
  }

  function accuracy(container) {
    return lineChart(container, {
      title: "Accuracy final sobre el conjunto de prueba",
      x: WORKER_AXIS,
      y: { domain: [0, 1], ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], fmt: percent, title: "accuracy" },
      series: [
        { name: "All-Reduce", color: C.ar, points: [[2, 0.9836], [4, 0.9821], [6, 0.9821], [8, 0.9834], [12, 0.9714], [16, 0.9704]] },
        { name: "Parameter Server", color: C.ps, points: [[2, 0.9836], [4, 0.9821], [6, 0.9821], [8, 0.9833], [12, 0.9807], [16, 0.9798]] },
        { name: "PS asincrónico", color: C.psHi, dash: true, points: [[2, 0.9564], [4, 0.8645], [6, 0.4734], [8, 0.0777], [12, 0.1073], [16, 0.1065]] },
        { name: "Strategy-Switch", color: C.ss, points: [[2, 0.9836], [4, 0.9821], [6, 0.9821], [8, 0.9805], [12, 0.9681], [16, 0.9665]] },
      ],
      note: { x: 16, y: 0.84, lines: ["AR, PS y Strategy-Switch", "superpuestas arriba"] },
    });
  }

  function oeacc(container) {
    const byE = [
      [0.9841, 0.9823, 0.9817, 0.9804, 0.9737],
      [0.9739, 0.969, 0.9658, 0.9632, 0.9513],
      [0.9625, 0.9546, 0.9483, 0.9436, 0.9358],
      [0.9515, 0.9423, 0.9337, 0.9255, 0.9174],
      [0.9426, 0.9323, 0.9244, 0.9177, 0.9076],
    ];
    const colors = [C.accent, "#b5e655", C.grad, "#ff9248", C.bad];
    return lineChart(container, {
      title: "All-Reduce: accuracy final según E",
      x: { domain: [1.4, 10.6], ticks: [2, 4, 6, 8, 10], title: "cantidad de nodos" },
      y: { domain: [0.9, 1], ticks: [0.9, 0.92, 0.94, 0.96, 0.98, 1], fmt: percent, title: "accuracy" },
      series: byE.map((vals, e) => ({ name: `E = ${e}`, color: colors[e], points: vals.map((v, k) => [2 + 2 * k, v]) })),
    });
  }

  const BUILDERS = { syncasync, arch, flatbuffer, ps, ring, dsync, sswindow, ssflow, compress, topo, cost, scaleup, scaleout, parallelism, drift, alternatives, testbed, tcost, dsyncwhy, im2col, agenda, blackbox, synctime, asynctime, accuracy, oeacc };

  window.ONO_DIAGRAMS = {
    build(container) {
      const fn = BUILDERS[container.dataset.diagram];
      if (!fn) return null;
      return fn(container);
    },
  };
})();
