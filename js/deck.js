/*
 * Slide engine: one slide per screen on a 1920×1080 stage scaled to fit,
 * keyboard/button navigation with intra-slide steps, presenter notes window,
 * pause and reduced-motion modes.
 */
(function () {
  const stage = document.getElementById("stage");
  const slides = Array.from(stage.querySelectorAll(".slide"));
  const mainCount = slides.filter((s) => !s.hasAttribute("data-appendix")).length;
  const ui = {
    counter: document.getElementById("counter"),
    section: document.getElementById("section-label"),
    flags: document.getElementById("flags"),
    bar: document.getElementById("progress-bar"),
    help: document.getElementById("help"),
    goto: document.getElementById("goto"),
    gotoList: document.getElementById("goto-list"),
    jump: document.getElementById("jump"),
    pause: document.getElementById("btn-pause"),
  };

  const reducedQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const state = { index: 0, step: 0, paused: false, reduced: reducedQuery.matches, started: false };
  const views = new Map();
  let notesWin = null;
  let notesStart = null;
  let jumpBuffer = "";
  let jumpTimer = null;

  window.ONO_MOTION = state.reduced ? "reduced" : "full";


  function view(slide) {
    if (views.has(slide)) return views.get(slide);
    const v = {
      diagrams: Array.from(slide.querySelectorAll("[data-diagram]")).map((el) => window.ONO_DIAGRAMS.build(el)).filter(Boolean),
      steppers: Array.from(slide.querySelectorAll("[data-step]")),
      captions: Array.from(slide.querySelectorAll("[data-step-caption]")),
      anims: Array.from(slide.querySelectorAll("[data-anim]")),
      tuis: Array.from(slide.querySelectorAll("[data-tui]")),
      floats: Array.from(slide.querySelectorAll("[data-float]")),
      sceneEl: slide.querySelector("[data-three]"),
      scene: null,
      player: null,
      floatTl: null,
      tuisDrawn: false,
    };
    const stepAttr = v.steppers.map((el) => Number(el.dataset.step));
    const capAttr = v.captions.map((el) => Number(el.dataset.stepCaption));
    v.steps = Math.max(0, ...stepAttr, ...capAttr, ...v.diagrams.map((d) => d.steps));

    const playerEl = slide.querySelector("[data-tui-player]");
    if (playerEl && window.ONO_TUI_VIEW.available) {
      v.player = window.ONO_TUI_VIEW.createPlayer(playerEl, {
        toggle: slide.querySelector("[data-player-toggle]"),
        speed: slide.querySelector("[data-player-speed]"),
        range: slide.querySelector("[data-player-range]"),
        time: slide.querySelector("[data-player-time]"),
      });
    }
    views.set(slide, v);
    return v;
  }

  function drawTuis(v, force) {
    if (!window.ONO_TUI_VIEW.available) return;
    if (v.tuisDrawn && !force) return;
    v.tuis.forEach((el) => window.ONO_TUI_VIEW.renderShot(el));
    if (v.player) v.player.redraw();
    v.tuisDrawn = true;
  }

  function instant() {
    return state.reduced || state.paused;
  }


  function applyStep(slide, step, animate) {
    const v = view(slide);
    const fx = animate && !instant();
    v.steppers.forEach((el) => {
      const s = Number(el.dataset.step);
      if (s <= step) {
        if (fx && s === step) gsap.fromTo(el, { autoAlpha: 0, y: 14 }, { autoAlpha: 1, y: 0, duration: 0.45, ease: "power2.out", overwrite: true });
        else gsap.set(el, { autoAlpha: 1, y: 0, overwrite: true });
      } else gsap.set(el, { autoAlpha: 0, overwrite: true });
    });
    v.captions.forEach((el) => {
      const s = Number(el.dataset.stepCaption);
      const target = s > step ? 0 : s === step ? 1 : 0.5;
      if (fx && s === step) gsap.fromTo(el, { autoAlpha: 0, x: -10 }, { autoAlpha: target, x: 0, duration: 0.35, overwrite: true });
      else gsap.set(el, { autoAlpha: target, x: 0, overwrite: true });
    });
    v.diagrams.forEach((d) => d.go(Math.min(step, d.steps), animate && !instant()));
  }


  function enter(slide, step, direction) {
    const v = view(slide);
    slide.classList.add("is-active");
    gsap.killTweensOf(slide);
    drawTuis(v);

    const fresh = direction >= 0 && step === 0;
    if (instant()) gsap.set(slide, { autoAlpha: 1 });
    else gsap.fromTo(slide, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.35, ease: "power1.out" });

    if (fresh && !instant()) {
      gsap.fromTo(v.anims, { autoAlpha: 0, y: 16 }, { autoAlpha: 1, y: 0, duration: 0.55, ease: "power2.out", stagger: 0.07, delay: 0.08, overwrite: true });
    } else gsap.set(v.anims, { autoAlpha: 1, y: 0, overwrite: true });

    applyStep(slide, step, fresh);
    startAmbient(slide);
  }

  function leave(slide) {
    const v = view(slide);
    v.diagrams.forEach((d) => d.leave());
    stopAmbient(slide);
    if (v.player) v.player.stop();
    gsap.killTweensOf(slide);
    if (instant()) {
      gsap.set(slide, { autoAlpha: 0 });
      slide.classList.remove("is-active");
    } else {
      gsap.to(slide, { autoAlpha: 0, duration: 0.22, ease: "power1.in", onComplete: () => slide.classList.remove("is-active") });
    }
  }

  function startAmbient(slide) {
    const v = view(slide);
    if (v.sceneEl && !v.scene) v.scene = window.ONO_SCENE.createScene(v.sceneEl);
    const moving = !state.reduced && !state.paused;
    if (v.scene) moving ? v.scene.start() : v.scene.renderStatic();
    if (v.floats.length && moving && !v.floatTl) {
      v.floatTl = gsap.timeline({ repeat: -1, repeatDelay: 1.6, delay: 0.8 });
      v.floatTl.to(v.floats, { y: -14, duration: 1.7, ease: "sine.inOut" })
        .to(v.floats, { y: 0, duration: 1.3, ease: "back.out(1.4)" });
    }
  }

  function stopAmbient(slide) {
    const v = view(slide);
    if (v.scene) v.scene.stop();
    if (v.floatTl) { v.floatTl.kill(); v.floatTl = null; gsap.set(v.floats, { y: 0 }); }
  }

  function show(index, step, direction) {
    index = Math.max(0, Math.min(slides.length - 1, index));
    const prev = slides[state.index];
    const next = slides[index];
    if (prev !== next || !state.started) {
      if (state.started && prev !== next) leave(prev);
      state.index = index;
      state.step = step === "last" ? view(next).steps : Math.min(step, view(next).steps);
      enter(next, state.step, direction);
      state.started = true;
    } else {
      state.step = step === "last" ? view(next).steps : Math.min(step, view(next).steps);
      applyStep(next, state.step, false);
    }
    updateChrome();
  }

  function next() {
    const v = view(slides[state.index]);
    if (state.step < v.steps) {
      state.step++;
      applyStep(slides[state.index], state.step, true);
      updateChrome();
    } else if (state.index < slides.length - 1) show(state.index + 1, 0, 1);
  }

  function prev() {
    if (state.step > 0) {
      state.step--;
      applyStep(slides[state.index], state.step, false);
      updateChrome();
    } else if (state.index > 0) show(state.index - 1, "last", -1);
  }

  function replay() {
    view(slides[state.index]).diagrams.forEach((d) => d.replay());
  }


  function label(i) {
    const s = slides[i];
    if (s.hasAttribute("data-appendix")) {
      const n = i - mainCount + 1;
      return `A${n} / A${slides.length - mainCount}`;
    }
    return `${String(i + 1).padStart(2, "0")} / ${mainCount}`;
  }

  function updateChrome() {
    const slide = slides[state.index];
    const v = view(slide);
    ui.counter.textContent = label(state.index);
    const stepTxt = v.steps ? `  ·  etapa ${state.step}/${v.steps}` : "";
    ui.section.innerHTML = `<b>${slide.dataset.section || ""}</b>  ·  ${slide.dataset.title || ""}${stepTxt}`;
    const progress = Math.min(state.index + 1, mainCount) / mainCount;
    ui.bar.style.width = `${progress * 100}%`;
    const flags = [];
    if (state.paused) flags.push("⏸ animaciones en pausa");
    if (state.reduced) flags.push("movimiento reducido");
    ui.flags.textContent = flags.join("  ·  ");
    ui.pause.setAttribute("aria-pressed", String(state.paused));
    const hash = `#${state.index + 1}${state.step ? "." + state.step : ""}`;
    if (location.hash !== hash) history.replaceState(null, "", hash);
    updateNotes();
  }

  function fit() {
    const gutter = window.innerWidth < 700 ? 32 : 0;
    const s = Math.min((window.innerWidth - gutter) / 1920, window.innerHeight / 1080);
    window.ONO_STAGE_SCALE = s;
    stage.style.transform = `scale(${s})`;
    stage.style.left = `${(window.innerWidth - 1920 * s) / 2}px`;
    stage.style.top = `${(window.innerHeight - 1080 * s) / 2}px`;
  }

  let resizeTimer = null;
  function onResize() {
    fit();
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      views.forEach((v) => { v.tuisDrawn = false; if (v.scene) v.scene.resize(); });
      drawTuis(view(slides[state.index]), true);
    }, 150);
  }


  function togglePause() {
    state.paused = !state.paused;
    if (state.paused) gsap.globalTimeline.pause();
    else gsap.globalTimeline.resume();
    const v = view(slides[state.index]);
    if (v.scene) state.paused ? v.scene.stop() : (!state.reduced && v.scene.start());
    updateChrome();
  }

  function setReduced(on) {
    state.reduced = on;
    window.ONO_MOTION = on ? "reduced" : "full";
    document.body.classList.toggle("reduced", on);
    const slide = slides[state.index];
    stopAmbient(slide);
    applyStep(slide, state.step, false);
    view(slide).diagrams.forEach((d) => d.motionChanged());
    startAmbient(slide);
    updateChrome();
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
  }

  function toggleOverlay(el, force) {
    const open = force !== undefined ? force : !el.classList.contains("open");
    [ui.help, ui.goto].forEach((o) => o.classList.remove("open"));
    if (open) {
      el.classList.add("open");
      if (el === ui.goto) buildGoto();
    }
  }

  function buildGoto() {
    ui.gotoList.innerHTML = "";
    slides.forEach((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      const num = s.hasAttribute("data-appendix") ? `A${i - mainCount + 1}` : String(i + 1).padStart(2, "0");
      b.innerHTML = `<span>${num}</span>${s.dataset.title}`;
      if (i === state.index) b.classList.add("current");
      b.addEventListener("click", () => { toggleOverlay(ui.goto, false); show(i, 0, i >= state.index ? 1 : -1); });
      ui.gotoList.appendChild(b);
    });
    const cur = ui.gotoList.querySelector(".current");
    if (cur) cur.focus();
  }


  function pushDigit(d) {
    jumpBuffer += d;
    ui.jump.textContent = `ir a ${jumpBuffer}`;
    ui.jump.classList.add("open");
    clearTimeout(jumpTimer);
    jumpTimer = setTimeout(clearJump, 2500);
  }

  function clearJump() {
    jumpBuffer = "";
    ui.jump.classList.remove("open");
  }

  function commitJump() {
    const n = parseInt(jumpBuffer, 10);
    clearJump();
    if (n >= 1 && n <= slides.length) show(n - 1, 0, n - 1 >= state.index ? 1 : -1);
  }


  function parseTime(t) {
    const [m, s] = (t || "0:00").split(":").map(Number);
    return m * 60 + (s || 0);
  }

  function fmt(sec) {
    const m = Math.floor(sec / 60);
    return `${String(m).padStart(2, "0")}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
  }

  function notesFor(slide) {
    const n = slide.querySelector(".notes");
    const pick = (sel) => (n && n.querySelector(sel) ? n.querySelector(sel).innerHTML : "");
    return { idea: pick(".n-idea"), oral: pick(".n-oral"), trans: pick(".n-trans"), src: pick(".n-src") };
  }

  function openNotes() {
    if (notesWin && !notesWin.closed) { notesWin.focus(); return; }
    notesWin = window.open("", "ono-notas", "width=1100,height=820");
    if (!notesWin) return;
    notesStart = notesStart || Date.now();
    notesWin.document.write(`<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Notas · O.N.O.</title><style>
      body{margin:0;background:#050806;color:#e3eee6;font:18px/1.5 "IBM Plex Sans",system-ui,sans-serif;padding:28px 34px}
      .top{display:flex;gap:28px;align-items:baseline;font-family:"JetBrains Mono",monospace;color:#a7bcae;font-size:16px;flex-wrap:wrap}
      .top b{color:#0af886;font-weight:500}.clock{font-size:34px;color:#e3eee6;margin-left:auto}
      h1{font:500 28px "JetBrains Mono",monospace;margin:18px 0 6px}
      .idea{font-size:22px;color:#0af886;margin:10px 0 16px}
      .oral p{margin:0 0 10px}.label{font-family:"JetBrains Mono",monospace;font-size:13px;color:#6a8072;letter-spacing:.06em;text-transform:uppercase;margin-top:18px}
      .trans{color:#ffd166}.src{color:#6a8072;font-family:"JetBrains Mono",monospace;font-size:14px}
      .next{margin-top:22px;padding-top:14px;border-top:1px solid #1f4d31;color:#a7bcae}
      code{color:#00be00}button{background:none;border:1px solid #1f4d31;color:#a7bcae;font:14px "JetBrains Mono",monospace;padding:4px 10px;cursor:pointer}
    </style></head><body><div id="root"></div></body></html>`);
    notesWin.document.close();
    notesWin.document.addEventListener("keydown", (e) => onKey(e));
    notesWin.setInterval(() => updateNotes(), 1000);
    updateNotes();
  }

  function updateNotes() {
    if (!notesWin || notesWin.closed) return;
    const slide = slides[state.index];
    const v = view(slide);
    const n = notesFor(slide);
    const planned = slides.slice(0, state.index).filter((s) => !s.hasAttribute("data-appendix")).reduce((acc, s) => acc + parseTime(s.dataset.time), 0);
    const elapsed = notesStart ? (Date.now() - notesStart) / 1000 : 0;
    const nextSlide = slides[state.index + 1];
    const root = notesWin.document.getElementById("root");
    root.innerHTML = `
      <div class="top"><span><b>${label(state.index)}</b> · etapa ${state.step}/${v.steps}</span>
        <span>expositor: <b>${slide.dataset.speaker || "—"}</b></span>
        <span>tiempo sugerido: <b>${slide.dataset.time || "—"}</b></span>
        <span>plan acumulado al iniciar: <b>${fmt(planned)}</b></span>
        <span class="clock">${fmt(elapsed)}</span><button id="reset">reiniciar reloj</button></div>
      <h1>${slide.dataset.title}</h1>
      <div class="idea">${n.idea}</div>
      <div class="label">desarrollo oral</div><div class="oral">${n.oral}</div>
      <div class="label">transición</div><div class="trans">${n.trans}</div>
      <div class="label">respaldo</div><div class="src">${n.src}</div>
      <div class="next">siguiente: ${nextSlide ? `${nextSlide.dataset.title} · ${nextSlide.dataset.speaker || ""}` : "—"}</div>`;
    root.querySelector("#reset").onclick = () => { notesStart = Date.now(); updateNotes(); };
  }


  function onKey(e) {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    const k = e.key;
    if (/^[0-9]$/.test(k)) { pushDigit(k); e.preventDefault(); return; }
    if (k === "Enter" && jumpBuffer) { commitJump(); e.preventDefault(); return; }
    if (k === "Escape") {
      if (jumpBuffer) clearJump();
      toggleOverlay(ui.help, false);
      return;
    }
    if (ui.goto.classList.contains("open") && k !== "g" && k !== "G") return;
    switch (k) {
      case "ArrowRight": case "PageDown": case " ":
        e.preventDefault();
        if (e.shiftKey || k === "PageDown") show(state.index + 1, 0, 1); else next();
        break;
      case "ArrowLeft": case "PageUp":
        e.preventDefault();
        if (e.shiftKey || k === "PageUp") show(state.index - 1, "last", -1); else prev();
        break;
      case "Home": show(0, 0, -1); break;
      case "End": show(mainCount - 1, 0, 1); break;
      case "f": case "F": toggleFullscreen(); break;
      case "p": case "P": togglePause(); break;
      case "m": case "M": setReduced(!state.reduced); break;
      case "r": case "R": replay(); break;
      case "n": case "N": openNotes(); break;
      case "g": case "G": toggleOverlay(ui.goto); break;
      case "?": case "h": case "H": toggleOverlay(ui.help); break;
      default: return;
    }
  }

  document.addEventListener("keydown", onKey);
  document.getElementById("btn-prev").addEventListener("click", prev);
  document.getElementById("btn-next").addEventListener("click", next);
  document.getElementById("btn-pause").addEventListener("click", togglePause);
  document.getElementById("btn-full").addEventListener("click", toggleFullscreen);
  document.getElementById("btn-help").addEventListener("click", () => toggleOverlay(ui.help));
  [ui.help, ui.goto].forEach((o) => o.addEventListener("click", (e) => { if (e.target === o) toggleOverlay(o, false); }));
  reducedQuery.addEventListener("change", (e) => setReduced(e.matches));
  window.addEventListener("resize", onResize);

  let idleTimer = null;
  document.addEventListener("mousemove", () => {
    document.body.classList.remove("hide-cursor");
    clearTimeout(idleTimer);
    if (document.fullscreenElement) idleTimer = setTimeout(() => document.body.classList.add("hide-cursor"), 2500);
  });

  function fromHash() {
    const m = location.hash.match(/^#(\d+)(?:\.(\d+))?$/);
    if (!m) return [0, 0];
    return [Math.max(0, Number(m[1]) - 1), Number(m[2] || 0)];
  }

  window.addEventListener("hashchange", () => {
    const [i, s] = fromHash();
    if (i !== state.index || s !== state.step) show(i, s, i >= state.index ? 1 : -1);
  });

  fit();
  document.body.classList.toggle("reduced", state.reduced);
  const boot = () => {
    const [i, s] = fromHash();
    show(i, s, s === 0 ? 1 : -1);
  };
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(boot);

  window.ONO_DECK = { show, next, prev, state, slides, view };
})();
