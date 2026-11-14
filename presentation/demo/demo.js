/*
 * Demo page: loads a .safetensors saved by orchestui, runs the network in the
 * browser, reports its accuracy on the MNIST test set and classifies digits
 * drawn with the mouse. Layer semantics follow machine_learning: conv weights
 * are [filters, channels, k, k] (cross-correlation), dense weights [in, out].
 */
(function () {
  const data = window.ONO_DEMO;
  const $ = (id) => document.getElementById(id);
  const ui = {
    drop: $("drop"), file: $("file"), loaded: $("loaded"), status: $("status"), results: $("results"),
    acc: $("acc"), accValue: $("acc-value"), accTotal: $("acc-total"), accBar: $("acc-bar"), accCount: $("acc-count"),
    pad: $("pad"), seen: $("seen"), digit: $("digit"), truth: $("truth"), probs: $("probs"),
    clear: $("clear"), sample: $("sample"),
  };
  const SIDE = 28;
  const BOX = 20;

  let net = null;
  let test = null;
  let run = 0;

  const fmtInt = (n) => n.toLocaleString("es-AR");
  const fmtPct = (v, digits = 2) => (v * 100).toFixed(digits).replace(".", ",");
  const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));


  function setStatus(text, error = false) {
    ui.status.textContent = text;
    ui.status.classList.toggle("error", error);
  }

  function base64Bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  async function gunzip(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  async function loadTest() {
    if (test) return test;
    const raw = await gunzip(base64Bytes(data.test));
    const n = data.count;
    test = { n, images: raw.subarray(0, n * SIDE * SIDE), labels: raw.subarray(n * SIDE * SIDE, n * SIDE * SIDE + n) };
    return test;
  }


  function parseSafetensors(buffer) {
    const size = Number(new DataView(buffer).getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, size)));
    const base = 8 + size;
    const tensors = {};
    for (const [name, meta] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      if (meta.dtype !== "F32") throw new Error(`${name} es ${meta.dtype}; se esperaba F32`);
      const [start, end] = meta.data_offsets;
      tensors[name] = { shape: meta.shape, data: new Float32Array(buffer.slice(base + start, base + end)) };
    }
    return tensors;
  }

  function outDim(size, kernel, stride, padding) {
    return Math.floor((size + 2 * padding - kernel) / stride) + 1;
  }

  function plan(model) {
    let shape = null;
    let flat = null;
    return model.layers.map((layer, i) => {
      const [kind, cfg] = Object.entries(layer)[0];
      const act = cfg.act_fn ?? null;
      if (kind === "conv") {
        const [c, h, w] = cfg.input_dim;
        const [f, , k] = cfg.kernel_dim;
        const ho = outDim(h, k, cfg.stride, cfg.padding);
        const wo = outDim(w, k, cfg.stride, cfg.padding);
        shape = [f, ho, wo];
        return { kind, i, act, c, h, w, f, k, stride: cfg.stride, pad: cfg.padding, ho, wo, wShape: [f, c, k, k], bShape: [f] };
      }
      if (kind === "max_pooling") {
        const [c, h, w] = cfg.input_dim;
        const size = cfg.filter_size;
        const ho = outDim(h, size, cfg.stride, cfg.padding);
        const wo = outDim(w, size, cfg.stride, cfg.padding);
        shape = [c, ho, wo];
        return { kind, i, act, c, h, w, size, stride: cfg.stride, pad: cfg.padding, ho, wo };
      }
      const inSize = flat ?? (shape ? shape[0] * shape[1] * shape[2] : SIDE * SIDE);
      flat = cfg.output_size;
      return { kind: "dense", i, act, inSize, out: cfg.output_size, wShape: [inSize, cfg.output_size], bShape: [cfg.output_size] };
    });
  }

  function bind(steps, tensors) {
    const same = (a, b) => a.length === b.length && a.every((v, k) => v === b[k]);
    let count = 0;
    for (const step of steps) {
      if (!step.wShape) continue;
      const weight = tensors[`layer_${step.i}.weight`];
      const bias = tensors[`layer_${step.i}.bias`];
      if (!weight || !bias) throw new Error(`falta layer_${step.i} en el archivo`);
      if (!same(weight.shape, step.wShape)) throw new Error(`layer_${step.i}.weight mide [${weight.shape}]; esta arquitectura espera [${step.wShape}]`);
      if (!same(bias.shape, step.bShape)) throw new Error(`layer_${step.i}.bias mide [${bias.shape}]; esta arquitectura espera [${step.bShape}]`);
      step.W = weight.data;
      step.B = bias.data;
      count += weight.data.length + bias.data.length;
    }
    return count;
  }


  function conv(s, x) {
    const { c, h, w, f, k, stride, pad, ho, wo, W, B } = s;
    const hp = h + 2 * pad, wp = w + 2 * pad;
    let src = x;
    if (pad) {
      src = new Float32Array(c * hp * wp);
      for (let ch = 0; ch < c; ch++) {
        for (let y = 0; y < h; y++) src.set(x.subarray((ch * h + y) * w, (ch * h + y + 1) * w), (ch * hp + y + pad) * wp + pad);
      }
    }
    const out = new Float32Array(f * ho * wo);
    for (let o = 0; o < f; o++) {
      for (let y = 0; y < ho; y++) {
        for (let z = 0; z < wo; z++) {
          let sum = B[o];
          for (let ch = 0; ch < c; ch++) {
            for (let ky = 0; ky < k; ky++) {
              const xRow = (ch * hp + y * stride + ky) * wp + z * stride;
              const wRow = ((o * c + ch) * k + ky) * k;
              for (let kx = 0; kx < k; kx++) sum += src[xRow + kx] * W[wRow + kx];
            }
          }
          out[(o * ho + y) * wo + z] = sum;
        }
      }
    }
    return out;
  }

  function pool(s, x) {
    const out = new Float32Array(s.c * s.ho * s.wo);
    for (let ch = 0; ch < s.c; ch++) {
      for (let y = 0; y < s.ho; y++) {
        for (let z = 0; z < s.wo; z++) {
          let best = -Infinity;
          for (let ky = 0; ky < s.size; ky++) {
            const iy = y * s.stride + ky - s.pad;
            if (iy < 0 || iy >= s.h) continue;
            for (let kx = 0; kx < s.size; kx++) {
              const ix = z * s.stride + kx - s.pad;
              if (ix >= 0 && ix < s.w) best = Math.max(best, x[(ch * s.h + iy) * s.w + ix]);
            }
          }
          out[(ch * s.ho + y) * s.wo + z] = best;
        }
      }
    }
    return out;
  }

  function dense(s, x) {
    const out = Float32Array.from(s.B);
    for (let i = 0; i < s.inSize; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * s.out;
      for (let j = 0; j < s.out; j++) out[j] += xi * s.W[row + j];
    }
    return out;
  }

  function activate(x, act) {
    if (!act) return;
    if (act === "softmax") {
      let max = -Infinity;
      for (const v of x) max = Math.max(max, v);
      let sum = 0;
      for (let i = 0; i < x.length; i++) { x[i] = Math.exp(x[i] - max); sum += x[i]; }
      for (let i = 0; i < x.length; i++) x[i] /= sum;
      return;
    }
    const [name, cfg] = Object.entries(act)[0];
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      if (name === "sigmoid") x[i] = cfg.amp / (1 + Math.exp(-v));
      else if (name === "tanh") x[i] = cfg.amp * Math.tanh(v);
      else if (name === "relu") x[i] = Math.max(v, cfg.slope * v);
    }
  }

  function forward(steps, input) {
    let x = input;
    for (const step of steps) {
      x = step.kind === "conv" ? conv(step, x) : step.kind === "max_pooling" ? pool(step, x) : dense(step, x);
      activate(x, step.act);
    }
    return x;
  }

  function argmax(values) {
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
  }


  async function readFiles(files) {
    const list = Array.from(files);
    const weights = list.find((f) => f.name.endsWith(".safetensors"));
    const config = list.find((f) => f.name.endsWith(".json"));
    if (!weights) { setStatus("Falta el archivo .safetensors.", true); return; }
    try {
      const model = config ? JSON.parse(await config.text()) : data.model;
      const steps = plan(model);
      const count = bind(steps, parseSafetensors(await weights.arrayBuffer()));
      net = steps;
      showLoaded(weights.name, config ? config.name : "arquitectura de la demo", count, model.layers.length);
      setStatus("");
      await evaluate();
    } catch (err) {
      setStatus(`No se pudo cargar el modelo: ${err.message}`, true);
    }
  }

  function showLoaded(file, arch, count, layers) {
    ui.drop.hidden = true;
    ui.loaded.hidden = false;
    ui.loaded.innerHTML = "";
    const parts = [["modelo", file], ["parámetros", fmtInt(count)], ["capas", String(layers)], ["arquitectura", arch]];
    for (const [label, value] of parts) {
      const span = document.createElement("span");
      span.append(`${label}: `);
      const b = document.createElement("b");
      b.textContent = value;
      span.append(b);
      ui.loaded.append(span);
    }
    const change = document.createElement("button");
    change.className = "btn";
    change.type = "button";
    change.textContent = "cambiar archivo";
    change.addEventListener("click", () => ui.file.click());
    ui.loaded.append(change);
    ui.results.hidden = false;
    clearPad();
  }

  async function evaluate() {
    const id = ++run;
    const t = await loadTest();
    const input = new Float32Array(SIDE * SIDE);
    ui.acc.classList.add("pending");
    ui.accTotal.textContent = fmtInt(t.n);
    let ok = 0;
    for (let n = 0; n < t.n; n++) {
      if (id !== run) return;
      const offset = n * SIDE * SIDE;
      for (let p = 0; p < input.length; p++) input[p] = t.images[offset + p] / 255;
      if (argmax(forward(net, input)) === t.labels[n]) ok++;
      if ((n + 1) % 250 === 0 || n + 1 === t.n) {
        ui.accValue.textContent = fmtPct(ok / (n + 1));
        ui.accBar.style.width = `${((n + 1) / t.n) * 100}%`;
        ui.accCount.textContent = `${fmtInt(n + 1)} / ${fmtInt(t.n)} imágenes evaluadas`;
        await nextFrame();
      }
    }
    ui.acc.classList.remove("pending");
    ui.accCount.textContent = `${fmtInt(ok)} de ${fmtInt(t.n)} bien clasificadas`;
  }


  const pad = ui.pad.getContext("2d");
  const seen = ui.seen.getContext("2d");
  let drawing = false;
  let last = null;
  let pending = false;

  function clearPad() {
    pad.fillStyle = "#000";
    pad.fillRect(0, 0, ui.pad.width, ui.pad.height);
    seen.fillStyle = "#000";
    seen.fillRect(0, 0, SIDE, SIDE);
    showPrediction(null);
    ui.truth.textContent = "";
    ui.truth.className = "truth";
  }

  function padPoint(e) {
    const r = ui.pad.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * ui.pad.width, y: ((e.clientY - r.top) / r.height) * ui.pad.height };
  }

  function stroke(from, to) {
    pad.strokeStyle = "#fff";
    pad.fillStyle = "#fff";
    pad.lineWidth = ui.pad.width / 13;
    pad.lineCap = "round";
    pad.lineJoin = "round";
    pad.beginPath();
    pad.moveTo(from.x, from.y);
    pad.lineTo(to.x, to.y);
    pad.stroke();
  }

  function schedulePredict() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; predictDrawing(); });
  }

  function toMnist() {
    const { width, height } = ui.pad;
    const pixels = pad.getImageData(0, 0, width, height).data;
    let x0 = width, y0 = height, x1 = -1, y1 = -1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (pixels[(y * width + x) * 4] > 20) {
          x0 = Math.min(x0, x); x1 = Math.max(x1, x);
          y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        }
      }
    }
    if (x1 < 0) return null;
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    const scale = BOX / Math.max(bw, bh);
    const sw = Math.max(1, Math.round(bw * scale)), sh = Math.max(1, Math.round(bh * scale));
    const small = document.createElement("canvas");
    small.width = sw;
    small.height = sh;
    const sctx = small.getContext("2d");
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(ui.pad, x0, y0, bw, bh, 0, 0, sw, sh);
    const sp = sctx.getImageData(0, 0, sw, sh).data;
    let mass = 0, mx = 0, my = 0;
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const v = sp[(y * sw + x) * 4];
        mass += v; mx += v * (x + 0.5); my += v * (y + 0.5);
      }
    }
    const dx = Math.round(SIDE / 2 - mx / mass), dy = Math.round(SIDE / 2 - my / mass);
    const input = new Float32Array(SIDE * SIDE);
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const tx = x + dx, ty = y + dy;
        if (tx >= 0 && tx < SIDE && ty >= 0 && ty < SIDE) input[ty * SIDE + tx] = sp[(y * sw + x) * 4] / 255;
      }
    }
    return input;
  }

  function showSeen(input) {
    const img = seen.createImageData(SIDE, SIDE);
    for (let p = 0; p < input.length; p++) {
      const v = Math.round(input[p] * 255);
      img.data.set([v, v, v, 255], p * 4);
    }
    seen.putImageData(img, 0, 0);
  }

  function predictDrawing() {
    if (!net) return;
    const input = toMnist();
    ui.truth.textContent = "";
    ui.truth.className = "truth";
    if (!input) { clearPad(); return; }
    showSeen(input);
    showPrediction(forward(net, input));
  }

  function showPrediction(probs) {
    ui.probs.innerHTML = "";
    const best = probs ? argmax(probs) : -1;
    ui.digit.textContent = probs ? String(best) : "·";
    ui.digit.classList.toggle("empty", !probs);
    for (let d = 0; d < 10; d++) {
      const li = document.createElement("li");
      if (d === best) li.className = "best";
      const value = probs ? probs[d] : 0;
      li.innerHTML = `<span>${d}</span><span class="track"><span class="fill" style="display:block;width:${(value * 100).toFixed(1)}%"></span></span><span class="val">${probs ? fmtPct(value, 1) + " %" : ""}</span>`;
      ui.probs.append(li);
    }
  }

  async function randomSample() {
    if (!net) return;
    const t = await loadTest();
    const n = Math.floor(Math.random() * t.n);
    const input = new Float32Array(SIDE * SIDE);
    for (let p = 0; p < input.length; p++) input[p] = t.images[n * SIDE * SIDE + p] / 255;
    showSeen(input);
    pad.imageSmoothingEnabled = false;
    pad.drawImage(ui.seen, 0, 0, ui.pad.width, ui.pad.height);
    const probs = forward(net, input);
    showPrediction(probs);
    const ok = argmax(probs) === t.labels[n];
    ui.truth.textContent = `imagen ${fmtInt(n)} del test · era un ${t.labels[n]}`;
    ui.truth.className = `truth ${ok ? "ok" : "miss"}`;
  }


  ui.pad.addEventListener("pointerdown", (e) => {
    if (!net) return;
    drawing = true;
    ui.pad.setPointerCapture(e.pointerId);
    last = padPoint(e);
    stroke(last, last);
    schedulePredict();
  });
  ui.pad.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const point = padPoint(e);
    stroke(last, point);
    last = point;
    schedulePredict();
  });
  ["pointerup", "pointercancel"].forEach((type) => ui.pad.addEventListener(type, () => { drawing = false; }));
  ui.clear.addEventListener("click", clearPad);
  ui.sample.addEventListener("click", randomSample);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") clearPad(); });

  ui.file.addEventListener("change", () => { if (ui.file.files.length) readFiles(ui.file.files); ui.file.value = ""; });
  ["dragenter", "dragover"].forEach((type) => document.addEventListener(type, (e) => { e.preventDefault(); ui.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((type) => document.addEventListener(type, (e) => { e.preventDefault(); ui.drop.classList.remove("over"); }));
  document.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) readFiles(e.dataTransfer.files); });

  if (!data) setStatus("Falta data.js: generarlo con python3 tools/build_demo_data.py.", true);
  else if (!window.DecompressionStream) setStatus("Este navegador no soporta DecompressionStream; usar Chrome, Edge o Firefox actualizados.", true);
  else loadTest().catch((err) => setStatus(`No se pudo leer el test de MNIST: ${err.message}`, true));
})();
