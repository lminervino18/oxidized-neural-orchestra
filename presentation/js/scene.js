/*
 * Subtle Three.js backdrop for the cover: a tilted ring of nodes orbiting
 * behind the logo, with occasional messages travelling between neighbours.
 * Renders only while its slide is visible and motion is allowed.
 */
(function () {
  const ACCENT = 0x0af886;
  const NODES = 8;
  const RADIUS = 3.3;

  function webglAvailable() {
    try {
      const probe = document.createElement("canvas");
      return !!(probe.getContext("webgl2") || probe.getContext("webgl"));
    } catch (e) {
      return false;
    }
  }

  function createScene(container) {
    if (!window.THREE || !webglAvailable()) return null;
    const THREE = window.THREE;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (e) {
      return null;
    }
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    camera.position.set(0, 0, 10);

    const orbit = new THREE.Group();
    orbit.rotation.x = 1.18;
    orbit.rotation.z = -0.12;
    scene.add(orbit);

    const ringPts = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(Math.cos(a) * RADIUS, Math.sin(a) * RADIUS, 0));
    }
    const ring = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(ringPts),
      new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.16 })
    );
    orbit.add(ring);

    const outerPts = [];
    for (let i = 0; i < 180; i++) {
      const a = (i / 180) * Math.PI * 2;
      outerPts.push(new THREE.Vector3(Math.cos(a) * RADIUS * 1.22, Math.sin(a) * RADIUS * 1.22, 0));
    }
    const outer = new THREE.Points(
      new THREE.BufferGeometry().setFromPoints(outerPts),
      new THREE.PointsMaterial({ color: ACCENT, size: 0.025, transparent: true, opacity: 0.22 })
    );
    orbit.add(outer);

    const nodeGeo = new THREE.SphereGeometry(0.075, 16, 16);
    const nodes = [];
    for (let i = 0; i < NODES; i++) {
      const a = (i / NODES) * Math.PI * 2;
      const mesh = new THREE.Mesh(nodeGeo, new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.55 }));
      mesh.position.set(Math.cos(a) * RADIUS, Math.sin(a) * RADIUS, 0);
      orbit.add(mesh);
      nodes.push({ mesh, angle: a });
    }

    const pulseGeo = new THREE.SphereGeometry(0.05, 12, 12);
    const pulses = [0, 1, 2].map((k) => {
      const mesh = new THREE.Mesh(pulseGeo, new THREE.MeshBasicMaterial({ color: 0xffc24a, transparent: true, opacity: 0 }));
      orbit.add(mesh);
      return { mesh, from: k * 3, t: -k * 0.9 };
    });

    let running = false;
    let raf = 0;
    let last = 0;

    function resize() {
      const scale = (window.devicePixelRatio || 1) * (window.ONO_STAGE_SCALE || 1);
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      renderer.setPixelRatio(Math.min(scale, 2.5));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    }

    function step(dt) {
      orbit.rotation.z += dt * 0.045;
      outer.rotation.z -= dt * 0.02;
      for (const p of pulses) {
        p.t += dt * 0.42;
        if (p.t >= 1.6) {
          p.t = 0;
          p.from = (p.from + 2 + Math.floor(Math.random() * 3)) % NODES;
        }
        const k = Math.max(0, Math.min(1, p.t));
        const a = nodes[p.from].angle + k * ((Math.PI * 2) / NODES);
        p.mesh.position.set(Math.cos(a) * RADIUS, Math.sin(a) * RADIUS, 0);
        p.mesh.material.opacity = p.t > 0 && p.t < 1 ? Math.sin(k * Math.PI) * 0.9 : 0;
        if (p.t >= 1 && p.t < 1.25) nodes[(p.from + 1) % NODES].mesh.material.opacity = 0.95 - (p.t - 1) * 1.6;
      }
      for (const n of nodes) n.mesh.material.opacity += (0.55 - n.mesh.material.opacity) * Math.min(1, dt * 3);
    }

    function loop(now) {
      if (!running) return;
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
      last = now;
      step(dt);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    }

    return {
      start() {
        resize();
        if (running) return;
        running = true;
        last = 0;
        raf = requestAnimationFrame(loop);
      },
      stop() {
        running = false;
        cancelAnimationFrame(raf);
      },
      renderStatic() {
        this.stop();
        for (const p of pulses) p.mesh.material.opacity = 0;
        resize();
      },
      resize,
    };
  }

  window.ONO_SCENE = { createScene };
})();
