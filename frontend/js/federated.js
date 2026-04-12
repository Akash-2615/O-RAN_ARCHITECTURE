/**
 * federated.js — Federated Learning Visualization
 * Animated network graph: BS agents → coordinator → global model.
 * Edge thickness = attention weight.
 * All driven from live WebSocket data.
 */

window.Federated = (() => {
  let canvas, ctx;
  let animId;
  let fedState = {
    weights:      [],
    performances: [],
    round:        0,
    interval:     50,
    tti:          0,
    last_event:   null,
  };
  let particleFlows = [];  // animated particles on edges
  let fedRound = false;    // flash on new round

  // 3D Subsystem (Global Brain)
  let threeScene, threeCamera, threeRenderer, threeCore, threeSats = [];
  let threeClock = new THREE.Clock();

  function createTextTexture(text, color = '#ffffff') {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256; canvas.height = 256; // Higher res
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 4;
    ctx.fillStyle = color;
    ctx.font = 'bold 80px Inter'; // Bigger font
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    // Add text stroke for extreme clarity
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 8;
    ctx.strokeText(text, 128, 128);
    ctx.fillText(text, 128, 128);
    const tex = new THREE.CanvasTexture(canvas);
    return tex;
  }

  const COLORS = {
    ric:    '#ffb300',
    agent:  '#0088ff',
    edge:   '#00d4ff',
    packet: '#ffffff',
    bg:     '#050a14',
  };

  // ── Layout ──────────────────────────────────────────────────────── //
  function getLayout(n, cw, ch) {
    const cx = cw / 2, cy = ch / 2;
    const r  = Math.min(cw, ch) * 0.40;
    const agents = Array.from({length: n}, (_, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
      return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), id: i };
    });
    return { coord: { x: cx, y: cy }, agents };
  }

  // ── Particle flows ──────────────────────────────────────────────── //
  function spawnParticles(from, to, color, n = 8) {
    for (let i = 0; i < n; i++) {
      particleFlows.push({
        x: from.x, y: from.y,
        tx: to.x,  ty: to.y,
        t:  -(i / n),  // staggered start
        color, speed: 0.015 + Math.random() * 0.01,
        size: 2 + Math.random() * 2,
      });
    }
  }

  function updateParticles() {
    particleFlows = particleFlows.filter(p => {
      p.t += p.speed;
      return p.t < 1.2;
    });
  }

  function drawParticles() {
    particleFlows.forEach(p => {
      if (p.t < 0 || p.t > 1) return;
      const x = p.x + (p.tx - p.x) * p.t;
      const y = p.y + (p.ty - p.y) * p.t;
      ctx.beginPath();
      ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.color + 'cc';
      ctx.shadowColor = p.color;
      ctx.shadowBlur  = 6;
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  }

  function draw() {
    if (!canvas || !ctx) return;
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    if (W === 0 || H === 0) return;
    ctx.clearRect(0, 0, W, H);

    const n = Math.max(fedState.weights.length, 3);
    const { coord, agents } = getLayout(n, W, H);
    const weights = fedState.weights.length ? fedState.weights : Array(n).fill(1/n);

    // ── Edges: agent → coordinator ──────────────────────────────── //
    agents.forEach((ag, i) => {
      const w = weights[i] || (1 / n);
      const lineW = 1 + w * 12;
      const alpha = (0.15 + w * 0.7).toFixed(2);

      // Edge line
      ctx.beginPath();
      ctx.moveTo(ag.x, ag.y);
      ctx.lineTo(coord.x, coord.y);
      ctx.strokeStyle = `rgba(0,212,255,${alpha})`;
      ctx.lineWidth   = lineW;
      ctx.shadowColor = '#00d4ff';
      ctx.shadowBlur  = lineW * 2;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Weight label on edge: position at 70% distance from center
      const d = 0.70;
      const lx = coord.x + (ag.x - coord.x) * d;
      const ly = coord.y + (ag.y - coord.y) * d;
      
      // Calculate perpendicular offset for better clearance
      const angle = Math.atan2(ag.y - coord.y, ag.x - coord.x);
      // Increase offset to 18px and use a background box for clarity
      const perpX = Math.cos(angle + Math.PI/2) * 22;
      const perpY = Math.sin(angle + Math.PI/2) * 22;

      // Draw small glow background for the weight
      ctx.fillStyle = 'rgba(5, 10, 25, 0.7)';
      const labelW = 40;
      const labelH = 16;
      ctx.beginPath();
      ctx.roundRect(lx + perpX - labelW/2, ly + perpY - labelH/2, labelW, labelH, 4);
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,212,255,0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#00d4ff';
      ctx.font      = 'bold 12px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(w.toFixed(3), lx + perpX, ly + perpY);
    });

    // ── Coordinator node ─────────────────────────────────────────── //
    const sr = 50;
    const cGrad = ctx.createRadialGradient(coord.x, coord.y, 0, coord.x, coord.y, sr);
    cGrad.addColorStop(0, '#ffb30088');
    cGrad.addColorStop(1, '#ffb30011');
    ctx.beginPath();
    ctx.arc(coord.x, coord.y, sr, 0, Math.PI * 2);
    ctx.fillStyle = cGrad;
    ctx.fill();
    ctx.strokeStyle = '#ffb300';
    ctx.lineWidth   = 2;
    ctx.shadowColor = '#ffb300';
    ctx.shadowBlur  = 20;
    ctx.stroke(); ctx.shadowBlur = 0;
    ctx.fillStyle   = '#ffb300';
    ctx.font        = 'bold 13px Inter';
    ctx.textAlign   = 'center';
    ctx.textBaseline= 'middle';
    ctx.fillText('GLOBAL', coord.x, coord.y - 8);
    ctx.fillText('RIC', coord.x, coord.y + 8);
    ctx.font = '11px JetBrains Mono';
    ctx.fillStyle = 'rgba(255,179,0,0.85)';
    ctx.fillText(`Rnd ${fedState.round}`, coord.x, coord.y + 24);

    // ── Agent nodes ──────────────────────────────────────────────── //
    agents.forEach((ag, i) => {
      const w    = weights[i] || (1 / n);
      const perf = fedState.performances[i] || 0;
      const nor  = Math.max(0, Math.min(1, (perf + 0.5) / 1.5));

      const r = 30 + w * 25;
      const grd = ctx.createRadialGradient(ag.x, ag.y, 0, ag.x, ag.y, r);
      grd.addColorStop(0, `rgba(0,136,255,${0.3 + w * 0.5})`);
      grd.addColorStop(1, 'rgba(0,136,255,0.02)');
      ctx.beginPath();
      ctx.arc(ag.x, ag.y, r, 0, Math.PI * 2);
      ctx.fillStyle   = grd;
      ctx.fill();
      ctx.strokeStyle = `rgba(0,212,255,${0.4 + w * 0.6})`;
      ctx.lineWidth   = 2;
      ctx.shadowColor = '#00d4ff';
      ctx.shadowBlur  = 10 + w * 15;
      ctx.stroke(); ctx.shadowBlur = 0;

      ctx.fillStyle    = '#e8f4ff';
      ctx.font         = 'bold 14px Inter';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`BS-${ag.id}`, ag.x, ag.y - 6);
      ctx.fillStyle = 'rgba(0,212,255,0.9)';
      ctx.font = '12px JetBrains Mono';
      ctx.fillText(perf.toFixed(3), ag.x, ag.y + 11);
    });

    // ── Particles ────────────────────────────────────────────────── //
    drawParticles();

    // ── Stats panel ──────────────────────────────────────────────── //
    ctx.textAlign = 'left';
    ctx.font = '16px JetBrains Mono';
    ctx.fillStyle = 'rgba(230,240,255,0.85)';
    ctx.fillText(`Fed Interval: ${fedState.interval} TTI`, 24, H - 46);
    ctx.fillText(`TTI: ${fedState.tti}`, 24, H - 24);
  }

  // ── Animate ─────────────────────────────────────────────────────── //
  function animate() {
    animId = requestAnimationFrame(animate);
    updateParticles();
    draw();
    render3D();
  }

  // ── 3D Global Model ──────────────────────────────────────────────── //
  function init3D() {
    const container = document.getElementById('fed-3d-container');
    if (!container) return;

    threeScene = new THREE.Scene();
    threeCamera = new THREE.PerspectiveCamera(45, container.offsetWidth / container.offsetHeight, 0.1, 1000);
    threeCamera.position.z = 240;

    threeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    threeRenderer.setSize(container.offsetWidth, container.offsetHeight);
    threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(threeRenderer.domElement);

    // Global Core
    const coreGeo = new THREE.IcosahedronGeometry(45, 1);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0xffb300, emissive: 0xffb300, emissiveIntensity: 0.8,
      wireframe: true, transparent: true, opacity: 0.4
    });
    threeCore = new THREE.Mesh(coreGeo, coreMat);
    threeScene.add(threeCore);

    // Inner Core Glow
    const innerGeo = new THREE.SphereGeometry(30, 32, 32);
    const innerMat = new THREE.MeshStandardMaterial({
      color: 0xffb300, emissive: 0xffb300, emissiveIntensity: 1.2,
      transparent: true, opacity: 0.2
    });
    const innerCore = new THREE.Mesh(innerGeo, innerMat);
    threeCore.add(innerCore);

    // Lights
    const p1 = new THREE.PointLight(0xffb300, 2, 300); p1.position.set(50, 50, 100); threeScene.add(p1);
    const p2 = new THREE.PointLight(0x00d4ff, 1.5, 300); p2.position.set(-50, -50, 100); threeScene.add(p2);
    threeScene.add(new THREE.AmbientLight(0xffffff, 0.2));

    // Satellites (Agents)
    for (let i = 0; i < 3; i++) {
        const sat = createSatellite(i);
        threeSats.push(sat);
        threeScene.add(sat.group);
    }
  }

  function createSatellite(id) {
    const group = new THREE.Group();
    const geo = new THREE.SphereGeometry(8, 16, 16);
    const mat = new THREE.MeshStandardMaterial({
        color: 0x00d4ff, emissive: 0x00d4ff, emissiveIntensity: 0.5,
        transparent: true, opacity: 0.8
    });
    const mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);

    // Label Sprite
    const labelTex = createTextTexture(`BS-${id}`, '#ffffff');
    const labelMat = new THREE.SpriteMaterial({ map: labelTex, transparent: true, opacity: 1.0, depthTest: false });
    const label    = new THREE.Sprite(labelMat);
    label.scale.set(45, 45, 1); // Smaller scale to prevent clutter
    label.position.y = 20;      // Lifted above the sphere
    group.add(label);
    
    // Orbit line
    const orbitGeo = new THREE.RingGeometry(115, 116, 64);
    const orbitMat = new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.05, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(orbitGeo, orbitMat);
    ring.rotation.x = Math.PI/2;
    // threeScene.add(ring); // Optional background ring

    return { group, mesh, mat, id, angle: (id * 2 * Math.PI) / 3 };
  }

  function render3D() {
    if (!threeRenderer) return;
    const t = threeClock.getElapsedTime();
    
    // Rotate core
    threeCore.rotation.y = t * 0.4;
    threeCore.rotation.z = t * 0.2;
    threeCore.material.emissiveIntensity = 0.8 + 0.4 * Math.sin(t * 3);

    // Orbit satellites
    const weights = fedState.weights.length ? fedState.weights : [0.33, 0.33, 0.33];
    threeSats.forEach((sat, i) => {
        const w = weights[i] || 0.1;
        sat.angle += 0.01 + w * 0.05; // speed proportional to weight
        sat.group.position.x = Math.cos(sat.angle) * 110;
        sat.group.position.z = Math.sin(sat.angle) * 110;
        sat.group.position.y = Math.sin(sat.angle * 0.5) * 40;
        
        const pulse = 0.5 + Math.sin(t * (3 + w * 10)) * 0.5;
        sat.mat.emissiveIntensity = 0.5 + w * 2.5 + pulse * 0.5;
        sat.mesh.scale.setScalar(1 + w * 1.5);
    });

    threeRenderer.render(threeScene, threeCamera);
  }

  // ── Init ─────────────────────────────────────────────────────────── //
  function init() {
    canvas = document.getElementById('fed-graph-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resizeCanvas();
    new ResizeObserver(resizeCanvas).observe(canvas.parentElement || canvas);
    init3D();
    animate();
  }

  function resizeCanvas() {
    if (!canvas) return;
    canvas.width  = canvas.offsetWidth  * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    if (threeRenderer) {
      const container = document.getElementById('fed-3d-container');
      threeRenderer.setSize(container.offsetWidth, container.offsetHeight);
      threeCamera.aspect = container.offsetWidth / container.offsetHeight;
      threeCamera.updateProjectionMatrix();
    }
  }

  // ── Update from WebSocket ─────────────────────────────────────────── //
  function update(frame) {
    const fed = frame.federated || {};
    const ev  = frame.fed_event;
    const agg = fed.aggregator || {};
    const last = agg.last_weights || [];
    const hist  = agg.history || [];
    const lastH = hist[hist.length - 1] || {};

    fedState.weights      = last;
    fedState.performances = lastH.performances || fedState.performances;
    fedState.round        = fed.round_count || 0;
    fedState.interval     = fed.fed_interval || 50;
    fedState.tti          = fed.tti_counter  || 0;

    if (ev) {
      // New federated round — spawn particles
      const n = Math.max(fedState.weights.length, 3);
      const cw = canvas ? canvas.offsetWidth : 400;
      const ch = canvas ? canvas.offsetHeight : 280;
      const { coord, agents } = getLayout(n, cw, ch);
      agents.forEach(ag => {
        spawnParticles(ag, coord, '#00d4ff', 6);
        spawnParticles(coord, ag, '#ffb300', 4);
      });
      fedState.weights      = ev.weights      || fedState.weights;
      fedState.performances = ev.performances || fedState.performances;
    }

    // Update round stats UI
    const el = document.getElementById('fed-stats');
    if (el) {
      el.innerHTML = `
        <div class="agent-stat" style="font-size:14px; margin-bottom: 5px;"><span class="agent-stat-k" style="color:#d1d5db;">Round</span><span class="agent-stat-v text-amber" style="font-size:16px;">${fedState.round}</span></div>
        <div class="agent-stat" style="font-size:14px; margin-bottom: 5px;"><span class="agent-stat-k" style="color:#d1d5db;">Interval</span><span class="agent-stat-v" style="font-size:16px;">${fedState.interval} TTI</span></div>
        <div class="agent-stat" style="font-size:14px; margin-bottom: 20px;"><span class="agent-stat-k" style="color:#d1d5db;">TTI Counter</span><span class="agent-stat-v mono" style="font-size:16px;">${fedState.tti}</span></div>
        
        <div style="color:var(--text-primary); font-size: 15px; font-weight: 600; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">Agent Attention Model Weights</div>
        
        ${(fedState.weights||[]).map((w,i)=>{
          const pct = Math.min(w * 100, 100).toFixed(1);
          return `
          <div style="margin-bottom: 18px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
              <span style="font-weight:600; color:#e8f4ff; font-size:14px;">Node BS-${i}</span>
              <span class="mono text-cyan" style="font-size:14px;">${w.toFixed(4)} (${pct}%)</span>
            </div>
            <div style="height: 12px; background: rgba(0,212,255,0.08); border-radius: 6px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5); border: 1px solid rgba(0,212,255,0.2);">
              <div style="height: 100%; width: ${pct}%; background: linear-gradient(90deg, #0088ff, #00d4ff); box-shadow: 0 0 10px #00d4ff; border-radius: 6px; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
          </div>`;
        }).join('')}
      `;
    }
  }

  return { init, update };
})();
