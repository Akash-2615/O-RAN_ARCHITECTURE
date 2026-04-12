/**
 * background.js — Dynamic Interactive O-RAN Data Matrix
 * Redesigned to be distinct from the 3D Network Towers:
 *   - Vertical "Data Streams" (Network Packet Flow)
 *   - Mouse-reactive "Interference Ripples"
 *   - Horizon-style "Grid Waves" (Throughput Pulses)
 *   - High-transparency for blending with UI panels
 */

window.SimBackground = (() => {
  let canvas, ctx, W, H, animId;
  let mouse = { x: -9999, y: -9999 };
  let streams = [];
  let bursts = [];
  let horizonWaves = [];
  let t = 0;

  const COLORS = ['#00d4ff', '#0088ff', '#8b5cf6', '#00ff88', '#ffb300'];

  function init() {
    canvas = document.getElementById('bg-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resize();

    buildStreams();

    window.addEventListener('resize', () => { resize(); buildStreams(); });
    window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; }, true);
    window.addEventListener('mousedown', e => spawnBurst(e.clientX, e.clientY), true);
    window.addEventListener('touchstart', e => {
        if (e.touches[0]) spawnBurst(e.touches[0].clientX, e.touches[0].clientY);
    }, { passive: true, capture: true });

    animate();
  }

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function buildStreams() {
    streams = [];
    const count = Math.floor(W / 45);
    for (let i = 0; i < count; i++) {
        streams.push({
            x: i * (W / count),
            y: Math.random() * H,
            speed: 1 + Math.random() * 3,
            len: 10 + Math.random() * 30,
            col: COLORS[Math.floor(Math.random() * COLORS.length)],
            op: 0.05 + Math.random() * 0.1,
            pulse: Math.random() * Math.PI
        });
    }
  }

  function spawnBurst(x, y) {
    bursts.push({ x, y, r: 0, maxR: 450, op: 1.0, col: COLORS[Math.floor(Math.random() * COLORS.length)] });
  }

  function drawStreams() {
    ctx.lineWidth = 1.5;
    streams.forEach(s => {
        s.y += s.speed;
        if (s.y > H) s.y = -s.len * 5;

        // Mouse repulsion - pull streams slightly
        const dx = s.x - mouse.x;
        if (Math.abs(dx) < 150) {
            const intensity = (150 - Math.abs(dx)) / 150;
            s.y += intensity * 5; // Gravity pull
        }

        const grad = ctx.createLinearGradient(0, s.y, 0, s.y + s.len * 5);
        grad.addColorStop(0, s.col + '00');
        grad.addColorStop(0.5, s.col + Math.floor(s.op * 255).toString(16).padStart(2,'0'));
        grad.addColorStop(1, s.col + '00');

        ctx.strokeStyle = grad;
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(s.x, s.y + s.len * 5);
        ctx.stroke();

        // Little bright "packet" head
        if (Math.random() < 0.02) {
            ctx.fillStyle = s.col;
            ctx.shadowColor = s.col; ctx.shadowBlur = 8;
            ctx.fillRect(s.x - 1, s.y + s.len * 4, 3, 3);
            ctx.shadowBlur = 0;
        }
    });
  }

  function drawHorizonWaves() {
    t += 0.02;
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const yBase = H * 0.85;
        const drift = Math.sin(t * 0.5 + i) * 20;
        const y = yBase + i * 40 + drift;
        const op = 0.4 - (i * 0.05);
        
        ctx.strokeStyle = `rgba(0, 212, 255, ${op})`;
        ctx.beginPath();
        for (let x = 0; x <= W; x += 50) {
            const wave = Math.sin(x * 0.005 + t + i) * 15;
            x === 0 ? ctx.moveTo(x, y + wave) : ctx.lineTo(x, y + wave);
        }
        ctx.stroke();
    }
  }

  function drawBursts() {
    bursts = bursts.filter(b => b.op > 0.01);
    bursts.forEach(b => {
        b.r += 4;
        b.op -= 0.01;
        ctx.beginPath();
        ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        ctx.strokeStyle = b.col + Math.floor(b.op * 255).toString(16).padStart(2,'0');
        ctx.lineWidth = 2;
        ctx.stroke();
    });
  }

  function animate() {
    animId = requestAnimationFrame(animate);
    ctx.fillStyle = '#050a14';
    ctx.fillRect(0, 0, W, H);

    drawHorizonWaves();
    drawStreams();
    drawBursts();
  }

  return { init };
})();
