/**
 * oranConsole.js — O-RAN xApp Console + Topology
 */
window.OranConsole = (() => {
  let topoCanvas, topoCtx;
  let topoAnim;
  let topoState = { ric: null, xapps: [], cells: [] };
  let topoT = 0;

  function init() {
    topoCanvas = document.getElementById('xapp-topology-canvas');
    if (topoCanvas) topoCtx = topoCanvas.getContext('2d');
    resizeTopo();
    new ResizeObserver(resizeTopo).observe(topoCanvas?.parentElement || document.body);
    animateTopo();
  }

  function resizeTopo() {
    if (!topoCanvas) return;
    topoCanvas.width  = topoCanvas.offsetWidth  * devicePixelRatio;
    topoCanvas.height = topoCanvas.offsetHeight * devicePixelRatio;
    topoCtx && topoCtx.scale(devicePixelRatio, devicePixelRatio);
  }

  function animateTopo() {
    topoAnim = requestAnimationFrame(animateTopo);
    topoT += 0.02;
    drawTopology();
  }

  function drawTopology() {
    if (!topoCanvas || !topoCtx) return;
    const ctx = topoCtx;
    const W   = topoCanvas.width  / devicePixelRatio;
    const H   = topoCanvas.height / devicePixelRatio;
    ctx.clearRect(0, 0, W, H);

    const xapps = topoState.xapps.length ? topoState.xapps : [
      {id:'SliceManager-xApp',status:'ACTIVE'},
      {id:'Interference-xApp',status:'ACTIVE'},
      {id:'Mobility-xApp',    status:'ACTIVE'},
      {id:'PowerCtrl-xApp',   status:'ACTIVE'},
    ];
    const cells = topoState.cells.length ? topoState.cells :
      [{id:'BS0'},{id:'BS1'},{id:'BS2'}];

    const cx=W/2, cy=H*0.50;
    const r1=Math.min(W,H)*0.28;  // xApps radius
    const r2=Math.min(W,H)*0.43;  // Cells radius

    // Draw RIC central node
    const rg = ctx.createRadialGradient(cx,cy,0,cx,cy,36);
    rg.addColorStop(0,'rgba(255,179,0,0.5)'); rg.addColorStop(1,'rgba(255,179,0,0.02)');
    ctx.beginPath(); ctx.arc(cx,cy,36,0,Math.PI*2);
    ctx.fillStyle=rg; ctx.fill();
    ctx.strokeStyle='#ffb300'; ctx.lineWidth=2;
    ctx.shadowColor='#ffb300'; ctx.shadowBlur=20; ctx.stroke(); ctx.shadowBlur=0;
    ctx.fillStyle='#ffb300'; ctx.font='bold 14px Inter'; ctx.textAlign='center';
    ctx.fillText('near-RT',cx,cy-6); ctx.fillText('RIC',cx,cy+9);

    // xApp nodes
    xapps.forEach((xapp, i) => {
      const angle = -Math.PI/2 + (2*Math.PI*i)/xapps.length;
      const nx = cx + r1*Math.cos(angle), ny = cy + r1*Math.sin(angle);
      const pulse = 1 + 0.06*Math.sin(topoT*2 + i);

      // Edge RIC→xApp
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nx,ny);
      ctx.strokeStyle='rgba(139,92,246,0.4)'; ctx.lineWidth=1.5;
      ctx.stroke();

      // Particle on edge
      const pt = ((topoT*0.4 + i*0.25) % 1);
      const px=cx+(nx-cx)*pt, py=cy+(ny-cy)*pt;
      ctx.beginPath(); ctx.arc(px,py,2.5,0,Math.PI*2);
      ctx.fillStyle='rgba(139,92,246,0.9)'; ctx.fill();

      // xApp node
      const nr=16*pulse;
      ctx.beginPath(); ctx.arc(nx,ny,nr,0,Math.PI*2);
      ctx.fillStyle='rgba(139,92,246,0.2)'; ctx.fill();
      ctx.strokeStyle='#8b5cf6'; ctx.lineWidth=1.5;
      ctx.shadowColor='#8b5cf6'; ctx.shadowBlur=10; ctx.stroke(); ctx.shadowBlur=0;
      const shortId = xapp.id.replace('-xApp','').slice(0,8);
      ctx.fillStyle='#c4b5fd'; ctx.font='bold 11px Inter'; ctx.textAlign='center';
      ctx.fillText(shortId, nx, ny+4);
    });

    // Cell (BS) nodes
    cells.forEach((cell, i) => {
      const angle = -Math.PI/2 + (2*Math.PI*i)/cells.length;
      const nx=cx+r2*Math.cos(angle), ny=cy+r2*Math.sin(angle);

      // Edge RIC→BS (E2 interface)
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nx,ny);
      ctx.strokeStyle='rgba(0,212,255,0.2)'; ctx.lineWidth=1;
      ctx.setLineDash([4,6]); ctx.stroke(); ctx.setLineDash([]);

      // E2 packet
      const pt2 = ((topoT*0.25 + i*0.33) % 1);
      const px2=cx+(nx-cx)*pt2, py2=cy+(ny-cy)*pt2;
      ctx.beginPath(); ctx.arc(px2,py2,2,0,Math.PI*2);
      ctx.fillStyle='rgba(0,212,255,0.7)'; ctx.fill();

      // Cell tower icon
      ctx.beginPath(); ctx.arc(nx,ny,18,0,Math.PI*2);
      ctx.fillStyle='rgba(0,136,255,0.15)'; ctx.fill();
      ctx.strokeStyle='#0088ff'; ctx.lineWidth=1.5;
      ctx.shadowColor='#0088ff'; ctx.shadowBlur=8; ctx.stroke(); ctx.shadowBlur=0;
      ctx.fillStyle='#88ccff'; ctx.font='bold 12px Inter'; ctx.textAlign='center';
      ctx.fillText(cell.id, nx, ny+4);
    });

    // Legend
    ctx.textAlign='left'; ctx.font='12px Inter';
    [['#ffb300','near-RT RIC'],['#8b5cf6','xApp'],['#0088ff','gNB/BS'],['rgba(0,212,255,0.6)','E2 Interface']].forEach(([c,t],i)=>{
      ctx.fillStyle=c; ctx.fillRect(10,H-80+i*18,12,12);
      ctx.fillStyle='rgba(200,220,255,0.8)'; ctx.fillText(t,28,H-70+i*18);
    });
  }

  function update(frame) {
    const xapp = frame.xapp || {};
    if (xapp.xapp_status) {
      topoState.xapps = Object.entries(xapp.xapp_status).map(([id,st])=>({id,status:st}));
    }
    const bs = frame.bs_positions || [];
    topoState.cells = bs.map(b=>({id:`BS${b.bs_id}`}));

    // Log stream
    const log = xapp.log || [];
    const el  = document.getElementById('xapp-log');
    if (el && log.length) {
      el.innerHTML = log.map(e => `
        <div class="log-entry">
          <span class="log-ts">[${e.ts}]</span>
          <span class="log-lvl-${e.level}"> ${e.level} </span>
          <span class="log-source"> ${e.source}:</span>
          <span class="log-msg"> ${e.msg}</span>
        </div>`).join('');
    }

    // E2 stats
    const statsEl = document.getElementById('e2-stats');
    if (statsEl) {
      statsEl.innerHTML = `
        <div class="kpi-card" style="padding: 24px 10px; margin-bottom: 8px;">
            <div class="kpi-label">E2 Latency Mean</div>
            <div class="kpi-value text-cyan" style="font-size: 32px;">${(xapp.e2_latency_mean||0).toFixed(2)}<span style="font-size:16px;">ms</span></div>
        </div>
        <div class="kpi-card" style="padding: 24px 10px; margin-bottom: 8px;">
            <div class="kpi-label">E2 Latency p99</div>
            <div class="kpi-value text-amber" style="font-size: 32px;">${(xapp.e2_latency_p99||0).toFixed(2)}<span style="font-size:16px;">ms</span></div>
        </div>
        <div class="kpi-card" style="padding: 24px 10px;">
            <div class="kpi-label">Agent Conflicts</div>
            <div class="kpi-value text-red" style="font-size: 32px;">${xapp.conflict_count||0}</div>
        </div>
        <div class="kpi-card" style="padding: 24px 10px;">
            <div class="kpi-label">Network TTI</div>
            <div class="kpi-value text-green" style="font-size: 32px;">${xapp.tti||0}</div>
        </div>`;
    }
  }

  return { init, update };
})();
