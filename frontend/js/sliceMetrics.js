/**
 * sliceMetrics.js — Per-slice KPI Dashboard
 * Real-time KPI cards, slice rows, throughput bars, latency gauges.
 */
window.SliceMet = (() => {
  const MAX_PTS = 200;
  const history = { utility:[], sla:[], energy:[], fairness:[] };

  function init() {}

  function getSliceServiceName(type, id) {
    const services = {
      'eMBB':  ['4K Video Streaming', 'High-Speed Web', 'Mobile Cloud Gaming'],
      'URLLC': ['Remote Surgery V2X', 'Autonomous Driving', 'Grid Smart Meters'],
      'mMTC':  ['Smart City Sensors', 'Industrial IoT', 'Wearable Telemetry'],
    };
    const list = services[type] || ['General Data'];
    return list[id % list.length];
  }

  function drawLine(canvas, data, color) {
    if (!canvas || data.length < 2) return;
    const W = canvas.width  = canvas.offsetWidth  * devicePixelRatio;
    const H = canvas.height = canvas.offsetHeight * devicePixelRatio;
    if (W === 0 || H === 0) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);
    const pad=8*devicePixelRatio, w=W-pad*2, h=H-pad*2;
    let mn=Math.min(...data), mx=Math.max(...data);
    if(mx===mn){mx+=0.1;mn-=0.1;}
    const sx=i=>pad+(i/(data.length-1))*w;
    const sy=v=>pad+h-((v-mn)/(mx-mn))*h;
    const grad=ctx.createLinearGradient(0,0,0,H);
    grad.addColorStop(0,color+'44'); grad.addColorStop(1,color+'00');
    ctx.beginPath(); ctx.moveTo(sx(0),sy(data[0]));
    data.forEach((v,i)=>ctx.lineTo(sx(i),sy(v)));
    ctx.lineTo(sx(data.length-1),H); ctx.lineTo(sx(0),H);
    ctx.closePath(); ctx.fillStyle=grad; ctx.fill();
    ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=1.5*devicePixelRatio;
    ctx.shadowColor=color; ctx.shadowBlur=4;
    ctx.moveTo(sx(0),sy(data[0]));
    data.forEach((v,i)=>ctx.lineTo(sx(i),sy(v)));
    ctx.stroke(); ctx.shadowBlur=0;
  }

  function push(arr, v) { arr.push(v); if(arr.length>MAX_PTS)arr.shift(); }

  function update(frame) {
    const kpi = frame.kpi || {};
    push(history.utility,  kpi.network_utility || 0);
    push(history.sla,      kpi.sla_violation   || 0);
    push(history.energy,   kpi.energy_kwh_h    || 0);
    push(history.fairness, kpi.fairness_index  || 1);

    // KPI cards
    setKPI('kpi-utility',  ((kpi.network_utility||0)*100).toFixed(2)+'%', 'var(--neon-cyan)');
    setKPI('kpi-sla',      ((kpi.sla_violation  ||0)*100).toFixed(3)+'%', 'var(--neon-red)');
    setKPI('kpi-energy',   (kpi.energy_kwh_h    ||0).toFixed(4),          'var(--neon-amber)');
    setKPI('kpi-fairness', (kpi.fairness_index  ||1).toFixed(4),          'var(--neon-green)');

    // History charts
    drawLine(document.getElementById('chart-utility'),  history.utility,  '#00d4ff');
    drawLine(document.getElementById('chart-sla'),      history.sla,      '#ff4466');
    drawLine(document.getElementById('chart-energy'),   history.energy,   '#ffb300');
    drawLine(document.getElementById('chart-fairness'), history.fairness, '#00ff88');

    // Slice rows
    renderSliceTable(frame.slices || [], kpi.throughputs || {}, kpi.latencies || {});

    // Throughput bars
    renderThroughputBars(kpi.throughputs || {}, frame.slices || []);
  }

  function setKPI(id, val, color) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.color = color;
    if (el.textContent !== val) {
      el.textContent = val;
      el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    }
  }

  function renderSliceTable(slices, throughputs, latencies) {
    const el = document.getElementById('slice-table');
    if (!el) return;
    el.innerHTML = slices.map(sl => {
      const sid   = String(sl.slice_id);
      const prbs  = sl.prbs || 0;
      const th    = throughputs[sid] || 0;
      const lat   = latencies[sid] || 0;
      const qmb   = (sl.queue_bytes/1e6).toFixed(2);
      const pct   = Math.min(prbs, 100);
      const clx   = sl.slice_type;
      const fillCls = 'slicefill-'+clx.toLowerCase();
      const service = getSliceServiceName(clx, sl.slice_id);
      const demandMbps = (sl.arrival_rate * 0.0008).toFixed(2);
      return `<div class="slice-row">
        <span class="slice-badge ${clx}">${clx}</span>
        <span style="font-size:12px; font-weight:600; color:var(--text-secondary); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${service}</span>
        <div class="slice-prb-bar"><div class="slice-prb-fill ${fillCls}" style="width:${pct}%"></div></div>
        <span class="mono text-cyan" style="font-size:11px" title="Allocated PRBs">${prbs} PRB</span>
        <span class="mono" style="font-size:10px;color:#a8b2d1" title="Real-time Network Demand">Req: ${demandMbps} Mb/s</span>
        <span class="mono" style="font-size:10px;color:var(--neon-amber)" title="Actual Throughput">Got: ${th.toFixed(2)} Mb/s</span>
        <span class="mono" style="font-size:10px;color:${lat>sl.latency_req?'var(--neon-red)':'var(--neon-green)'}">${lat.toFixed(1)}ms</span>
      </div>`;
    }).join('');
  }

  function renderThroughputBars(throughputs, slices) {
    const el = document.getElementById('th-bars');
    if (!el) return;
    const maxTH = Math.max(...Object.values(throughputs).map(Number), 1);
    el.innerHTML = slices.map(sl => {
      const th  = throughputs[String(sl.slice_id)] || 0;
      const pct = Math.min((th / maxTH) * 100, 100).toFixed(1);
      const col = {eMBB:'#0088ff', URLLC:'#ff4466', mMTC:'#00ff88'}[sl.slice_type]||'#888';
      const label = { eMBB:'HIGH SPEED', URLLC:'LOW LATENCY', mMTC:'IoT DEVICE' }[sl.slice_type] || sl.slice_type;
      const service = getSliceServiceName(sl.slice_type, sl.slice_id);
      
      const demandMbps = (sl.arrival_rate * 0.0008).toFixed(2);
      return `<div class="th-bar-row">
        <div class="th-bar-header">
          <span class="th-bar-label">${service} (${label})</span>
          <div style="display:flex; gap:12px; align-items:center; flex-shrink:0;">
             <span style="font-size:10px; font-weight:600; color:#a8b2d1; letter-spacing:0.5px;">Demand: ${demandMbps} Mb/s</span>
             <span style="font-size:10px; font-weight:800; color:var(--text-muted); letter-spacing:0.5px;">${sl.prbs} PRB</span>
             <span class="th-bar-val" style="min-width:65px; text-align:right; color:var(--neon-amber);">Thp: ${th.toFixed(2)} Mb/s</span>
          </div>
        </div>
        <div class="th-bar" style="height:5px;"><div class="th-fill" style="width:${pct}%;background:linear-gradient(90deg,${col}88,${col})"></div></div>
      </div>`;
    }).join('');
  }

  return { init, update };
})();
