/**
 * explainability.js — SHAP Feature Attribution Dashboard
 */
window.Explain = (() => {
  let heatData = [];
  const MAX_HEAT_ROWS = 50;

  function init() {
    new ResizeObserver(drawAll).observe(document.body);
    new ResizeObserver(drawAll).observe(document.body);
  }

  function formatFeatureName(feat) {
    if (feat === 'bs_power_dbm') return 'BS Transmit Power';
    if (feat === 'bs_interf') return 'Radio Interference';
    
    // Parse neural network inputs like "sl2_delay_norm"
    const match = feat.match(/^sl(\d+)_(.*)$/);
    if (match) {
        const slId = match[1];
        const metric = match[2];
        const type = (['eMBB', 'URLLC', 'mMTC', 'eMBB', 'mMTC', 'URLLC'])[slId] || 'Slice';
        
        let mName = metric;
        if (metric === 'delay_norm') mName = 'SLA Latency (ms)';
        if (metric === 'cqi') mName = 'Channel Quality (CQI)';
        if (metric === 'queue_bdt') mName = 'Data Queue Size';
        if (metric === 'priority') mName = 'QoS Priority';
        if (metric === 'class') mName = 'Traffic Class';
        
        return `${type} [Slice ${slId}] ${mName}`;
    }
    return feat;
  }

  function drawSHAPBars(el, importance) {
    if (!el || !importance) return;
    const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 12);
    el.innerHTML = entries.map(([feat, val]) => {
      const pct = Math.min(val * 100 * 6, 100).toFixed(1);
      const readableFeat = formatFeatureName(feat);
      return `<div class="shap-bar-row">
        <span class="shap-feat-name" title="${readableFeat}">${readableFeat}</span>
        <div class="shap-bar-bg"><div class="shap-bar-fill" style="width:${pct}%"></div></div>
        <span class="shap-val">${(val * 100).toFixed(2)}%</span>
      </div>`;
    }).join('');
  }

  function drawHeatmap(canvas, rowsData) {
    if (!canvas || !rowsData.length) return;
    const nRows = rowsData.length, nCols = rowsData[0].length;
    // Rows represent Time, Cols represent Network Features
    const cellW = Math.max(12, (canvas.offsetWidth || 800) / nCols);
    const cellH = 14; 
    canvas.width  = nCols * cellW * devicePixelRatio;
    canvas.height = nRows * cellH * devicePixelRatio;
    canvas.style.height = (nRows * cellH) + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(devicePixelRatio, devicePixelRatio);
    rowsData.forEach((row, ri) => {
      row.forEach((val, ci) => {
        // Expand color scale from dark blue (low impact) to neon pink (high impact)
        const intensity = Math.min(val * 18, 1);
        const r = Math.round(intensity * 255);
        const g = Math.round(intensity * 50 + 20);
        const b = Math.round((1-intensity)*255 + intensity*120);
        
        ctx.fillStyle = `rgba(${r},${g},${b},${0.35 + intensity*0.65})`;
        ctx.fillRect(ci * cellW, ri * cellH, cellW - 2, cellH - 2);
      });
    });
  }

  function drawAll() {
    const gi = window.SIM?.importance_global || {};
    drawSHAPBars(document.getElementById('shap-bars-list'), gi);
  }

  function update(frame) {
    const gi  = frame.importance_global || {};
    const exp = frame.explainability    || {};
    const vals = Object.values(gi);
    if (vals.length) { heatData.push(vals); if (heatData.length > MAX_HEAT_ROWS) heatData.shift(); }
    drawSHAPBars(document.getElementById('shap-bars-list'),  gi);
    drawSHAPBars(document.getElementById('shap-agent-bars'), Object.values(exp)[0] || gi);
    drawHeatmap(document.getElementById('shap-heat-canvas'), heatData);
    const nAgents = frame.agents ? frame.agents.length : (Object.keys(exp).length || 3);
    const nFeatures = Object.keys(gi).length || 31;

    const el = document.getElementById('explain-stats');
    if (!el) return;
    const top3 = Object.entries(gi).sort((a,b)=>b[1]-a[1]).slice(0,3);

    el.innerHTML = `
      <div class="agent-stat">
        <span class="agent-stat-k">Input Network Features</span>
        <span class="agent-stat-v text-cyan">${nFeatures} Variables</span>
      </div>
      <div class="agent-stat">
        <span class="agent-stat-k">Independent RL Agents</span>
        <span class="agent-stat-v text-green">${nAgents} Active Nodes</span>
      </div>
      
      <div style="margin-top:24px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.1); color:var(--text-secondary); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:1px;">
        Key Decision Drivers (Global)
      </div>
      <div style="margin-bottom:12px; font-size:11px; color:var(--text-muted); line-height:1.4;">
        Features with the highest impact on PRB allocation logic across all base stations.
      </div>
      ${top3.map(([k,v])=>`
        <div class="agent-stat" style="margin: 8px 0;">
          <span class="agent-stat-k text-purple" style="font-size:11px;">${formatFeatureName(k)}</span>
          <span class="agent-stat-v" style="font-size:13px;">${(v*100).toFixed(2)}%</span>
        </div>`).join('')}
    `;
  }

  return { init, update };
})();
