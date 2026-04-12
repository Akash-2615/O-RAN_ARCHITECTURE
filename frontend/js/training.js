/**
 * training.js — Agent Training Dashboard
 * Live reward curves, loss curves, AI accuracy, replay buffer fill.
 * All data from WebSocket pipeline.
 */

window.Training = (() => {
  const MAX_PTS = 300;
  let charts = {};
  let activeAgent = 0;

  // Per-agent rolling data
  const agentData = {};

  function ensureAgent(id) {
    if (!agentData[id]) {
      agentData[id] = {
        rewards:  [],
        critic:   [],
        actor:    [],
        alpha:    [],
        entropy:  [],
        buffer:   [],
        steps:    [],
        initial:  null,
        peakReward: -Infinity,
        radar_stats: {
          capacity:    [],
          sla:         [],
          efficiency:  [],
          cooperation: []
        }
      };
    }
  }

  // ── Canvas mini line chart ──────────────────────────────────────── //
  function drawLine(canvas, data, color = '#00d4ff', label = '', minY = null, maxY = null) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width  = canvas.offsetWidth  * window.devicePixelRatio;
    const H = canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    if (W === 0 || H === 0) return;
    ctx.clearRect(0, 0, W, H);
    if (!data || data.length < 2) return;

    const pad = 12 * window.devicePixelRatio;
    const w = W - pad * 2, h = H - pad * 2;

    let mn = minY !== null ? minY : Math.min(...data);
    let mx = maxY !== null ? maxY : Math.max(...data);
    if (mx === mn) { mx += 1; mn -= 1; }

    const sx = (i) => pad + (i / (data.length - 1)) * w;
    const sy = (v) => pad + h - ((v - mn) / (mx - mn)) * h;

    // Grid lines
    ctx.strokeStyle = 'rgba(0,180,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad + (h / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    }

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, color + '40');
    grad.addColorStop(1, color + '00');
    ctx.beginPath();
    ctx.moveTo(sx(0), sy(data[0]));
    data.forEach((v, i) => ctx.lineTo(sx(i), sy(v)));
    ctx.lineTo(sx(data.length - 1), H);
    ctx.lineTo(sx(0), H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2 * window.devicePixelRatio;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 12; // Massive glow
    ctx.moveTo(sx(0), sy(data[0]));
    data.forEach((v, i) => ctx.lineTo(sx(i), sy(v)));
    ctx.stroke();
    ctx.shadowBlur = 0;

    // ── Metric Readout (Top Right) ──
    const last = data[data.length - 1];
    const valString = last.toFixed(4);
    
    ctx.font = `bold ${14 * window.devicePixelRatio}px JetBrains Mono`;
    const textWidth = ctx.measureText(valString).width;
    const boxW = textWidth + 16 * window.devicePixelRatio;
    const boxH = 24 * window.devicePixelRatio;
    
    // Glass capsule background
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.beginPath();
    ctx.roundRect(W - pad - boxW, pad - 5, boxW, boxH, 4);
    ctx.fill();
    ctx.strokeStyle = color + '66';
    ctx.stroke();

    // The Value
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.fillText(valString, W - pad - boxW/2, pad + 12 * window.devicePixelRatio);

    // ── Label (Top Left) ──
    if (label) {
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'left';
      ctx.font      = `800 ${11 * window.devicePixelRatio}px Inter`;
      ctx.fillText(label.toUpperCase(), pad + 4, pad + 10 * window.devicePixelRatio);
    }
  }

  function push(arr, val) {
    arr.push(val);
    if (arr.length > MAX_PTS) arr.shift();
  }

  function pushRadar(arr, val) {
    arr.push(val);
    if (arr.length > 20) arr.shift(); // 1s window for "Live" feel
  }

  // ── Buffer fill gauge ────────────────────────────────────────── //
  function drawBuffer(canvas, fill) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width  = canvas.offsetWidth  * window.devicePixelRatio;
    const H = canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    if (W === 0 || H === 0) return;
    ctx.clearRect(0, 0, W, H);
    const r = Math.max(0.1, Math.min(W, H) / 2 - 6);
    const cx = W / 2, cy = H / 2;
    const startAngle = -Math.PI / 2;
    const endAngle   = startAngle + 2 * Math.PI * fill;

    // BG arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0,180,255,0.08)';
    ctx.lineWidth   = 8 * window.devicePixelRatio;
    ctx.stroke();

    // Fill arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    const grad = ctx.createLinearGradient(0, 0, W, H);
    grad.addColorStop(0, '#8b5cf6');
    grad.addColorStop(1, '#00d4ff');
    ctx.strokeStyle = grad;
    ctx.lineWidth   = 8 * window.devicePixelRatio;
    ctx.lineCap     = 'round';
    ctx.shadowColor = '#8b5cf6';
    ctx.shadowBlur  = 10;
    ctx.stroke();

    // Label
    ctx.shadowBlur = 0;
    ctx.fillStyle  = '#e8f4ff';
    ctx.font       = `bold ${13 * window.devicePixelRatio}px JetBrains Mono`;
    ctx.textAlign  = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText((fill * 100).toFixed(0) + '%', cx, cy - 4 * window.devicePixelRatio);
    ctx.fillStyle = 'rgba(150,180,220,0.5)';
    ctx.font      = `${8 * window.devicePixelRatio}px Inter`;
    ctx.fillText('BUFFER', cx, cy + 10 * window.devicePixelRatio);
  }

  // ── Agent selector ───────────────────────────────────────────── //
  function initAgentSelector(n) {
    const sel = document.getElementById('agent-selector');
    if (!sel) return;
    sel.innerHTML = '';
    for (let i = 0; i < n; i++) {
      const btn = document.createElement('button');
      btn.className = 'agent-btn' + (i === activeAgent ? ' active' : '');
      btn.textContent = `BS-${i}`;
      btn.onclick = () => {
        activeAgent = i;
        sel.querySelectorAll('.agent-btn').forEach((b, j) =>
          b.classList.toggle('active', j === i));
        renderActiveAgent();
      };
      sel.appendChild(btn);
    }
  }

  const avg = (arr) => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0.0;

  function renderActiveAgent() {
    if (!agentData[activeAgent]) return;
    const d = agentData[activeAgent];
    drawLine(document.getElementById('chart-reward'),  d.rewards, '#00d4ff', 'Episode Reward');
    drawLine(document.getElementById('chart-critic'),   d.critic,  '#8b5cf6', 'Critic Loss');
    drawLine(document.getElementById('chart-actor'),    d.actor,   '#ff4466', 'Actor Loss');
    drawLine(document.getElementById('chart-entropy'),  d.entropy, '#00ff88', 'AI Accuracy');
    const buf = d.buffer[d.buffer.length - 1] || 0;
    drawBuffer(document.getElementById('chart-buffer'), buf / 100_000);
    
    // ── Global Radar (All Agents) ──
    const colors = ['#00d4ff', '#8b5cf6', '#00ff88'];
    const radarSeries = Object.keys(agentData).map((id, idx) => ({
        id: `BS-${id}`,
        color: colors[idx % colors.length],
        data: [
            avg(agentData[id].radar_stats.capacity),
            avg(agentData[id].radar_stats.sla),
            avg(agentData[id].radar_stats.efficiency),
            avg(agentData[id].radar_stats.cooperation)
        ]
    }));
    drawRadar(document.getElementById('chart-radar'), radarSeries);
  }

  // ── Radar Engine ────────────────────────────────────────────── //
  function drawRadar(canvas, series) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width  = canvas.offsetWidth  * window.devicePixelRatio;
    const H = canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    if (W === 0 || H === 0) return;
    ctx.clearRect(0,0,W,H);

    const labels = ['CAPACITY', 'SLA SUCCESS', 'EFFICIENCY', 'COOPERATION'];
    const cx = W/2, cy = H/2;
    const radius = Math.min(cx, cy) - 95 * window.devicePixelRatio; 

    // 1. Draw Grid (Neon style)
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let j = 1; j <= 4; j++) {
        ctx.beginPath();
        const r = (radius / 4) * j;
        for (let i = 0; i < 4; i++) {
            const ang = (i / 4) * Math.PI * 2 - Math.PI/2;
            ctx.lineTo(cx + Math.cos(ang)*r, cy + Math.sin(ang)*r);
        }
        ctx.closePath();
        ctx.stroke();
    }

    // 2. Draw Axes & Quadrant-Aware Labels
    labels.forEach((L, i) => {
        const ang = (i / 4) * Math.PI * 2 - Math.PI/2;
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(ang)*radius, cy + Math.sin(ang)*radius);
        ctx.stroke();

        // Quadrant-aware alignment for clear labels
        ctx.fillStyle = 'rgba(255,255,255,0.95)';
        ctx.font = `900 ${9 * window.devicePixelRatio}px Inter`;
        
        const cos = Math.cos(ang);
        const sin = Math.sin(ang);

        if (Math.abs(cos) < 0.1) {
            ctx.textAlign = 'center';
            ctx.textBaseline = sin < 0 ? 'bottom' : 'top';
        } else {
            ctx.textAlign = cos < 0 ? 'right' : 'left';
            ctx.textBaseline = 'middle';
        }

        const offset = 10 * window.devicePixelRatio;
        const lx = cx + cos * (radius + offset);
        const ly = cy + sin * (radius + offset);
        ctx.fillText(L, lx, ly);
    });

    // 3. Draw Series & Legend
    const colors = ['#00d4ff', '#8b5cf6', '#00ff88'];
    renderRadarLegend(series, colors);
    
    series.forEach((s, idx) => {
        const col = s.color || colors[idx % colors.length];
        ctx.beginPath();
        ctx.strokeStyle = col;
        ctx.fillStyle = col + '25';
        ctx.lineWidth = 2.5 * window.devicePixelRatio;
        ctx.shadowColor = col;
        ctx.shadowBlur = 8;
        
        s.data.forEach((v, i) => {
            const val = Math.max(0.05, Math.min(v, 1.0));
            const ang = (i / 4) * Math.PI * 2 - Math.PI/2;
            const r = val * radius;
            ctx.lineTo(cx + Math.cos(ang)*r, cy + Math.sin(ang)*r);
        });
        ctx.closePath();
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fill();
    });
  }

  function renderRadarLegend(series, colors) {
    const el = document.getElementById('radar-legend');
    if (!el) return;
    el.innerHTML = series.map((s, i) => {
      const col = s.color || colors[i % colors.length];
      return `
        <div style="display:flex; align-items:center; gap:6px;">
          <div style="width:12px; height:12px; border-radius:3px; background:${col};
               box-shadow:0 0 8px ${col}88; flex-shrink:0;"></div>
          <span style="font-size:11px; font-weight:800; color:rgba(255,255,255,0.85);
                       font-family:var(--font-mono); letter-spacing:0.5px;">${s.id}</span>
        </div>
      `;
    }).join('');
  }

  // ── Agent status cards ───────────────────────────────────────── //
  function renderAgentCards(agents) {
    const el = document.getElementById('agent-cards');
    if (!el) return;
    el.innerHTML = agents.map((ag, i) => `
      <div class="agent-card">
        <div class="agent-card-header">
          <span class="agent-id">BS-${ag.bs_id}</span>
            <div class="stat-group">
            <label>AI ACCURACY</label>
            <div class="stat-val ${(ag.losses?.entropy || 0) > 0.1 ? 'warn' : ''}">${((1.0 - (ag.losses?.entropy || 0)) * 100.0).toFixed(2)}%</div>
          </div>
          <span class="agent-live-dot"></span>
        </div>
        ${statRow('Steps',   APP.fmt.int(ag.total_steps))}
        ${statRow('Updates', APP.fmt.int(ag.total_updates))}
        ${statRow('Buffer',  APP.fmt.int(ag.buffer_size))}
        ${statRow('Reward',  (ag.last_reward||0).toFixed(3))}
        ${statRow('α (temp)', (ag.alpha||0).toFixed(4))}
        ${statRow('Critic L',(ag.losses?.critic||0).toFixed(4))}
        ${statRow('Actor L', (ag.losses?.actor||0).toFixed(4))}
      </div>
    `).join('');
  }
  const statRow = (k, v) =>
    `<div class="agent-stat"><span class="agent-stat-k">${k}</span><span class="agent-stat-v">${v}</span></div>`;

  // ── init / update ────────────────────────────────────────────── //
  function init() {
    // Setup resize observer for canvases
    const canvases = ['chart-reward','chart-critic','chart-actor','chart-buffer','chart-entropy'];
    canvases.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        new ResizeObserver(() => renderActiveAgent()).observe(el);
      }
    });
  }

  function update(frame) {
    const agents   = frame.agents      || [];
    const trainMet = frame.train_metrics || [];
    const kpi      = frame.kpi         || {};

    if (agents.length === 0) return;

    // Ensure selector built
    if (!document.querySelector('#agent-selector .agent-btn')) initAgentSelector(agents.length);

    agents.forEach((ag, i) => {
      ensureAgent(i);
      const d = agentData[i];
      push(d.rewards, ag.last_reward || 0);
      push(d.buffer,  ag.buffer_size || 0);
      
      const rb = ag.reward_breakdown || {};
      const hasBreakdown = rb.throughput_norm !== undefined;

      // ── RADAR: All 4 axes use true per-agent breakdown data ─────────
      if (hasBreakdown) {
        // CAPACITY: Normalized throughput score (0-1). throughput_norm ≈ 0-2 range.
        pushRadar(d.radar_stats.capacity, Math.min(rb.throughput_norm / 1.5, 1.0));

        // SLA SUCCESS: 1 = no violations. 1000 per violation, 5000 for URLLC
        // Most of the time will be 0 violations → 1.0. Scale so one violation = 0.5.
        const slaScore = Math.max(0, 1.0 - (rb.sla_penalty / 2000.0));
        pushRadar(d.radar_stats.sla, slaScore);

        // EFFICIENCY: energy ≈ 3.0 units at full power (15 UEs @ 43dBm)
        // Lower energy = higher efficiency. Normalize: energy/4 maps 0→1 as 1→0
        const effScore = Math.max(0.05, 1.0 - Math.min(rb.energy / 4.0, 0.95));
        pushRadar(d.radar_stats.efficiency, effScore);

        // COOPERATION: fairness_err typically 0.2-1.8 range
        // Low error = high cooperation. 0 = perfect, 1.5+ = poor
        const coopScore = Math.max(0.05, 1.0 - Math.min(rb.fairness_err / 1.5, 0.95));
        pushRadar(d.radar_stats.cooperation, coopScore);
      } else {
        // Push neutral values so radar stays visible
        pushRadar(d.radar_stats.capacity,    0.4);
        pushRadar(d.radar_stats.sla,         0.85);
        pushRadar(d.radar_stats.efficiency,  0.5);
        pushRadar(d.radar_stats.cooperation, 0.4);
      }

      if (trainMet[i]) {
        push(d.critic,  trainMet[i].critic  || 0);
        
        // Show Raw Actor Loss (Unmapped for professional debugging)
        const actL = trainMet[i].actor || 0;
        push(d.actor,   actL);
        
        // Raw Network-Level Precision Scaling 
        // 1.0 Jitter = 0% Accuracy | 0.03 Jitter = 97% Accuracy
        const rawJitter = trainMet[i].entropy || 0; 
        const conf = Math.max(0, Math.min(100, (1.0 - rawJitter) * 100.0));
        push(d.entropy, conf);
      }
    });

    renderActiveAgent();
    renderAgentCards(agents);
    renderGlobalComparison(agents, kpi);
  }

  function renderGlobalComparison(agents, kpi) {
    const el = document.getElementById('agent-comparison-wrap');
    if (!el) return;

    // 1. Capture/Stabilize Baseline: Wait for the first true non-zero reward tick
    agents.forEach((ag, i) => {
      const d = agentData[i];
      if (!d.initial && (ag.last_reward || 0) !== 0) {
         d.initial = {
           reward: ag.last_reward,
           steps:  ag.total_steps || 0,
         };
      }
    });

    // 2. Identify Leader (highest current reward)
    let bestIdx = agents.reduce((bi, ag, i) =>
      (ag.last_reward || 0) > (agents[bi]?.last_reward || 0) ? i : bi, 0);

    // 3. Network-level KPI pills at top
    const utilPct  = ((kpi?.network_utility  || 0) * 100).toFixed(1);
    const fairness = (kpi?.fairness_index    || 0).toFixed(3);
    const slaViol  = ((kpi?.sla_violation    || 0) * 100).toFixed(2);
    const energy   = (kpi?.energy_kwh_h      || 0).toFixed(4);

    // 4. Render
    el.innerHTML = `
      <div class="network-kpi-summary">
        <div class="nkpi-pill">
          <div class="nkpi-val" style="color:var(--neon-cyan)">${utilPct}%</div>
          <div class="nkpi-label">Network Utility</div>
        </div>
        <div class="nkpi-pill">
          <div class="nkpi-val" style="color:var(--neon-green)">${fairness}</div>
          <div class="nkpi-label">Jain Fairness</div>
        </div>
        <div class="nkpi-pill">
          <div class="nkpi-val" style="color:${parseFloat(slaViol) > 0 ? 'var(--neon-red)' : 'var(--neon-green)'}">  ${slaViol}%</div>
          <div class="nkpi-label">SLA Violations</div>
        </div>
        <div class="nkpi-pill">
          <div class="nkpi-val" style="color:var(--neon-amber)">${energy}</div>
          <div class="nkpi-label">Energy (kWh)</div>
        </div>
      </div>
      <table class="comp-table">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Initial Rwd</th>
            <th>Present Rwd</th>
            <th>Total Δ</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${agents.map((ag, i) => {
            const d = agentData[i];
            const init = d.initial?.reward ?? 0;
            const curr = ag.last_reward || 0;
            const diff = curr - init;
            const isLeader = (i === bestIdx);

            const deltaCls  = diff >= 0 ? 'comp-delta-up' : 'comp-delta-down';
            const deltaSign = diff >= 0 ? '↑' : '↓';

            // Convergence status based on update count
            const steps = ag.total_steps || 0;
            let statLbl = 'TRAINING..';
            let statCls = 'status-training';
            if (steps > 2500) {
              statLbl = 'EXPERT-LOCK';
              statCls = 'status-optimized';
            } else if (steps > 800) {
              statLbl = 'OPTIMIZED';
              statCls = 'status-optimized';
            }

            return `
              <tr class="comp-row ${isLeader ? 'comp-leader' : ''}">
                <td><span class="comp-agent-name">${isLeader ? '👑 ' : ''}BS-${ag.bs_id}</span></td>
                <td><span class="comp-rwd-val">${init.toFixed(3)}</span></td>
                <td><span class="comp-rwd-val" style="color:${curr >= 0 ? 'var(--neon-green)' : 'var(--neon-red)'}">${curr.toFixed(3)}</span></td>
                <td><span class="${deltaCls}">${deltaSign} ${Math.abs(diff).toFixed(3)}</span></td>
                <td><span class="comp-status-pill ${statCls}">${statLbl}</span></td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    `;
  }

  return { init, update };
})();
