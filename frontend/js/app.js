/**
 * app.js — Application bootstrap, WebSocket client, tab routing
 * Connects to the FastAPI backend and dispatches live data to all modules.
 */

// Connect to the backend dynamically, supporting Live Server on 5500 routing to 8000
const _host    = window.location.port === '5500' ? '127.0.0.1:8000' : '127.0.0.1:8000';
const WS_URL   = `ws://${_host}/ws`;
const API_BASE = `http://${_host}`;

// ── Global state ────────────────────────────────────────────────────── //
window.SIM = {
  tti:       0,
  kpi:       {},
  slices:    [],
  agents:    [],
  federated: {},
  xapp:      {},
  explainability: {},
  connected: false,
  ws:        null,
};

// ── DOM ready ────────────────────────────────────────────────────────── //
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  connectWebSocket();
  initHeaderKPIs();
  // Boot each module once DOM is ready
  window.Network3D  && window.Network3D.init();
  window.Training   && window.Training.init();
  window.Federated  && window.Federated.init();
  window.Explain    && window.Explain.init();
  window.SliceMet   && window.SliceMet.init();
  window.OranConsole && window.OranConsole.init();
  window.SimBackground && window.SimBackground.init();
  setInterval(heartbeatPing, 20000);
});

// ── Tab routing ───────────────────────────────────────────────────────  //
function initTabs() {
  const tabs  = document.querySelectorAll('.nav-tab');
  const pages = document.querySelectorAll('.tab-page');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove('active'));
      pages.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const pg = document.getElementById('page-' + target);
      if (pg) pg.classList.add('active');
      // Notify 3D module when its tab becomes active
      if (target === 'network3d' && window.Network3D) window.Network3D.onResize();
    });
  });
}

// ── WebSocket ─────────────────────────────────────────────────────────  //
function connectWebSocket() {
  setStatus('connecting');
  try {
    const ws = new WebSocket(WS_URL);
    window.SIM.ws = ws;

    ws.onopen = () => {
      setStatus('connected');
      window.SIM.connected = true;
      console.log('[WS] Connected');
    };

    ws.onmessage = (event) => {
      try {
        const frame = JSON.parse(event.data);
        if (frame.type === 'pong' || frame.type === 'heartbeat') return;
        handleFrame(frame);
      } catch(e) {
        console.warn('[WS] parse error', e);
      }
    };

    ws.onerror = (e) => {
      console.error('[WS] Error', e);
      setStatus('error');
    };

    ws.onclose = () => {
      setStatus('disconnected');
      window.SIM.connected = false;
      console.warn('[WS] Closed — reconnecting in 3s');
      setTimeout(connectWebSocket, 3000);
    };
  } catch(e) {
    console.error('[WS] connect failed', e);
    setTimeout(connectWebSocket, 3000);
  }
}

function heartbeatPing() {
  if (window.SIM.ws && window.SIM.ws.readyState === WebSocket.OPEN) {
    window.SIM.ws.send('ping');
  }
}

// ── Frame dispatch ─────────────────────────────────────────────────── //
function handleFrame(frame) {
  // Update global state
  window.SIM.tti       = frame.tti       || window.SIM.tti;
  window.SIM.kpi       = frame.kpi       || {};
  window.SIM.slices    = frame.slices    || [];
  window.SIM.agents    = frame.agents    || [];
  window.SIM.federated = frame.federated || {};
  window.SIM.xapp      = frame.xapp      || {};
  window.SIM.train     = frame.train_metrics || [];
  window.SIM.explainability = frame.explainability || {};
  window.SIM.importance_global = frame.importance_global || {};
  window.SIM.prb_maps  = frame.prb_maps  || {};
  window.SIM.ue_positions = frame.ue_positions || {};
  window.SIM.bs_positions = frame.bs_positions || [];
  window.SIM.fed_event = frame.fed_event;

  // Update header KPIs
  updateHeaderKPIs(frame.kpi || {});

  // Dispatch to active/all modules
  window.Network3D  && window.Network3D.update(frame);
  window.SliceMet   && window.SliceMet.update(frame);
  window.Training   && window.Training.update(frame);
  window.Federated  && window.Federated.update(frame);
  window.OranConsole && window.OranConsole.update(frame);

  // Explainability only when data is present
  if (frame.importance_global && Object.keys(frame.importance_global).length > 0) {
    window.Explain && window.Explain.update(frame);
  }
}

// ── Header KPIs ──────────────────────────────────────────────────────  //
function initHeaderKPIs() {
  // render placeholders
  updateHeaderKPIs({ network_utility: 0, sla_violation: 0, energy_kwh_h: 0, fairness_index: 1 });
}

function updateHeaderKPIs(kpi) {
  setVal('hkpi-utility',  ((kpi.network_utility || 0) * 100).toFixed(1) + '%');
  setVal('hkpi-sla',      ((kpi.sla_violation   || 0) * 100).toFixed(2) + '%');
  setVal('hkpi-energy',   (kpi.energy_kwh_h || 0).toFixed(3));
  setVal('hkpi-fairness', (kpi.fairness_index || 1).toFixed(3));
  setVal('tti-counter',   `TTI: ${window.SIM.tti.toLocaleString()}`);
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  if (el.textContent !== val) {
    el.textContent = val;
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
  }
}

// ── Connection status ─────────────────────────────────────────────────  //
function setStatus(state) {
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  const pill  = document.querySelector('.status-pill');
  if (!dot || !label) return;
  const map = {
    connected:    { text: 'LIVE',         color: 'var(--neon-green)',  bg: 'rgba(0,255,136,0.08)',  border: 'rgba(0,255,136,0.2)' },
    connecting:   { text: 'CONNECTING…',  color: 'var(--neon-amber)', bg: 'rgba(255,179,0,0.08)', border: 'rgba(255,179,0,0.2)'  },
    disconnected: { text: 'OFFLINE',      color: 'var(--neon-red)',    bg: 'rgba(255,68,102,0.08)', border: 'rgba(255,68,102,0.2)' },
    error:        { text: 'ERROR',        color: 'var(--neon-red)',    bg: 'rgba(255,68,102,0.08)', border: 'rgba(255,68,102,0.2)' },
  };
  const s = map[state] || map.disconnected;
  dot.style.background  = s.color;
  label.textContent     = s.text;
  label.style.color     = s.color;
  if (pill) {
    pill.style.background = s.bg;
    pill.style.borderColor= s.border;
  }
}

// ── Utility helpers ────────────────────────────────────────────────────  //
window.APP = {
  api: async (path) => {
    try {
      const r = await fetch(API_BASE + path);
      return r.ok ? r.json() : null;
    } catch(e) { return null; }
  },
  fmt: {
    pct:   v => ((v||0)*100).toFixed(1) + '%',
    mbps:  v => ((v||0)/1e6).toFixed(2) + ' Mb/s',
    num2:  v => (v||0).toFixed(2),
    num3:  v => (v||0).toFixed(3),
    int:   v => Math.round(v||0).toLocaleString(),
  }
};
