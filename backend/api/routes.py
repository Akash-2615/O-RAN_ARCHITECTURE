"""
routes.py
REST API route handlers.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
_sim = None   # injected from main.py


def set_simulator(sim):
    global _sim
    _sim = sim


# ── Models ────────────────────────────────────────────────────────────── #
class ConfigUpdate(BaseModel):
    tti_sleep_s:    Optional[float] = None
    fed_interval:   Optional[int]   = None
    alpha:          Optional[float] = None
    beta:           Optional[float] = None
    gamma:          Optional[float] = None
    delta:          Optional[float] = None


# ── Endpoints ─────────────────────────────────────────────────────────── #
@router.get("/api/status")
async def get_status():
    return _sim.get_status()


@router.get("/api/metrics/history")
async def get_metrics_history(n: int = 200):
    return {"history": _sim.get_metrics_history(n)}


@router.get("/api/metrics/current")
async def get_metrics_current():
    return _sim.metrics.get_current()


@router.get("/api/agents")
async def get_agents():
    return {"agents": _sim.mac.get_agent_statuses()}


@router.get("/api/federated")
async def get_federated():
    return _sim.fed.get_status()


@router.get("/api/explainability")
async def get_explainability():
    return {
        "global_mean":  _sim.imp_tracker.get_global_mean(),
        "per_agent":    {i: _sim.imp_tracker.get_latest(i)
                         for i in range(_sim.net.n_cells)},
        "heatmap":      _sim.explainers[0].get_heatmap(50),
    }


@router.get("/api/xapp")
async def get_xapp():
    return _sim.xapp.get_status()


@router.get("/api/xapp/topology")
async def get_xapp_topology():
    return _sim.xapp.get_topology()


@router.get("/api/xapp/log")
async def get_xapp_log(n: int = 50):
    return {"log": _sim.xapp.log.get(n)}


@router.get("/api/network/positions")
async def get_positions():
    return {
        "bs":  _sim.net.get_bs_positions(),
        "ues": _sim.net.get_ue_positions(),
    }


@router.get("/api/snapshot")
async def get_snapshot():
    return _sim.get_last_frame()


@router.post("/api/reset")
async def reset():
    _sim.reset()
    _sim.start()
    return {"status": "reset", "msg": "Simulator reset and restarted"}


@router.post("/api/sim/start")
async def start_sim():
    """Ensure the simulation loop is running."""
    if not _sim.running:
        _sim.start()
        return {"status": "started", "tti": _sim._tti}
    return {"status": "already_running", "tti": _sim._tti}


@router.get("/api/sim/status")
async def sim_status():
    return {"running": _sim.running, "tti": _sim._tti}


@router.post("/api/config")
async def update_config(cfg: ConfigUpdate):
    import backend.simulation.simulator as sim_mod
    if cfg.tti_sleep_s is not None:
        sim_mod.TTI_SLEEP_S = cfg.tti_sleep_s
    if cfg.fed_interval is not None:
        _sim.fed.fed_interval = cfg.fed_interval
    if cfg.alpha is not None:
        _sim.mac.env.cfg.alpha = cfg.alpha
    if cfg.beta is not None:
        _sim.mac.env.cfg.beta = cfg.beta
    if cfg.gamma is not None:
        _sim.mac.env.cfg.gamma = cfg.gamma
    if cfg.delta is not None:
        _sim.mac.env.cfg.delta = cfg.delta
    return {"status": "ok", "config": cfg.dict(exclude_none=True)}
