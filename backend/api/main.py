"""
main.py
FastAPI application entry point.
Starts simulation loop on startup, serves REST + WebSocket.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from api.websocket import ConnectionManager
from api.routes import router, set_simulator
from simulation.simulator import Simulator


# ── Global instances ──────────────────────────────────────────────────── #
manager   = ConnectionManager()
simulator = Simulator()


# ── Lifespan ──────────────────────────────────────────────────────────── #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start simulation on startup, clean up on shutdown."""
    set_simulator(simulator)

    async def broadcast(frame: dict):
        await manager.broadcast(frame)

    simulator.set_broadcast(broadcast)
    simulator.start()
    print(f"[Main] Simulator started — {simulator.get_status()}")
    yield
    simulator.stop()
    print("[Main] Simulator stopped")


# ── App ───────────────────────────────────────────────────────────────── #
app = FastAPI(
    title="5G DRL O-RAN Network Slicing Simulator",
    description="Real-time Deep RL simulator for 5G network slicing with federated learning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# ── Serve Frontend Static Files ───────────────────────────────────────── #
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'frontend')
_frontend_dir = os.path.normpath(_frontend_dir)
if os.path.isdir(_frontend_dir):
    app.mount("/frontend", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

@app.get("/ui", include_in_schema=False)
async def ui_redirect():
    return RedirectResponse(url="/frontend/index.html")


# ── WebSocket endpoint ────────────────────────────────────────────────── #
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send initial snapshot immediately
        last = simulator.get_last_frame()
        if last:
            await ws.send_text(json.dumps(last, default=_serial))

        # Keep alive — receive pings from client
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "heartbeat",
                                                "tti": simulator._tti}))
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        print(f"[WS] Error: {e}")
        manager.disconnect(ws)


def _serial(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(type(obj))


# ── Health check ─────────────────────────────────────────────────────── #
@app.get("/")
async def root():
    return {
        "status":  "running",
        "tti":     simulator._tti,
        "clients": manager.n_clients,
        "docs":    "/docs",
    }
