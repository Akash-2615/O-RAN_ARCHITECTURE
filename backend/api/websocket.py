"""
websocket.py
WebSocket connection manager + broadcast to all connected clients.
"""
import json
import asyncio
from typing import Set
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.add(ws)
        print(f"[WS] Client connected — total={len(self._connections)}")

    def disconnect(self, ws: WebSocket):
        self._connections.discard(ws)
        print(f"[WS] Client disconnected — total={len(self._connections)}")

    async def broadcast(self, data: dict):
        if not self._connections:
            return
        msg = json.dumps(data, default=_json_serial)
        dead = set()
        tasks = [ws.send_text(msg) for ws in self._connections]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for ws, res in zip(list(self._connections), results):
            if isinstance(res, Exception):
                dead.add(ws)
        for ws in dead:
            self._connections.discard(ws)

    @property
    def n_clients(self) -> int:
        return len(self._connections)


def _json_serial(obj):
    """JSON serialiser for numpy types."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
