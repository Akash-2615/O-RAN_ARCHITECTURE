"""
oran_xapp.py
O-RAN xApp interface layer simulation.
Mimics near-RT RIC at 10ms timescale: E2 subscription, control output, xApp log.
"""
import time
from collections import deque
from typing import Dict, List
import numpy as np


class XAppLog:
    def __init__(self, max_entries: int = 500):
        self._entries: deque = deque(maxlen=max_entries)

    def log(self, level: str, source: str, msg: str):
        self._entries.appendleft({
            "ts":     time.strftime("%H:%M:%S"),
            "level":  level,
            "source": source,
            "msg":    msg,
        })

    def get(self, n: int = 50) -> list:
        return list(self._entries)[:n]


class ORANxApp:
    """
    Simulates the O-RAN near-RT RIC xApp layer:
    - Subscribes to E2 metrics (PRBs, traffic, CSI)
    - Applies DRL decisions as RIC control messages
    - Resolves conflicts between multiple xApps
    - Emits console-style log entries
    """

    XAPP_NAMES = [
        "SliceManager-xApp",
        "InterferenceMitigation-xApp",
        "MobilityMgmt-xApp",
        "PowerControl-xApp",
    ]

    def __init__(self, n_cells: int):
        self.n_cells  = n_cells
        self.log      = XAppLog()
        self._tti     = 0
        self._decisions: List[Dict] = []
        self._e2_metrics: Dict = {}
        self._xapp_status = {name: "ACTIVE" for name in self.XAPP_NAMES}
        self._conflict_count = 0

        # E2 interface simulated latency histogram
        self._e2_latencies: deque = deque(maxlen=200)

        self.log.log("INFO", "RIC", "O-RAN near-RT RIC initialised")
        for name in self.XAPP_NAMES:
            self.log.log("INFO", name, f"xApp registered on RIC — subscribing to E2 nodes")

    # ------------------------------------------------------------------ #
    def ingest_e2_metrics(self, snapshots: dict, metrics: dict):
        """Simulate E2 report subscription ingestion."""
        self._tti += 1
        lat = float(np.random.exponential(1.5))   # ms E2 latency
        self._e2_latencies.append(lat)

        self._e2_metrics = {
            "tti":           self._tti,
            "n_cells":       self.n_cells,
            "e2_latency_ms": round(lat, 2),
            "kpis":          metrics,
        }

        if self._tti % 50 == 0:
            self.log.log("DEBUG", "SliceManager-xApp",
                         f"E2 report: TTI={self._tti} | "
                         f"Utility={metrics.get('network_utility', 0):.3f} | "
                         f"SLA_viol={metrics.get('sla_violation', 0):.3f}")

    # ------------------------------------------------------------------ #
    def emit_control(self, prb_maps: List[Dict], powers: List[np.ndarray]):
        """Emit RIC control messages for each BS."""
        decisions = []
        for bsid, (pm, pw) in enumerate(zip(prb_maps, powers)):
            total_prbs = sum(pm.values())
            mean_pw    = float(np.mean(pw)) if len(pw) else 0.0
            decisions.append({
                "bs_id":      bsid,
                "prb_alloc":  pm,
                "mean_power": round(mean_pw, 2),
                "total_prbs": total_prbs,
            })

            if self._tti % 100 == 0:
                self.log.log(
                    "INFO", "SliceManager-xApp",
                    f"BS{bsid}: PRBs={pm} | AvgPwr={mean_pw:.1f}dBm"
                )

        # Conflict detection (simplified): if total PRBs exceed capacity
        for d in decisions:
            if d["total_prbs"] > 100:
                self._conflict_count += 1
                self.log.log("WARN", "SliceManager-xApp",
                             f"BS{d['bs_id']}: PRB over-allocation detected — trimming")

        self._decisions = decisions

        if self._tti % 200 == 0:
            self.log.log("INFO", "PowerControl-xApp",
                         f"Power control applied across {self.n_cells} cells")
            self.log.log("INFO", "MobilityMgmt-xApp",
                         f"Handover assessment — no triggers this window")

    # ------------------------------------------------------------------ #
    def get_status(self) -> dict:
        e2_lats = list(self._e2_latencies)
        return {
            "tti":              self._tti,
            "xapp_status":      self._xapp_status,
            "conflict_count":   self._conflict_count,
            "e2_latency_mean":  round(float(np.mean(e2_lats)), 2) if e2_lats else 0.0,
            "e2_latency_p99":   round(float(np.percentile(e2_lats, 99)), 2) if e2_lats else 0.0,
            "last_decisions":   self._decisions,
            "log":              self.log.get(30),
        }

    def get_topology(self) -> dict:
        """Return xApp topology for UI visualisation."""
        return {
            "ric": {"id": "near-RT-RIC", "type": "RIC"},
            "xapps": [{"id": n, "status": self._xapp_status[n]}
                       for n in self.XAPP_NAMES],
            "cells": [{"id": f"BS{i}", "type": "gNB"}
                       for i in range(self.n_cells)],
        }
