"""
traffic_model.py
Generates realistic eMBB, URLLC, mMTC traffic for each slice per TTI.
"""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class SliceType(str, Enum):
    eMBB  = "eMBB"
    URLLC = "URLLC"
    mMTC  = "mMTC"


@dataclass
class SliceTrafficState:
    slice_id:     int
    slice_type:   SliceType
    queue_bytes:  float       # bytes waiting in queue
    arrival_rate: float       # bytes/s arriving this TTI
    latency_req:  float       # max tolerable latency ms
    priority:     float       # operator priority weight
    min_prbs:     int         # minimum guaranteed PRBs (increased to 18 for high-load protection)
    max_prbs:     int         # maximum allowed PRBs


class TrafficModel:
    """
    Generates per-TTI traffic arrivals for eMBB, URLLC, mMTC slices.
    Models: Poisson (eMBB), Periodic+burst (URLLC), Bursty IoT (mMTC)
    """

    def __init__(self, n_slices_per_type: int = 2, total_prbs: int = 100,
                 rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(0)
        self.total_prbs = total_prbs
        self.tti_ms = 10  # ms per TTI

        # Build slice list: n_slices_per_type of each type
        self.slices: list[SliceTrafficState] = []
        sid = 0
        for st in [SliceType.eMBB, SliceType.URLLC, SliceType.mMTC]:
            for _ in range(n_slices_per_type):
                self.slices.append(self._init_slice(sid, st))
                sid += 1

        self._t = 0  # time counter (TTIs)

    # ------------------------------------------------------------------ #
    def _init_slice(self, sid: int, st: SliceType) -> SliceTrafficState:
        cfg = {
            SliceType.eMBB:  dict(latency_req=100.0, priority=0.6, min_prbs=5,  max_prbs=100),
            SliceType.URLLC: dict(latency_req=5.0,   priority=1.0, min_prbs=12, max_prbs=30),
            SliceType.mMTC:  dict(latency_req=500.0,  priority=0.3, min_prbs=2,  max_prbs=20),
        }[st]
        return SliceTrafficState(
            slice_id=sid, slice_type=st,
            queue_bytes=0.0, arrival_rate=0.0,
            **cfg
        )

    # ------------------------------------------------------------------ #
    def _embb_arrival(self, slice_id: int) -> float:
        """Poisson with diurnal pattern, offset by slice_id for variety."""
        base_rate_mbps = 3.0 + (slice_id % 3) * 0.5  # Slight baseline variation per slice
        phase_shift = slice_id * (np.pi / 4.0)       # Phase shift to decouple peaks
        peak = 1.0 + 0.5 * np.sin(2 * np.pi * self._t / 1000.0 + phase_shift)
        lam = base_rate_mbps * peak * 1e6 / 8 * (self.tti_ms / 1000)
        return float(self.rng.poisson(lam))

    def _urllc_arrival(self) -> float:
        """Periodic small packets + rare bursts."""
        base_bytes = 50.0
        is_burst = self.rng.random() < 0.02
        burst = self.rng.uniform(100, 300) if is_burst else 0.0
        jitter = self.rng.normal(0, 20)
        return max(0.0, base_bytes + jitter + burst)

    def _mmtc_arrival(self) -> float:
        """Bursty IoT — many silent devices, occasional mass wakeup."""
        n_devices = 1000
        p_active = 0.005 + 0.02 * (self.rng.random() < 0.01)  # rare mass wakeup
        active = self.rng.binomial(n_devices, p_active)
        bytes_each = self.rng.exponential(10)
        return float(active * bytes_each)

    # ------------------------------------------------------------------ #
    def step(self) -> list[SliceTrafficState]:
        """Advance one TTI — update arrivals and queue depths."""
        self._t += 1
        for sl in self.slices:
            if sl.slice_type == SliceType.eMBB:
                arr = self._embb_arrival(sl.slice_id)
            elif sl.slice_type == SliceType.URLLC:
                arr = self._urllc_arrival()
            else:
                arr = self._mmtc_arrival()
                
            # EMA for Little's law average arrival rate (lambda). Lowered floor to 5.0 to reveal live mMTC/URLLC data spikes.
            sl.arrival_rate = max(5.0, 0.85 * sl.arrival_rate + 0.15 * arr)
            sl.queue_bytes = max(0.0, sl.queue_bytes + arr)
        return self.slices

    def drain_queues(self, throughputs: dict[int, float]):
        """Remove served bytes from each slice queue after allocation."""
        for sl in self.slices:
            served_bps = throughputs.get(sl.slice_id, 0.0)
            served_bytes = (served_bps / 8.0) * (self.tti_ms / 1000.0)
            sl.queue_bytes = max(0.0, sl.queue_bytes - served_bytes)

    @property
    def n_slices(self) -> int:
        return len(self.slices)
