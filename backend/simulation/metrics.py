"""
metrics.py
Real-time KPI tracker: network utility, SLA violation rate,
energy consumption (kWh/h), Jain's fairness index.
"""
import numpy as np
from collections import deque
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class KPIFrame:
    tti:            int
    network_utility: float
    sla_violation:  float   # fraction 0-1
    energy_kwh_h:   float
    fairness_index: float   # Jain's index
    throughputs:    Dict[int, float] = field(default_factory=dict)
    latencies:      Dict[int, float] = field(default_factory=dict)
    queue_depths:   Dict[int, float] = field(default_factory=dict)
    rewards:        Dict[int, float] = field(default_factory=dict)


class MetricsTracker:
    """Rolling KPI tracker with EMA smoothing."""

    def __init__(self, window: int = 200, ema_alpha: float = 0.1):
        self.window    = window
        self.ema_alpha = ema_alpha
        self._history: deque[KPIFrame] = deque(maxlen=window)

        # EMA state
        self._ema: Dict[str, float] = {
            "network_utility": 0.0,
            "sla_violation":   0.0,
            "energy_kwh_h":    0.0,
            "fairness_index":  1.0,
        }

        self._sla_violations = 0
        self._total_ttis     = 0

    # ------------------------------------------------------------------ #
    def _ema_update(self, key: str, val: float) -> float:
        self._ema[key] = (self.ema_alpha * val
                          + (1 - self.ema_alpha) * self._ema[key])
        return self._ema[key]

    # ------------------------------------------------------------------ #
    def record(
        self,
        tti: int,
        reward_breakdowns: List[Dict],
        throughputs_per_bs: List[Dict[int, float]],
        slices_per_bs: list,
        tx_powers_per_bs: list,
    ) -> KPIFrame:

        self._total_ttis += 1

        # Network utility = normalised mean throughput across all BSs (0-1)
        # Uses throughput_norm from reward breakdowns directly - same scale as reward function
        thr_norms = [d.get('throughput_norm', 0.0) for d in reward_breakdowns if d]
        thr_norm_mean = float(np.mean(thr_norms)) if thr_norms else 0.0
        # Normalise by 1.6: This targets 75-80% utility during high eMBB saturation
        utility = float(np.clip(thr_norm_mean / 1.6, 0.0, 1.0))

        # Keep reward stats separately for analytics
        rewards = {i: d.get("reward", 0.0)
                   for i, d in enumerate(reward_breakdowns)}

        # SLA violations (Slice-specific Granularity)
        # Instead of binary network-wide Red, we show the % of slices currently in violation.
        n_slices = sum(len(sls) for sls in slices_per_bs)
        sla_cnt = sum(1 for d in reward_breakdowns if d and d.get("sla_penalty", 0.0) > 0)
        sla_frac = sla_cnt / max(n_slices, 1)

        # Count this TTI as 'Violated' if any single slice has a penalty (for historical total)
        if sla_cnt > 0:
            self._sla_violations += 1

        # Energy: sum of all tx powers across all BS → kWh/h
        total_mw = 0.0
        for powers in tx_powers_per_bs:
            total_mw += float(np.sum(10 ** (np.array(powers) / 10)))
        energy_w    = total_mw / 1000
        energy_kwh  = energy_w / 1000  # W → kW, per TTI (÷3600*100=per hour proxy)

        # Jain's fairness index across all slice throughputs
        all_th: List[float] = []
        for th_map in throughputs_per_bs:
            all_th.extend(th_map.values())
        if all_th and sum(all_th) > 0:
            n  = len(all_th)
            s  = sum(all_th)
            s2 = sum(t**2 for t in all_th)
            jain = (s**2) / (n * s2 + 1e-9)
        else:
            jain = 1.0

        # Aggregate throughputs / latencies / queue depths
        agg_th: Dict[int, float] = {}
        agg_lat: Dict[int, float] = {}
        agg_q: Dict[int, float] = {}

        for slices in slices_per_bs:
            for sl in slices:
                sid = sl.slice_id
                agg_q[sid] = float(sl.queue_bytes)
                if sl.arrival_rate > 0:
                    lat = (sl.queue_bytes / sl.arrival_rate) * 10.0
                else:
                    lat = 0.0
                agg_lat[sid] = round(lat, 2)

        for th_map in throughputs_per_bs:
            for sid, th in th_map.items():
                agg_th[sid] = agg_th.get(sid, 0.0) + th

        frame = KPIFrame(
            tti=tti,
            network_utility=round(self._ema_update("network_utility", utility), 4),
            sla_violation=round(self._ema_update("sla_violation", sla_frac), 4),
            energy_kwh_h=round(self._ema_update("energy_kwh_h", energy_kwh), 4),
            fairness_index=round(self._ema_update("fairness_index", jain), 4),
            throughputs=agg_th,
            latencies=agg_lat,
            queue_depths=agg_q,
            rewards=rewards,
        )
        self._history.append(frame)
        return frame

    # ------------------------------------------------------------------ #
    def get_history(self, last_n: int = 100) -> List[dict]:
        return [
            {
                "tti":             f.tti,
                "network_utility": f.network_utility,
                "sla_violation":   f.sla_violation,
                "energy_kwh_h":    f.energy_kwh_h,
                "fairness_index":  f.fairness_index,
            }
            for f in list(self._history)[-last_n:]
        ]

    def get_current(self) -> dict:
        if not self._history:
            return {}
        f = self._history[-1]
        return {
            "tti":             f.tti,
            "network_utility": f.network_utility,
            "sla_violation":   f.sla_violation,
            "energy_kwh_h":    f.energy_kwh_h,
            "fairness_index":  f.fairness_index,
            "throughputs":     f.throughputs,
            "latencies":       f.latencies,
            "queue_depths":    f.queue_depths,
            "rewards":         f.rewards,
        }

    def get_summary(self) -> dict:
        if not self._history:
            return {}
        utils  = [f.network_utility for f in self._history]
        slas   = [f.sla_violation   for f in self._history]
        return {
            "avg_utility":      round(float(np.mean(utils)), 4),
            "min_utility":      round(float(np.min(utils)), 4),
            "max_utility":      round(float(np.max(utils)), 4),
            "avg_sla_violation":round(float(np.mean(slas)), 4),
            "sla_violation_ttis": self._sla_violations,
            "total_ttis":       self._total_ttis,
        }
