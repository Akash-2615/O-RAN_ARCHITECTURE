"""
importance.py
Rolling feature importance tracker across all agents.
"""
from typing import Dict, List
import numpy as np


class ImportanceTracker:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self._per_agent: Dict[int, List[Dict]] = {i: [] for i in range(n_agents)}

    def record(self, bs_id: int, importance: Dict[str, float]):
        self._per_agent[bs_id].append(importance)
        if len(self._per_agent[bs_id]) > 500:
            self._per_agent[bs_id] = self._per_agent[bs_id][-500:]

    def get_latest(self, bs_id: int) -> Dict[str, float]:
        hist = self._per_agent[bs_id]
        return hist[-1] if hist else {}

    def get_global_mean(self) -> Dict[str, float]:
        all_keys: set = set()
        for records in self._per_agent.values():
            if records:
                all_keys.update(records[-1].keys())

        result = {}
        for k in all_keys:
            vals = []
            for records in self._per_agent.values():
                if records and k in records[-1]:
                    vals.append(records[-1][k])
            result[k] = round(float(np.mean(vals)), 4) if vals else 0.0
        return result
