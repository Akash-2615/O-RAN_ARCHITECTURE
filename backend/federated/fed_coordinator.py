"""
fed_coordinator.py
Manages federated learning rounds across all BS agents.
Collects params, runs attention aggregation, broadcasts global model.
"""
from typing import List, Dict
from collections import deque
import numpy as np
from .aggregator import AttentionAggregator, AgentReport


class FederatedCoordinator:
    """
    Orchestrates federated rounds:
      1. Collect actor params + performance from each agent
      2. Compute attention weights
      3. Aggregate params → global model
      4. Broadcast back to all agents
    """

    def __init__(self, n_agents: int, fed_interval: int = 50):
        self.n_agents     = n_agents
        self.fed_interval = fed_interval   # TTIs between rounds
        self.aggregator   = AttentionAggregator(n_agents)
        self.round_count  = 0
        self.tti_counter  = 0

        # Rolling reward window per agent for performance score
        self.reward_windows: Dict[int, deque] = {
            i: deque(maxlen=100) for i in range(n_agents)
        }

        # Visualisation state
        self.last_event: Dict = {}

    # ------------------------------------------------------------------ #
    def record_reward(self, bs_id: int, reward: float):
        self.reward_windows[bs_id].append(reward)

    def _agent_report(self, agent, bs_id: int) -> AgentReport:
        rewards = list(self.reward_windows[bs_id])
        perf  = float(np.mean(rewards))  if rewards else 0.0
        var   = float(np.var(rewards))   if rewards else 1.0
        return AgentReport(
            bs_id=bs_id,
            grad=agent.get_gradient_snapshot()["grad"],
            performance=perf,
            variance=var,
        )

    # ------------------------------------------------------------------ #
    def step(self, agents: list) -> bool:
        """
        Call every TTI. Returns True if a federated round was executed.
        """
        self.tti_counter += 1
        if self.tti_counter % self.fed_interval != 0:
            return False

        reports = [self._agent_report(ag, i) for i, ag in enumerate(agents)]
        weights = self.aggregator.compute_weights(reports)

        # Collect actor parameters
        all_params = [ag.get_actor_params() for ag in agents]

        # Aggregate
        global_params = self.aggregator.aggregate_params(all_params, weights)

        # Broadcast
        for ag in agents:
            ag.apply_global_params(global_params)

        self.aggregator.record_round(weights, reports)
        self.round_count += 1

        self.last_event = {
            "round":        self.round_count,
            "tti":          self.tti_counter,
            "weights":      weights.tolist(),
            "performances": [r.performance for r in reports],
        }
        return True

    # ------------------------------------------------------------------ #
    def get_status(self) -> dict:
        return {
            "round_count":   self.round_count,
            "fed_interval":  self.fed_interval,
            "tti_counter":   self.tti_counter,
            "aggregator":    self.aggregator.get_stats(),
            "last_event":    self.last_event,
        }
