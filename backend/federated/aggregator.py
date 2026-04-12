"""
aggregator.py
Attention-weighted federated gradient aggregation.
Implements: weight_i = exp(score_i) / Σ exp(score_j)
            score = recent_performance / variance
as described in the paper (section 5.3).
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class AgentReport:
    bs_id:       int
    grad:        np.ndarray
    performance: float    # recent avg reward
    variance:    float    # reward variance (stability)


class AttentionAggregator:
    """
    Computes attention weights from agent performance metrics,
    then returns weighted average of actor parameters.
    """

    def __init__(self, n_agents: int, eps: float = 1e-6):
        self.n_agents = n_agents
        self.eps = eps
        self.last_weights: np.ndarray = np.ones(n_agents) / n_agents
        self.round_history: List[Dict] = []

    def compute_weights(self, reports: List[AgentReport]) -> np.ndarray:
        """
        score_i = recent_perf_i / (variance_i + eps)
        weight_i = softmax(score_i)
        """
        scores = np.array([
            r.performance / (r.variance + self.eps)
            for r in reports
        ], dtype=np.float64)

        # Softmax
        scores -= scores.max()  # numerical stability
        exp_s   = np.exp(scores)
        weights = exp_s / (exp_s.sum() + self.eps)
        self.last_weights = weights
        return weights

    def aggregate_params(
        self,
        agent_params: List[List],  # List[agent] of List[tensor]
        weights: np.ndarray,
    ) -> List:
        """Weighted average of actor parameters (list of tensors)."""
        import torch
        n_layers = len(agent_params[0])
        global_params = []
        for l in range(n_layers):
            stack = torch.stack([agent_params[i][l] * float(weights[i])
                                  for i in range(len(agent_params))])
            global_params.append(stack.sum(dim=0))
        return global_params

    def record_round(self, weights: np.ndarray, reports: List[AgentReport]):
        self.round_history.append({
            "weights": weights.tolist(),
            "performances": [r.performance for r in reports],
            "variances":    [r.variance for r in reports],
        })
        if len(self.round_history) > 100:
            self.round_history = self.round_history[-100:]

    def get_stats(self) -> dict:
        return {
            "last_weights": self.last_weights.tolist(),
            "n_rounds":     len(self.round_history),
            "history":      self.round_history[-10:],
        }
