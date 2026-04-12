"""
shap_explainer.py
Perturbation-based feature attribution (SHAP-style) — no external dependency.
Explains each DRL allocation decision in terms of state features.
"""
import numpy as np
from typing import List, Dict


FEATURE_NAMES_TEMPLATE = ["prbs_avail"] + [
    f"sl{i}_{feat}"
    for i in range(6)
    for feat in ["class", "cqi", "priority", "queue_sat", "delay_norm"]
]


class SHAPExplainer:
    """
    Perturbation-based feature importance:
      For each feature f, compute Δoutput when f is masked → importance.
    Uses actor's action magnitude as the output proxy.
    """

    def __init__(self, actor, state_dim: int, n_samples: int = 20):
        self.actor      = actor
        self.state_dim  = state_dim
        self.n_samples  = n_samples
        self.feature_names = FEATURE_NAMES_TEMPLATE[:state_dim]

        # Rolling importance
        self._history: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def explain(self, state: np.ndarray) -> Dict[str, float]:
        """Return per-feature importance for this state."""
        import torch
        base_action = self.actor.get_action_numpy(state)
        base_mag    = float(np.linalg.norm(base_action))

        importances = np.zeros(len(state))
        for i in range(len(state)):
            deltas = []
            for _ in range(self.n_samples):
                perturbed = state.copy()
                perturbed[i] = np.random.uniform(0, 1)  # random ablation
                p_action = self.actor.get_action_numpy(perturbed)
                p_mag    = float(np.linalg.norm(p_action))
                deltas.append(abs(base_mag - p_mag))
            importances[i] = np.mean(deltas)

        # Normalise
        total = importances.sum() + 1e-9
        importances /= total

        self._history.append(importances.copy())
        if len(self._history) > 200:
            self._history = self._history[-200:]

        names = self.feature_names
        return {names[i]: round(float(importances[i]), 4)
                for i in range(len(state))}

    def get_heatmap(self, last_n: int = 50) -> List[List[float]]:
        """Return heatmap matrix [time × features] for UI."""
        window = self._history[-last_n:]
        return [row.tolist() for row in window]

    def get_mean_importance(self) -> Dict[str, float]:
        if not self._history:
            return {}
        mean = np.mean(self._history, axis=0)
        names = self.feature_names
        return {names[i]: round(float(mean[i]), 4) for i in range(len(mean))}
