"""
actor_critic.py
SAC Actor + Twin Critic neural networks (PyTorch).
Architecture per paper: 3-layer FC (256-256-128/1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


def mlp(in_dim: int, hidden: list[int], out_dim: int, activation=nn.ReLU) -> nn.Sequential:
    layers = []
    dims = [in_dim] + hidden
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Gaussian policy network.
    Outputs mean + log_std for continuous action (power scalar + PRB logits).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256, 128)):
        super().__init__()
        self.net     = mlp(state_dim, list(hidden[:-1]), hidden[-1])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.ls_head = nn.Linear(hidden[-1], action_dim)

    def forward(self, state: torch.Tensor):
        h = F.relu(self.net(state))
        # Numerical Safety: Clamping h to avoid extreme growth
        h = torch.clamp(h, -100, 100)
        
        mu = self.mu_head(h)
        log_std = self.ls_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        
        # NaN Recovery: Ensure we never pass NaN to Normal()
        mu = torch.nan_to_num(mu, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0)
        
        return mu, log_std

    def sample(self, state: torch.Tensor):
        mu, log_std = self.forward(state)
        std  = log_std.exp()
        dist = Normal(mu, std)
        x_t  = dist.rsample()
        y_t  = torch.tanh(x_t)

        log_prob = dist.log_prob(x_t)
        # REDESIGN: Cap the Jacobian (Tanh Wall) to prevent 55.0 entropy spikes
        # We ensure 1 - y_t^2 is never smaller than 1e-4 to stop the log(0) explosion
        log_prob -= torch.log(torch.clamp(1 - y_t.pow(2), 1e-4, 1.0))
        log_prob  = torch.nan_to_num(log_prob, nan=-20.0)
        log_prob  = log_prob.sum(-1, keepdim=True)
        # Expert Telemetry: Also return the raw log_std for jitter tracking
        return y_t, log_prob, torch.tanh(mu), log_std

    def get_action_numpy(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.mu_head.weight.device)
            if deterministic:
                mu, _ = self.forward(s)
                action = torch.tanh(mu)
            else:
                action, _, _, _ = self.sample(s)
        return action.squeeze(0).cpu().numpy()


class Critic(nn.Module):
    """Single Q-network: Q(s,a) → scalar."""

    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(state_dim + action_dim, list(hidden), 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class TwinCritic(nn.Module):
    """Twin critics for SAC to reduce Q-value overestimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden)
        self.q2 = Critic(state_dim, action_dim, hidden)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def min_q(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


def soft_update(source: nn.Module, target: nn.Module, tau: float = 0.005):
    """Polyak averaging target network update."""
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
