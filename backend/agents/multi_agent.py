"""
multi_agent.py
Multi-agent coordinator: manages N DRL agents (one per BS),
dispatches steps, collects results.
"""
import numpy as np
from typing import List, Dict, Optional
from .drl_agent import DRLAgent
from core.environment import Environment, EnvConfig
from core.network_state import NetworkSnapshot


class MultiAgentCoordinator:
    """
    Owns N DRLAgent instances (one per base station).
    Per TTI: build state → select action → decode → compute reward → store → train.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        env_config: EnvConfig = None,
        train_every: int = 4,
        device: str = "cpu",
    ):
        self.n_agents    = n_agents
        self.env_config  = env_config or EnvConfig()
        self.train_every = train_every

        self.env = Environment(self.env_config)

        self.agents: List[DRLAgent] = [
            DRLAgent(
                bs_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
            )
            for i in range(n_agents)
        ]

        self._prev_states: Dict[int, Optional[np.ndarray]] = {i: None for i in range(n_agents)}
        self._prev_actions: Dict[int, Optional[np.ndarray]] = {i: None for i in range(n_agents)}
        self._step_count = 0

        # Last training metrics
        self.last_train_metrics: List[Dict] = [{} for _ in range(n_agents)]

    # ------------------------------------------------------------------ #
    def step(self, snapshots: Dict[int, NetworkSnapshot]) -> Dict:
        """
        Run one TTI step for all agents.
        Returns dict with prb_maps, tx_powers, throughputs, reward_breakdowns.
        """
        self._step_count += 1
        prb_maps      = {}
        tx_powers_all = {}
        throughputs   = {}
        reward_brkdwn = []
        states        = {}

        for bs_id, snap in snapshots.items():
            agent = self.agents[bs_id]

            # Build state
            state = self.env.build_state(snap)
            state = self._pad_state(state, agent.state_dim)
            states[bs_id] = state

            # Select action
            action = agent.select_action(state, explore=True)
            
            # Divergence Guard: If action is NaN, reset the agent immediately
            if np.isnan(action).any():
                print(f"[MultiAgent] CRITICAL: Agent {bs_id} diverged (NaN). Resetting...")
                agent.reset_weights()
                action = agent.select_action(state, explore=True) # Retry with fresh weights

            # Decode action → PRB map + TX powers
            prb_map, powers = self.env.decode_action(action, snap)
            prb_maps[bs_id]      = prb_map
            tx_powers_all[bs_id] = powers

            # Compute throughput
            th = self.env.compute_throughputs(snap, prb_map)
            throughputs[bs_id] = th

            # Compute reward
            reward, breakdown = self.env.compute_reward(snap, prb_map, powers, th)
            reward_brkdwn.append(breakdown)

            # Store transition
            if self._prev_states[bs_id] is not None:
                agent.store(
                    self._prev_states[bs_id],
                    self._prev_actions[bs_id],
                    reward,
                    state,
                    breakdown=breakdown,
                    done=False,
                )

            self._prev_states[bs_id]  = state
            self._prev_actions[bs_id] = action

        # Train agents periodically
        if self._step_count % self.train_every == 0:
            for i, agent in enumerate(self.agents):
                metrics = agent.update()
                if metrics:
                    # Divergence Guard (Gradient Explosion): Reset if losses hit NaN
                    if np.isnan(metrics.get("critic", 0)) or np.isnan(metrics.get("actor", 0)):
                        print(f"[MultiAgent] CRITICAL: Training exploded for Agent {i}. Resetting...")
                        agent.reset_weights()
                    self.last_train_metrics[i] = metrics
            
            # Adaptive Rescue: Boost LR for underperforming agents
            if self._step_count % 20 == 0:
                self._adaptive_tuning(reward_brkdwn)
            
            # Phase 2: Neural Mimicry (Knowledge Injection)
            if self._step_count % 30 == 0:
                self._mimic_best_agent(reward_brkdwn)

        return {
            "prb_maps":        prb_maps,
            "tx_powers":       tx_powers_all,
            "throughputs":     throughputs,
            "reward_breakdowns": reward_brkdwn,
            "states":          states,
            "actions":         {bs_id: self._prev_actions[bs_id]
                                 for bs_id in snapshots},
        }

    def _adaptive_tuning(self, breakdowns: List[Dict]):
        """Detect underperforming agents and nudge their learning parameters."""
        rewards = [b["reward"] for b in breakdowns]
        avg_rwd = np.mean(rewards)
        
        for i, r in enumerate(rewards):
            agent = self.agents[i]
            # If agent is >20% below average or has negative reward, boost learning
            if r < avg_rwd * 0.8 or r < 0:
                agent.tune_lr(1.5)  # Boost LR to escape local minima
            else:
                # Naturally decay back to baseline if performing well
                agent.tune_lr(0.95) 
                # Ensure we don't drop below the original 3e-4 effectively
                # (Simple proportional logic for this demo)

    def _mimic_best_agent(self, breakdowns: List[Dict]):
        """Find the most successful agent and softly sync others to its actor."""
        rewards = [b["reward"] for b in breakdowns]
        best_idx = int(np.argmax(rewards))
        best_rwd = rewards[best_idx]
        leader_params = self.agents[best_idx].get_actor_params()
        
        for i, r in enumerate(rewards):
            if i == best_idx: continue
            
            # If agent is showing regressed/negative reward or huge delta, mimic leader aggressively
            if r < 0 or best_rwd - r > 0.5:
                rescuing = (r < -0.1)
                tau = 0.2 if rescuing else 0.1
                print(f"[MultiAgent] Agent {i} is mimicking Leader {best_idx} to jumpstart convergence (tau={tau}).")
                self.agents[i].soft_sync_actor(leader_params, tau=tau)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pad_state(state: np.ndarray, target_dim: int) -> np.ndarray:
        if len(state) < target_dim:
            return np.pad(state, (0, target_dim - len(state)))
        return state[:target_dim]

    # ------------------------------------------------------------------ #
    def get_agent_statuses(self) -> List[Dict]:
        return [ag.get_status() for ag in self.agents]

    def get_train_metrics(self) -> List[Dict]:
        return self.last_train_metrics

    @property
    def step_count(self) -> int:
        return self._step_count
