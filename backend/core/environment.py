"""
environment.py
5G Network Slicing MDP Environment.
State, action, reward as per the paper (Li et al. formulation + SAC extensions).
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from .network_state import NetworkSnapshot
from .traffic_model import SliceType


# ── State / Action / Config ───────────────────────────────────────────── #

@dataclass
class EnvConfig:
    total_prbs:      int   = 100
    max_power_dbm:   float = 43.0
    alpha:           float = 10.0   # Extreme Greed — Throughput is the only focus
    beta:            float = 0.15   # energy weight (gentle)
    gamma:           float = 3.0    # Amplified SLA Penalty — agents must respect latency limits
    delta:           float = 1.5    # Boosted to 1.5 — Stop resource hogging & rescue BS-2
    community_bonus: float = 0.5    # bonus when Jain fairness is high
    urllc_latency_ms: float = 5.0   # hard URLLC deadline


class Environment:
    """
    Wraps a NetworkSnapshot into RL state vector, validates actions,
    computes reward.  One Environment instance per base station.
    """

    def __init__(self, config: EnvConfig = None):
        self.cfg = config or EnvConfig()

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    def build_state(self, snap: NetworkSnapshot) -> np.ndarray:
        """
        Build normalised state vector from NetworkSnapshot.
        Dims: [prbs_norm, slice_class×n, cqi_mean×n, weight×n, fairness×n, delay×n]
        Fixed output: 6 * n_slices + 1
        """
        n_sl = len(snap.slices)
        state = []

        # Global: PRBs available (normalised)
        state.append(snap.prbs_avail / snap.total_prbs)

        for sl in snap.slices:
            # Slice class one-hot encoded as 0/1/2
            cls = {"eMBB": 0.0, "URLLC": 0.5, "mMTC": 1.0}[sl.slice_type.value]
            state.append(cls)

            # Mean CQI across UEs of this slice (normalised 0-1)
            slice_cqis = [ch.cqi for ch in snap.channels]  # simplified: all UEs
            state.append(np.mean(slice_cqis) / 15.0)

            # Priority weight (operator defined, 0-1)
            state.append(sl.priority)

            # Fairness index proxy (queue saturation)
            max_q = 1e7
            state.append(min(sl.queue_bytes / max_q, 1.0))

            # Queue delay (ms) normalised by latency requirement
            if sl.arrival_rate > 0:
                delay_ms = (sl.queue_bytes / sl.arrival_rate) * 10.0
            else:
                delay_ms = 0.0
            state.append(min(delay_ms / sl.latency_req, 2.0))

        return np.array(state, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        # 1 + 5 * n_slices  (n_slices inferred at runtime)
        return None  # dynamic — computed on first call

    # ------------------------------------------------------------------ #
    # Action                                                              #
    # ------------------------------------------------------------------ #
    def decode_action(self, raw_action: np.ndarray, snap: NetworkSnapshot
                      ) -> Tuple[Dict[int, int], np.ndarray]:
        """
        Decode raw actor output → (prbs_per_slice, tx_power_per_ue).
        raw_action shape: [n_slices, 1_power_scalar]
        PRB allocation: softmax over slice weights, then integer round.
        Power: sigmoid scaled to [0, max_power].
        """
        n_sl = len(snap.slices)
        prb_logits = raw_action[:n_sl]
        power_raw  = raw_action[n_sl] if len(raw_action) > n_sl else 0.0

        # Softmax → proportional PRB allocation
        prb_logits = prb_logits - prb_logits.max()
        prb_probs  = np.exp(prb_logits) / np.exp(prb_logits).sum()
        prb_per_slice = np.floor(prb_probs * snap.prbs_avail).astype(int)

        # Enforce minimums
        for i, sl in enumerate(snap.slices):
            prb_per_slice[i] = max(prb_per_slice[i], sl.min_prbs)

        # Clip total
        total = prb_per_slice.sum()
        if total > snap.prbs_avail:
            excess = total - snap.prbs_avail
            # Trim from largest slice
            idx = np.argsort(prb_per_slice)[::-1]
            for i in idx:
                trim = min(prb_per_slice[i] - snap.slices[i].min_prbs, excess)
                prb_per_slice[i] -= trim
                excess -= trim
                if excess <= 0:
                    break

        p_map = {sl.slice_id: int(prb_per_slice[i])
                 for i, sl in enumerate(snap.slices)}

        # Sigmoid power control (scalar → all UEs)
        power_norm = 1.0 / (1.0 + np.exp(-float(power_raw)))
        tx_power   = power_norm * self.cfg.max_power_dbm
        n_ues      = len(snap.channels)
        tx_powers  = np.full(n_ues, tx_power, dtype=np.float32)

        return p_map, tx_powers

    # ------------------------------------------------------------------ #
    # Reward                                                              #
    # ------------------------------------------------------------------ #
    def compute_reward(self, snap: NetworkSnapshot,
                       prb_map: Dict[int, int],
                       tx_powers: np.ndarray,
                       throughputs: Dict[int, float]) -> Tuple[float, Dict]:
        """
        R = α·throughput_norm − β·energy − γ·SLA_penalty − δ·unfairness
        Returns reward scalar + per-component breakdown dict.
        """
        cfg = self.cfg
        slices = snap.slices

        # ── Throughput component ──────────────────────────────────────
        max_th_per_slice = 1e7  # 10 Mbps normalisation matching Rayleigh fade
        thr_norm = sum(
            sl.priority * throughputs.get(sl.slice_id, 0.0) / max_th_per_slice
            for sl in slices
        )

        # ── Energy component ─────────────────────────────────────────
        total_power_mw = np.sum(10 ** (tx_powers / 10))  # dBm → mW
        energy = total_power_mw * 0.01 / 1000  # scale to reasonable range

        # ── SLA penalty ──────────────────────────────────────────────
        sla_penalty = 0.0
        for sl in slices:
            # Predict the actual remaining queue after this TTI's transmission
            served_bps = throughputs.get(sl.slice_id, 0.0)
            served_bytes = (served_bps / 8.0) * 0.01  # 10ms TTI
            remaining_queue = max(0.0, sl.queue_bytes - served_bytes)

            if sl.arrival_rate > 0:
                delay_ms = (remaining_queue / sl.arrival_rate) * 10.0
            else:
                delay_ms = 0.0
                
            if delay_ms > sl.latency_req:
                # Proportional penalty: 10 per ms over limit
                over_ratio = (delay_ms / sl.latency_req) - 1.0
                sla_penalty += min(1000.0, 10.0 * over_ratio * sl.latency_req)

        # URLLC hard constraint
        for sl in slices:
            if sl.slice_type == SliceType.URLLC:
                served_bps = throughputs.get(sl.slice_id, 0.0)
                served_bytes = (served_bps / 8.0) * 0.01
                remaining_queue = max(0.0, sl.queue_bytes - served_bytes)
                
                if sl.arrival_rate > 0:
                    delay_ms = (remaining_queue / sl.arrival_rate) * 10.0
                    if delay_ms > cfg.urllc_latency_ms:
                        # Major penalty: harder wall for URLLC breaches
                        sla_penalty += 5000.0

        # ── Fairness penalty (Jain's index deviation) ─────────────────
        th_vals = [throughputs.get(sl.slice_id, 0.0) for sl in slices]
        total_th = sum(th_vals) + 1e-9
        shares = [t / total_th for t in th_vals]
        weights = [sl.priority for sl in slices]
        w_total = sum(weights) + 1e-9
        w_shares = [w / w_total for w in weights]
        fairness_err = sum(abs(s - w) for s, w in zip(shares, w_shares))
        
        # Jain's Fairness Index for the community reward
        sq_sum = sum(t**2 for t in th_vals)
        jains = (total_th**2) / (len(th_vals) * sq_sum + 1e-9)
        
        # Dynamic Equilibrium: Bonus scales with how well the community is doing
        # But only if no one is being starved (min threshold 500kbps)
        min_th_val = min(th_vals) if th_vals else 0
        avg_th_val = total_th / len(th_vals)
        fairness_bonus = cfg.community_bonus * jains if min_th_val > 5e5 else 0.0

        # Dominance Penalty: soft nudge if any agent is far above average while others starve
        dominance_penalty = 0.0
        for val in th_vals:
            if val > 2 * avg_th_val and min_th_val < avg_th_val * 0.5:
                dominance_penalty += 0.3  # Soft nudge, not a sledgehammer

        # ── Final Composition ───────────────────────────────────────── #
        # We add a +5.0 baseline offset to ensure rewards stay positive for the user
        reward = (cfg.alpha * thr_norm
                  - cfg.beta  * energy
                  - cfg.gamma * sla_penalty / 500.0   # /500 doubles sensitivity vs old /1000
                  - cfg.delta * fairness_err
                  - dominance_penalty
                  + fairness_bonus
                  + 5.0)

        breakdown = {
            "throughput_norm": round(thr_norm, 4),
            "energy":          round(energy, 6),
            "sla_penalty":     round(sla_penalty, 1),
            "fairness_err":    round(fairness_err, 4),
            "reward":          round(reward, 4),
        }
        return float(reward), breakdown

    # ------------------------------------------------------------------ #
    # Throughput calculation                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_throughputs(snap: NetworkSnapshot,
                            prb_map: Dict[int, int]) -> Dict[int, float]:
        """
        Shannon capacity: C = B * SE * n_prbs
        RB bandwidth = 180 kHz (15 kHz SCS × 12 subcarriers)
        """
        rb_bw_hz = 180e3
        throughputs: Dict[int, float] = {}

        # Group UEs by slice
        ue_by_slice: Dict[int, list] = {}
        for i, ch in enumerate(snap.channels):
            # simplified: round-robin UE → slice mapping
            sl_idx = i % len(snap.slices)
            sl_id  = snap.slices[sl_idx].slice_id
            ue_by_slice.setdefault(sl_id, []).append(ch)

        for sl in snap.slices:
            n_prbs = prb_map.get(sl.slice_id, 0)
            ue_chs = ue_by_slice.get(sl.slice_id, [])
            if ue_chs:
                mean_se = np.mean([ch.spectral_eff for ch in ue_chs])
            else:
                mean_se = 1.0
            throughputs[sl.slice_id] = n_prbs * rb_bw_hz * mean_se  # bits/s

        return throughputs
