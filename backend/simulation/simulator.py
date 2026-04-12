"""
simulator.py
Main 10ms TTI simulation loop.
Orchestrates: NetworkState → MultiAgent → FederatedCoordinator →
              MetricsTracker → ORANxApp → WebSocket broadcast.
"""
import asyncio
import time
import numpy as np
from typing import Optional, Callable, Dict, Any

from core.network_state import NetworkStateManager
from core.environment import EnvConfig
from agents.multi_agent import MultiAgentCoordinator
from federated.fed_coordinator import FederatedCoordinator
from explainability.shap_explainer import SHAPExplainer
from explainability.importance import ImportanceTracker
from simulation.metrics import MetricsTracker
from simulation.oran_xapp import ORANxApp


# ── Simulation Configuration ──────────────────────────────────────────── #
N_CELLS             = 3
UES_PER_CELL        = 15
N_SLICES_PER_TYPE   = 2   # → 6 total slices
TOTAL_PRBS          = 100
TTI_SLEEP_S         = 0.05   # wall-clock between TTIs (50ms for UI smoothness)
FED_INTERVAL        = 50     # TTIs between federated rounds
TRAIN_EVERY         = 4      # train every 4 TTIs
STATE_DIM           = 31     # 1 + 5*6
ACTION_DIM          = 7      # 6 slice PRB logits + 1 power scalar


class Simulator:
    def __init__(self):
        self.running   = False
        self._task: Optional[asyncio.Task] = None
        self._broadcast_cb: Optional[Callable] = None

        # Core subsystems
        self.net    = NetworkStateManager(
            n_cells=N_CELLS, ues_per_cell=UES_PER_CELL,
            n_slices_per_type=N_SLICES_PER_TYPE,
            total_prbs=TOTAL_PRBS, seed=42)

        self.mac    = MultiAgentCoordinator(
            n_agents=N_CELLS,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            env_config=EnvConfig(),
            train_every=TRAIN_EVERY)

        self.fed    = FederatedCoordinator(
            n_agents=N_CELLS, fed_interval=FED_INTERVAL)

        self.metrics = MetricsTracker(window=300, ema_alpha=0.08)

        self.xapp   = ORANxApp(n_cells=N_CELLS)

        # Explainability — one explainer per agent
        self.explainers = [
            SHAPExplainer(self.mac.agents[i].actor, STATE_DIM, n_samples=10)
            for i in range(N_CELLS)
        ]
        self.imp_tracker = ImportanceTracker(N_CELLS)

        self._tti       = 0
        self._last_frame: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def set_broadcast(self, cb: Callable):
        self._broadcast_cb = cb

    # ------------------------------------------------------------------ #
    async def run(self):
        """Main async TTI loop."""
        self.running = True
        while self.running:
            t0 = time.perf_counter()
            try:
                frame = self._step()
                self._last_frame = frame
                if self._broadcast_cb:
                    await self._broadcast_cb(frame)
            except Exception as e:
                print(f"[Simulator] step error TTI={self._tti}: {e}")
                import traceback; traceback.print_exc()

            elapsed = time.perf_counter() - t0
            sleep   = max(0.0, TTI_SLEEP_S - elapsed)
            await asyncio.sleep(sleep)

    def start(self):
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.run())

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

    def reset(self):
        self.stop()
        self.__init__()

    # ------------------------------------------------------------------ #
    def _step(self) -> Dict[str, Any]:
        self._tti += 1

        # 1. Network step (mobility + channel + traffic)
        snapshots = self.net.step()

        # 2. Multi-agent step
        result = self.mac.step(snapshots)

        prb_maps     = result["prb_maps"]
        tx_powers    = result["tx_powers"]
        throughputs  = result["throughputs"]
        reward_brkdwn= result["reward_breakdowns"]
        states       = result["states"]

        # 3. Drain queues
        for bs_id, th_map in throughputs.items():
            self.net.drain_queues(th_map)

        # 4. Apply TX powers back
        for bs_id, powers in tx_powers.items():
            self.net.apply_power(bs_id, powers)

        # 5. Federated round (every FED_INTERVAL TTIs)
        fed_event = None
        for i, agent in enumerate(self.mac.agents):
            rwd = reward_brkdwn[i]["reward"] if i < len(reward_brkdwn) else 0.0
            self.fed.record_reward(i, rwd)
        did_fed = self.fed.step(self.mac.agents)
        if did_fed:
            fed_event = self.fed.last_event

        # 6. Metrics
        slices_all = [list(snapshots[bid].slices) for bid in sorted(snapshots)]
        powers_all = [tx_powers[bid].tolist() for bid in sorted(snapshots)]
        kpi = self.metrics.record(
            tti=self._tti,
            reward_breakdowns=reward_brkdwn,
            throughputs_per_bs=[throughputs[bid] for bid in sorted(snapshots)],
            slices_per_bs=slices_all,
            tx_powers_per_bs=powers_all,
        )

        # 7. O-RAN xApp
        self.xapp.ingest_e2_metrics(snapshots, self.metrics.get_current())
        self.xapp.emit_control(
            [prb_maps[bid] for bid in sorted(prb_maps)],
            [tx_powers[bid] for bid in sorted(tx_powers)],
        )

        # 8. Explainability (every 10 TTIs, for first agent)
        exp_data = {}
        if self._tti % 10 == 0:
            for bs_id, state in states.items():
                imp = self.explainers[bs_id].explain(state)
                self.imp_tracker.record(bs_id, imp)
                exp_data[bs_id] = imp

        # 9. Build broadcast frame
        ue_positions = self.net.get_ue_positions()
        bs_positions = self.net.get_bs_positions()

        # Serialize slice data
        slices_json = []
        if 0 in snapshots:
            for sl in snapshots[0].slices:
                slices_json.append({
                    "slice_id":   sl.slice_id,
                    "slice_type": sl.slice_type.value,
                    "queue_bytes":round(sl.queue_bytes, 1),
                    "arrival_rate":round(sl.arrival_rate, 1),
                    "latency_req": sl.latency_req,
                    "priority":   sl.priority,
                    "prbs": prb_maps.get(0, {}).get(sl.slice_id, 0),
                })

        prb_maps_json = {str(k): {str(sk): sv for sk, sv in v.items()}
                         for k, v in prb_maps.items()}

        frame = {
            "type":       "sim_frame",
            "tti":        self._tti,
            "kpi":        {
                "network_utility": kpi.network_utility,
                "sla_violation":   kpi.sla_violation,
                "energy_kwh_h":    kpi.energy_kwh_h,
                "fairness_index":  kpi.fairness_index,
                "throughputs":     {str(k): round(v/1e6, 3) for k, v in kpi.throughputs.items()},
                "latencies":       {str(k): v for k, v in kpi.latencies.items()},
            },
            "slices":     slices_json,
            "prb_maps":   prb_maps_json,
            "ue_positions": ue_positions,
            "bs_positions": bs_positions,
            "agents":     self.mac.get_agent_statuses(),
            "train_metrics": self.mac.get_train_metrics(),
            "federated":  self.fed.get_status(),
            "fed_event":  fed_event,
            "xapp":       self.xapp.get_status(),
            "explainability": {
                str(k): v for k, v in exp_data.items()
            },
            "importance_global": self.imp_tracker.get_global_mean(),
        }
        return frame

    # ------------------------------------------------------------------ #
    def get_last_frame(self) -> Dict:
        return self._last_frame

    def get_metrics_history(self, n: int = 200) -> list:
        return self.metrics.get_history(n)

    def get_status(self) -> dict:
        return {
            "running":  self.running,
            "tti":      self._tti,
            "n_cells":  N_CELLS,
            "n_slices": N_SLICES_PER_TYPE * 3,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "summary":  self.metrics.get_summary(),
        }
