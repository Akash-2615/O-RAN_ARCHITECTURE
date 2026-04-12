"""
network_state.py
Live network state manager — tracks all BS, UE, slice, and channel state.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from .channel_model import ChannelModel, ChannelState
from .traffic_model import TrafficModel, SliceType, SliceTrafficState


@dataclass
class BaseStation:
    bs_id:       int
    x:           float          # position metres
    y:           float
    z:           float
    total_prbs:  int    = 100
    max_power_dbm: float = 43.0
    cell_radius_m: float = 250.0


@dataclass
class UserEquipment:
    ue_id:    int
    slice_id: int
    x:        float
    y:        float
    velocity: float   # m/s
    heading:  float   # radians
    distance_to_bs: float = 0.0
    channel: ChannelState = None


@dataclass
class NetworkSnapshot:
    """Full state of the network at one TTI — fed into DRL state vector."""
    tti:          int
    bs_id:        int
    prbs_avail:   int
    slices:       List[SliceTrafficState]
    channels:     List[ChannelState]       # per UE
    tx_powers:    np.ndarray               # current power per UE (dBm)
    total_prbs:   int = 100


class NetworkStateManager:
    """
    Manages multi-BS network state: positions, mobility, channel updates.
    """

    def __init__(self, n_cells: int = 3, ues_per_cell: int = 20,
                 n_slices_per_type: int = 2, total_prbs: int = 100,
                 seed: int = 42):
        self.n_cells = n_cells
        self.total_prbs = total_prbs
        self.rng = np.random.default_rng(seed)

        # Place base stations in hexagonal layout
        self.base_stations = self._init_base_stations(n_cells)

        # Build one traffic model (shared slice definitions across cells)
        self.traffic_model = TrafficModel(
            n_slices_per_type=n_slices_per_type,
            total_prbs=total_prbs,
            rng=self.rng,
        )

        # One channel model per BS
        self.channel_models: Dict[int, ChannelModel] = {
            bs.bs_id: ChannelModel(rng=np.random.default_rng(seed + bs.bs_id))
            for bs in self.base_stations
        }

        # UEs per BS
        self.ues: Dict[int, List[UserEquipment]] = {
            bs.bs_id: self._init_ues(bs, ues_per_cell, self.traffic_model.n_slices)
            for bs in self.base_stations
        }

        # TX powers (initialise to max)
        self.tx_powers: Dict[int, np.ndarray] = {
            bs.bs_id: np.full(ues_per_cell, bs.max_power_dbm)
            for bs in self.base_stations
        }

        # PRBs available per BS (can change with interference)
        self.prbs_avail: Dict[int, int] = {bs.bs_id: total_prbs for bs in self.base_stations}

        self._tti = 0

    # ------------------------------------------------------------------ #
    def _init_base_stations(self, n: int) -> List[BaseStation]:
        stations = []
        positions = [(0, 150), (-300, -200), (300, -200), (0, -400), (-400, 200), (400, 200)]
        for i in range(n):
            x, y = positions[i % len(positions)]
            stations.append(BaseStation(bs_id=i, x=x, y=y, z=30.0))
        return stations

    def _init_ues(self, bs: BaseStation, n: int, n_slices: int) -> List[UserEquipment]:
        ues = []
        for i in range(n):
            r = self.rng.uniform(10, bs.cell_radius_m)
            theta = self.rng.uniform(0, 2 * np.pi)
            ue = UserEquipment(
                ue_id=i,
                slice_id=int(self.rng.integers(0, n_slices)),
                x=bs.x + r * np.cos(theta),
                y=bs.y + r * np.sin(theta),
                velocity=self.rng.uniform(0, 30),   # 0–30 m/s
                heading=self.rng.uniform(0, 2 * np.pi),
                distance_to_bs=r,
            )
            ues.append(ue)
        return ues

    # ------------------------------------------------------------------ #
    def _update_mobility(self, bs: BaseStation):
        """Random walk mobility model per UE."""
        dt = 0.01  # 10ms TTI in seconds
        for ue in self.ues[bs.bs_id]:
            # Update heading with small random turn
            ue.heading += self.rng.normal(0, 0.1)
            dx = ue.velocity * np.cos(ue.heading) * dt
            dy = ue.velocity * np.sin(ue.heading) * dt
            ue.x += dx
            ue.y += dy
            ue.distance_to_bs = np.sqrt((ue.x - bs.x)**2 + (ue.y - bs.y)**2)
            # Reflect off cell boundary
            if ue.distance_to_bs > bs.cell_radius_m:
                ue.heading += np.pi

    # ------------------------------------------------------------------ #
    def step(self) -> Dict[int, NetworkSnapshot]:
        """Advance one TTI — update mobility, channels, traffic."""
        self._tti += 1
        slices = self.traffic_model.step()

        snapshots: Dict[int, NetworkSnapshot] = {}
        for bs in self.base_stations:
            self._update_mobility(bs)
            cm = self.channel_models[bs.bs_id]
            distances = np.array([ue.distance_to_bs for ue in self.ues[bs.bs_id]])
            powers = self.tx_powers[bs.bs_id]
            channels = cm.batch_update(distances, powers)

            for ue, ch in zip(self.ues[bs.bs_id], channels):
                ue.channel = ch

            snapshots[bs.bs_id] = NetworkSnapshot(
                tti=self._tti,
                bs_id=bs.bs_id,
                prbs_avail=self.prbs_avail[bs.bs_id],
                slices=slices,
                channels=channels,
                tx_powers=powers.copy(),
                total_prbs=self.total_prbs,
            )

        return snapshots

    def drain_queues(self, throughputs: Dict[int, float]):
        self.traffic_model.drain_queues(throughputs)

    def apply_power(self, bs_id: int, powers: np.ndarray):
        self.tx_powers[bs_id] = np.clip(
            powers, 0, self.base_stations[bs_id].max_power_dbm
        )

    def get_ue_positions(self) -> dict:
        """Return UE positions for 3D visualisation."""
        result = {}
        for bs in self.base_stations:
            result[bs.bs_id] = [
                {"ue_id": ue.ue_id, "x": ue.x, "y": ue.y,
                 "slice_id": ue.slice_id,
                 "cqi": ue.channel.cqi if ue.channel else 0}
                for ue in self.ues[bs.bs_id]
            ]
        return result

    def get_bs_positions(self) -> list:
        return [{"bs_id": bs.bs_id, "x": bs.x, "y": bs.y, "z": bs.z}
                for bs in self.base_stations]

    @property
    def tti(self) -> int:
        return self._tti
