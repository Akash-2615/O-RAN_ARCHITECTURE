"""
channel_model.py
Rayleigh fading channel model with path loss, SINR, CQI/MCS mapping.
Based on 3GPP TR 38.901 simplified model.
"""
import numpy as np
from dataclasses import dataclass
from typing import List


# 3GPP CQI → MCS → spectral efficiency table (bits/s/Hz)
CQI_TABLE = {
    1:  0.15,
    2:  0.23,
    3:  0.38,
    4:  0.60,
    5:  0.88,
    6:  1.18,
    7:  1.48,
    8:  1.91,
    9:  2.41,
    10: 2.73,
    11: 3.32,
    12: 3.90,
    13: 4.52,
    14: 5.12,
    15: 5.55,
}


@dataclass
class ChannelState:
    sinr_db: float          # SINR in dB
    cqi: int                # CQI index (1–15)
    spectral_eff: float     # bits/s/Hz
    path_loss_db: float     # path loss in dB
    fading_gain: float      # instantaneous Rayleigh gain


class ChannelModel:
    """
    Per-UE channel model with Rayleigh fading + path loss.
    """

    def __init__(
        self,
        carrier_freq_ghz: float = 3.5,
        bs_power_dbm: float = 43.0,
        noise_figure_db: float = 7.0,
        bandwidth_mhz: float = 100.0,
        path_loss_exponent: float = 3.5,
        shadow_std_db: float = 8.0,
        rng: np.random.Generator = None,
    ):
        self.carrier_freq_ghz = carrier_freq_ghz
        self.bs_power_dbm = bs_power_dbm
        self.noise_figure_db = noise_figure_db
        self.bandwidth_mhz = bandwidth_mhz
        self.path_loss_exponent = path_loss_exponent
        self.shadow_std_db = shadow_std_db
        self.rng = rng or np.random.default_rng(42)

        # Thermal noise: N = kTB
        k_boltzmann = 1.38e-23
        T = 290  # Kelvin
        B = bandwidth_mhz * 1e6
        noise_power_watts = k_boltzmann * T * B
        self.noise_power_dbm = 10 * np.log10(noise_power_watts * 1e3)

    def _path_loss_db(self, distance_m: float) -> float:
        """Free-space + path loss exponent model (dB)."""
        d0 = 1.0
        pls = (10 * self.path_loss_exponent
               * np.log10(max(distance_m, d0))
               + 20 * np.log10(4 * np.pi * self.carrier_freq_ghz * 1e9 / 3e8))
        return pls

    def _rayleigh_fading_db(self) -> float:
        """Rayleigh fading gain in dB (complex envelope magnitude squared)."""
        h_real = self.rng.normal(0, 1 / np.sqrt(2))
        h_imag = self.rng.normal(0, 1 / np.sqrt(2))
        gain = h_real**2 + h_imag**2
        return 10 * np.log10(max(gain, 1e-10))

    def _shadowing_db(self) -> float:
        return self.rng.normal(0, self.shadow_std_db)

    def _sinr_to_cqi(self, sinr_db: float) -> int:
        """Map SINR to CQI using 3GPP-like thresholds."""
        thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1,
                      10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
        for cqi, thr in enumerate(thresholds, start=1):
            if sinr_db < thr:
                return max(1, cqi - 1)
        return 15

    def compute_channel(self, distance_m: float, tx_power_dbm: float = None) -> ChannelState:
        """Compute instantaneous channel state for a UE at given distance."""
        if tx_power_dbm is None:
            tx_power_dbm = self.bs_power_dbm

        pl_db = self._path_loss_db(distance_m)
        shadow_db = self._shadowing_db()
        fading_db = self._rayleigh_fading_db()
        fading_gain = 10 ** (fading_db / 10)

        rx_power_dbm = tx_power_dbm - pl_db - shadow_db + fading_db
        interference_dbm = rx_power_dbm - 10  # simplified co-channel

        sinr_linear = (10 ** (rx_power_dbm / 10)) / (
            10 ** (interference_dbm / 10) + 10 ** (self.noise_power_dbm / 10)
        )
        sinr_db = 10 * np.log10(max(sinr_linear, 1e-6))

        cqi = self._sinr_to_cqi(sinr_db)
        spectral_eff = CQI_TABLE[cqi]

        return ChannelState(
            sinr_db=sinr_db,
            cqi=cqi,
            spectral_eff=spectral_eff,
            path_loss_db=pl_db,
            fading_gain=fading_gain,
        )

    def batch_update(self, distances: np.ndarray, tx_powers_dbm: np.ndarray) -> List[ChannelState]:
        """Update channels for all UEs in a cell."""
        return [self.compute_channel(d, p) for d, p in zip(distances, tx_powers_dbm)]
