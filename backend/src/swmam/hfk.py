from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .base import DispersionMethod, register_dispersion_method
from .dispersion import MamDispersionResult
from .geometry import MamArrayGeometry
from .timeseries import MamTimeSeries


@dataclass
class HFKConfig:
    fmin: float
    fmax: float
    df: float
    vmin: float
    vmax: float
    dv: float
    azimuths: np.ndarray | None = None
    window_length: float | None = None
    window_overlap: float = 0.5
    diagonal_loading: float = 0.01


@dataclass
class HFKMethod(DispersionMethod):
    name: ClassVar[str] = "hfk"
    default_config: ClassVar[HFKConfig] = HFKConfig(
        fmin=0.1,
        fmax=10.0,
        df=0.1,
        vmin=50.0,
        vmax=1000.0,
        dv=10.0,
        azimuths=None,
        window_length=None,
        window_overlap=0.5,
        diagonal_loading=0.01,
    )

    def estimate(
        self,
        ts: MamTimeSeries,
        geom: MamArrayGeometry,
        config: HFKConfig,
    ) -> MamDispersionResult:
        data_z = ts.data[0]  # shape (n_stations, n_samples)
        n_stations, n_samples = data_z.shape

        # Windowing setup
        if config.window_length is None:
            windows = [(0, n_samples)]
            window_fn = None
        else:
            nper = int(round(config.window_length / ts.dt))
            if nper <= 0 or nper > n_samples:
                raise ValueError("window_length leads to invalid window size")
            step = max(1, int(round(nper * (1.0 - config.window_overlap))))
            windows = [(start, start + nper) for start in range(0, n_samples - nper + 1, step)]
            window_fn = np.hanning(nper)

        if not windows:
            raise ValueError("No windows generated for HFK estimation")

        # Frequency axis from window length
        nper_used = windows[0][1] - windows[0][0]
        freq_full = np.fft.rfftfreq(nper_used, d=ts.dt)
        freq_mask = (freq_full >= config.fmin) & (freq_full <= config.fmax)
        freq = freq_full[freq_mask]
        if freq.size == 0:
            raise ValueError("No frequency bins within requested band")

        # Select nearest bins to the requested df grid
        target_freq = np.arange(config.fmin, config.fmax + config.df * 0.5, config.df)
        target_idx = np.array([int(np.argmin(np.abs(freq_full - f))) for f in target_freq])
        target_idx = np.unique(target_idx)
        target_idx = target_idx[(freq_full[target_idx] >= config.fmin) & (freq_full[target_idx] <= config.fmax)]
        freq = freq_full[target_idx]

        # Accumulate spatial covariance
        S_accum = np.zeros((freq.size, n_stations, n_stations), dtype=complex)
        for start, end in windows:
            segment = data_z[:, start:end]
            if window_fn is not None:
                segment = segment * window_fn
            segment = segment - segment.mean(axis=1, keepdims=True)
            spec = np.fft.rfft(segment, axis=1)
            spec_sel = spec[:, target_idx]  # shape (n_stations, n_freq)
            # Outer products per frequency
            for k in range(freq.size):
                s = spec_sel[:, k:k+1]  # (n_stations,1)
                S_accum[k] += s @ s.conj().T

        S_accum /= len(windows)

        # Steering vector setup (azimuth = 0 deg)
        coords = geom.coords_xy
        x_proj = coords[:, 0]

        v_grid = np.arange(config.vmin, config.vmax + config.dv * 0.5, config.dv)
        phase_velocity = np.full(freq.shape, np.nan, dtype=float)
        uncertainty = np.full(freq.shape, np.nan, dtype=float)

        eye_n = np.eye(n_stations, dtype=complex)
        two_pi = 2.0 * np.pi

        for i_f, f in enumerate(freq):
            if f <= 0:
                continue
            S = S_accum[i_f]
            power_trace = np.trace(S).real / n_stations
            S_reg = S + config.diagonal_loading * power_trace * eye_n
            try:
                S_inv = np.linalg.inv(S_reg)
            except np.linalg.LinAlgError:
                continue

            delays = x_proj[:, None] / v_grid[None, :]
            a = np.exp(-1j * two_pi * f * delays)  # shape (n_stations, n_vel)

            denom = np.einsum("vi,ij,jv->v", a.conj().T, S_inv, a, optimize=True)
            denom_real = np.real(denom)
            powers = np.full(v_grid.shape, -np.inf, dtype=float)
            valid = np.isfinite(denom_real) & (denom_real != 0)
            powers[valid] = 1.0 / denom_real[valid]

            if not np.any(np.isfinite(powers)):
                continue

            best_idx = int(np.argmax(powers))
            phase_velocity[i_f] = v_grid[best_idx]
            # Use inverse of max power as a crude uncertainty proxy
            if np.isfinite(powers[best_idx]) and powers[best_idx] != 0:
                uncertainty[i_f] = float(1.0 / powers[best_idx])

        result = MamDispersionResult(
            method="HFK",
            frequency=freq,
            phase_velocity=phase_velocity,
            uncertainty=uncertainty,
            meta={
                "v_grid": v_grid,
                "n_windows": len(windows),
                "window_length": config.window_length,
                "window_overlap": config.window_overlap,
                "diagonal_loading": config.diagonal_loading,
            },
        )
        result.validate()
        return result


register_dispersion_method(HFKMethod())
