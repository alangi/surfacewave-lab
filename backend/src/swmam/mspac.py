from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.special import j0

from .base import DispersionMethod, register_dispersion_method
from .dispersion import MamDispersionResult
from .geometry import MamArrayGeometry
from .timeseries import MamTimeSeries


@dataclass
class MSPACConfig:
    fmin: float
    fmax: float
    df: float
    vmin: float
    vmax: float
    dv: float
    n_radii: int
    window_length: float | None = None
    window_overlap: float = 0.5


@dataclass
class MSPACMethod(DispersionMethod):
    name: ClassVar[str] = "mspac"
    default_config: ClassVar[MSPACConfig] = MSPACConfig(
        fmin=0.1,
        fmax=10.0,
        df=0.1,
        vmin=50.0,
        vmax=1000.0,
        dv=10.0,
        n_radii=8,
        window_length=None,
        window_overlap=0.5,
    )

    def estimate(
        self,
        ts: MamTimeSeries,
        geom: MamArrayGeometry,
        config: MSPACConfig,
    ) -> MamDispersionResult:
        # Use vertical component (index 0).
        data_z = ts.data[0]
        n_stations, n_samples = data_z.shape

        # Build window indices.
        if config.window_length is None:
            windows = [(0, n_samples)]
        else:
            nper = int(round(config.window_length / ts.dt))
            if nper <= 0 or nper > n_samples:
                raise ValueError("window_length leads to invalid window size")
            step = max(1, int(round(nper * (1.0 - config.window_overlap))))
            windows = [(start, start + nper) for start in range(0, n_samples - nper + 1, step)]

        # Distances and bins.
        coords = geom.coords_xy
        dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        upper_mask = np.triu(np.ones_like(dist_mat, dtype=bool), k=1)
        dists = dist_mat[upper_mask]
        if dists.size == 0:
            raise ValueError("At least two stations are required for MSPAC")
        dmin, dmax = float(dists.min()), float(dists.max())
        bin_edges = np.linspace(dmin, dmax, config.n_radii + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Preallocate accumulators.
        freq = np.fft.rfftfreq(windows[0][1] - windows[0][0], d=ts.dt)
        n_freq = freq.size
        rho_sum = np.zeros((config.n_radii, n_freq))
        rho_count = np.zeros((config.n_radii, n_freq), dtype=int)

        for start, end in windows:
            segment = data_z[:, start:end]
            spec = np.fft.rfft(segment, axis=1)
            auto = np.abs(spec) ** 2

            for i in range(n_stations):
                for j in range(i + 1, n_stations):
                    dist = dist_mat[i, j]
                    bin_idx = np.searchsorted(bin_edges, dist, side="right") - 1
                    if bin_idx < 0 or bin_idx >= config.n_radii:
                        continue

                    denom = np.sqrt(auto[i] * auto[j])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        spac_pair = np.real(spec[i] * np.conj(spec[j]) / denom)
                    spac_pair = np.nan_to_num(spac_pair)
                    rho_sum[bin_idx] += spac_pair
                    rho_count[bin_idx] += 1

        # Average rho over pairs/windows.
        rho = np.zeros_like(rho_sum)
        valid = rho_count > 0
        rho[valid] = rho_sum[valid] / rho_count[valid]
        rho[~valid] = np.nan

        # Fit phase velocity per frequency.
        v = np.arange(config.vmin, config.vmax + config.dv * 0.5, config.dv)
        phase_velocity = np.full_like(freq, np.nan, dtype=float)
        uncertainty = np.full_like(freq, np.nan, dtype=float)
        two_pi = 2.0 * np.pi

        for idx_f, f in enumerate(freq):
            rho_obs = rho[:, idx_f]
            mask = np.isfinite(rho_obs)
            if not np.any(mask):
                continue
            r_used = bin_centers[mask]
            rho_used = rho_obs[mask]

            arg = (two_pi * f * r_used[None, :]) / v[:, None]
            model = j0(arg)
            residuals = model - rho_used[None, :]
            mse = np.mean(residuals**2, axis=1)
            best_idx = int(np.argmin(mse))
            phase_velocity[idx_f] = v[best_idx]
            uncertainty[idx_f] = float(np.sqrt(mse[best_idx]))

        result = MamDispersionResult(
            method="MSPAC",
            frequency=freq,
            phase_velocity=phase_velocity,
            uncertainty=uncertainty,
            meta={
                "radii": bin_centers,
                "n_windows": len(windows),
                "window_length": config.window_length,
                "window_overlap": config.window_overlap,
            },
        )
        result.validate()
        return result


register_dispersion_method(MSPACMethod())
