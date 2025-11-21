import numpy as np

from .forward_base import ForwardSolver
from .model import VsModel


def dispersion_misfit(
    solver: ForwardSolver,
    model: VsModel,
    obs_freq: np.ndarray,
    obs_c: np.ndarray,
    obs_sigma: np.ndarray,
) -> float:
    """
    Weighted L2 misfit:
        phi = (1/N) * sum_i [ ((c_obs - c_pred)/sigma_i)**2 ]
    Assumes obs_freq and obs_c aligned, obs_sigma > 0.
    """
    freqs = np.asarray(obs_freq, dtype=float)
    c_obs = np.asarray(obs_c, dtype=float)
    sigma = np.asarray(obs_sigma, dtype=float)

    c_pred = np.asarray(solver.predict(model, freqs))

    residual = (c_obs - c_pred) / sigma
    return float(np.mean(residual**2))
