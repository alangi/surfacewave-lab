from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .forward_base import ForwardSolver, register_forward_solver
from .model import VsModel

try:
    from disba import PhaseDispersion

    _HAS_DISBA = True
except ImportError:  # pragma: no cover - optional dependency
    PhaseDispersion = None
    _HAS_DISBA = False


def model_to_disba_arrays(model: VsModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert VsModel to disba-friendly arrays (thickness km, vp km/s, vs km/s, rho g/cc)."""
    h_m, vp_mps, vs_mps, rho = model.as_arrays()
    # disba expects km and km/s, rho in g/cc
    thickness_km = h_m / 1000.0
    vp_kms = vp_mps / 1000.0
    vs_kms = vs_mps / 1000.0
    rho_gcc = rho / 1000.0  # convert kg/m^3 to g/cc
    return thickness_km, vp_kms, vs_kms, rho_gcc


@dataclass
class DisbaForwardSolver(ForwardSolver):
    name: ClassVar[str] = "disba"
    algorithm: str = "dunkin"

    def predict(self, model: VsModel, freqs: np.ndarray, mode: int = 0) -> np.ndarray:
        if not _HAS_DISBA:
            raise RuntimeError("disba is not installed; please add it to your environment to use DisbaForwardSolver")

        thickness, vp, vs, rho = model_to_disba_arrays(model)
        # disba expects periods; avoid division by zero
        if np.any(freqs <= 0):
            raise ValueError("Frequencies must be positive to compute periods")
        periods = 1.0 / freqs

        solver = PhaseDispersion(thickness=thickness, vp=vp, vs=vs, rho=rho, algorithm=self.algorithm)
        result = solver(periods=periods, mode=mode, wave="rayleigh")
        # result.c is in km/s; convert back to m/s
        return np.asarray(result.c) * 1000.0


register_forward_solver(DisbaForwardSolver())
