import numpy as np

from swinversion.misfit import dispersion_misfit
from swinversion.model import VsLayer, VsModel


class DummySolver:
    name = "dummy"

    def __init__(self, value: float):
        self.value = value

    def predict(self, model: VsModel, freqs: np.ndarray, mode: int = 0) -> np.ndarray:
        return np.full_like(freqs, fill_value=self.value, dtype=float)


def test_dispersion_misfit_constant_solver():
    freqs = np.array([1.0, 2.0, 3.0])
    obs_c = np.array([300.0, 320.0, 280.0])
    sigma = np.array([10.0, 10.0, 10.0])

    solver = DummySolver(value=300.0)
    model = VsModel(layers=[VsLayer(vs=300.0, h=5.0, vp=500.0, rho=2000.0)])

    phi = dispersion_misfit(solver, model, freqs, obs_c, sigma)

    expected = np.mean(((obs_c - 300.0) / sigma) ** 2)
    assert np.isclose(phi, expected)
