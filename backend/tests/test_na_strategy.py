import numpy as np
import pytest

from swinversion.forward_disba import DisbaForwardSolver, _HAS_DISBA
from swinversion.misfit import dispersion_misfit
from swinversion.model import LayerParamBounds, ModelParametrization, VsLayer, VsModel
from swinversion.search_na import NAConfig, NAStrategy


@pytest.mark.skipif(not _HAS_DISBA, reason="disba not installed")
def test_na_strategy_improves_best_misfit():
    solver = DisbaForwardSolver()

    # True model
    true_model = VsModel(
        layers=[
            VsLayer(vs=300.0, h=5.0, vp=500.0, rho=2000.0),
            VsLayer(vs=500.0, h=10.0, vp=800.0, rho=2100.0),
        ]
    )

    freqs = np.linspace(0.5, 2.0, 5)
    obs_c = solver.predict(true_model, freqs)
    obs_sigma = np.full_like(obs_c, 5.0)

    bounds = [
        LayerParamBounds(vs_min=200.0, vs_max=400.0, h_min=2.0, h_max=8.0),
        LayerParamBounds(vs_min=400.0, vs_max=600.0, h_min=8.0, h_max=12.0),
    ]
    param = ModelParametrization(nlayers=2, layer_bounds=bounds)

    config = NAConfig(n_initial=5, n_iterations=2, n_resample=2, n_keep=2, seed=42)
    strategy = NAStrategy()
    ensemble = strategy.run(param, freqs, obs_c, obs_sigma, solver, config)

    # Initial best (first n_initial entries)
    initial_misfits = ensemble.misfits[: config.n_initial]
    initial_best = np.min(initial_misfits)
    final_best = ensemble.best_misfit()

    assert final_best <= initial_best
