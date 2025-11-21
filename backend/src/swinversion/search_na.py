from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .ensemble import ModelEnsemble
from .forward_base import ForwardSolver
from .misfit import dispersion_misfit
from .model import LayerParamBounds, ModelParametrization, VsLayer, VsModel
from .search_base import GlobalSearchStrategy, register_global_search


def _sample_layer(rng: np.random.Generator, bounds: LayerParamBounds) -> VsLayer:
    vs = rng.uniform(bounds.vs_min, bounds.vs_max)
    h = rng.uniform(bounds.h_min, bounds.h_max)
    vp_vs_ratio = rng.uniform(bounds.vp_vs_min, bounds.vp_vs_max)
    vp = vp_vs_ratio * vs
    rho = rng.uniform(bounds.rho_min, bounds.rho_max)
    return VsLayer(vs=vs, h=h, vp=vp, rho=rho)


def _perturb_layer(rng: np.random.Generator, layer: VsLayer, bounds: LayerParamBounds) -> VsLayer:
    span_vs = bounds.vs_max - bounds.vs_min
    span_h = bounds.h_max - bounds.h_min
    span_rho = bounds.rho_max - bounds.rho_min
    span_ratio = bounds.vp_vs_max - bounds.vp_vs_min

    vs = np.clip(layer.vs + rng.normal(scale=0.1 * span_vs), bounds.vs_min, bounds.vs_max)
    h = np.clip(layer.h + rng.normal(scale=0.1 * span_h), bounds.h_min, bounds.h_max)
    ratio = (layer.vp / layer.vs) if layer.vs != 0 else bounds.vp_vs_min
    ratio = np.clip(ratio + rng.normal(scale=0.1 * span_ratio), bounds.vp_vs_min, bounds.vp_vs_max)
    vp = ratio * vs
    rho = np.clip(layer.rho + rng.normal(scale=0.1 * span_rho), bounds.rho_min, bounds.rho_max)
    return VsLayer(vs=vs, h=h, vp=vp, rho=rho)


def _sample_model(rng: np.random.Generator, param: ModelParametrization) -> VsModel:
    layers = [_sample_layer(rng, b) for b in param.layer_bounds]
    return VsModel(layers=layers)


def _perturb_model(rng: np.random.Generator, model: VsModel, param: ModelParametrization) -> VsModel:
    layers = [
        _perturb_layer(rng, layer, bounds) for layer, bounds in zip(model.layers, param.layer_bounds)
    ]
    return VsModel(layers=layers)


@dataclass
class NAConfig:
    n_initial: int
    n_iterations: int
    n_resample: int
    n_keep: int
    seed: int | None = None


@dataclass
class NAStrategy(GlobalSearchStrategy):
    name: ClassVar[str] = "na"

    def run(
        self,
        param: ModelParametrization,
        obs_freq: np.ndarray,
        obs_c: np.ndarray,
        obs_sigma: np.ndarray,
        solver: ForwardSolver,
        config: NAConfig,
    ) -> ModelEnsemble:
        rng = np.random.default_rng(config.seed)

        models: list[VsModel] = []
        misfits: list[float] = []

        # Initial sampling
        for _ in range(config.n_initial):
            m = _sample_model(rng, param)
            try:
                phi = dispersion_misfit(solver, m, obs_freq, obs_c, obs_sigma)
            except Exception:
                phi = np.inf
            models.append(m)
            misfits.append(phi)

        for _ in range(config.n_iterations):
            misfit_array = np.asarray(misfits)
            if misfit_array.size == 0:
                break
            keep = min(config.n_keep, misfit_array.size)
            best_idx = np.argsort(misfit_array)[:keep]
            best_models = [models[i] for i in best_idx]

            for base_model in best_models:
                for _ in range(config.n_resample):
                    new_model = _perturb_model(rng, base_model, param)
                    try:
                        phi = dispersion_misfit(solver, new_model, obs_freq, obs_c, obs_sigma)
                    except Exception:
                        phi = np.inf
                    models.append(new_model)
                    misfits.append(phi)

        ensemble = ModelEnsemble(models=models, misfits=np.asarray(misfits, dtype=float))
        return ensemble


register_global_search(NAStrategy())
