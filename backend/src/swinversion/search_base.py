from typing import Any, ClassVar, Protocol

import numpy as np

from .ensemble import ModelEnsemble
from .forward_base import ForwardSolver
from .model import ModelParametrization


class GlobalSearchStrategy(Protocol):
    name: ClassVar[str]

    def run(
        self,
        param: ModelParametrization,
        obs_freq: np.ndarray,
        obs_c: np.ndarray,
        obs_sigma: np.ndarray,
        solver: ForwardSolver,
        config: Any,
    ) -> ModelEnsemble:
        ...


GLOBAL_SEARCH_REGISTRY: dict[str, GlobalSearchStrategy] = {}


def register_global_search(strategy: GlobalSearchStrategy) -> None:
    GLOBAL_SEARCH_REGISTRY[strategy.name.lower()] = strategy


def get_global_search(name: str) -> GlobalSearchStrategy:
    return GLOBAL_SEARCH_REGISTRY[name.lower()]
