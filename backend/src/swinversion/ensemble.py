from dataclasses import dataclass

import numpy as np

from .model import VsModel


@dataclass
class ModelEnsemble:
    """Collection of inversion model realizations."""

    models: list[VsModel]
    misfits: np.ndarray  # shape (n_models,)

    def _validate(self) -> None:
        if self.misfits.ndim != 1:
            raise ValueError("misfits must be 1D")
        if len(self.models) != self.misfits.shape[0]:
            raise ValueError("models length must match misfits length")

    def best_index(self) -> int:
        """Return index of the best (minimum) misfit model."""
        self._validate()
        if self.misfits.size == 0:
            raise ValueError("No models available")
        return int(np.argmin(self.misfits))

    def best_model(self) -> VsModel:
        """Return the best-fit model."""
        return self.models[self.best_index()]

    def best_misfit(self) -> float:
        """Return the minimum misfit value."""
        return float(self.misfits[self.best_index()])
