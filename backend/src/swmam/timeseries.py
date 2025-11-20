from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class MamTimeSeries:
    """MAM time series data."""

    data: np.ndarray  # shape (n_comp, n_stations, n_samples)
    dt: float
    start_time: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return self.data.shape[0]

    @property
    def n_stations(self) -> int:
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[2]

    def validate(self) -> None:
        """Validate data array dimensions and sampling interval."""
        if self.data.ndim != 3:
            raise ValueError("data must be a 3D array shaped (n_comp, n_stations, n_samples)")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
