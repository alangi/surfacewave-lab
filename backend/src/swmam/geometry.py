from dataclasses import dataclass

import numpy as np


@dataclass
class MamArrayGeometry:
    """Microtremor array geometry definition."""

    station_ids: list[str]
    coords_xy: np.ndarray  # shape (n_stations, 2)
    elevations: np.ndarray | None = None

    def validate(self) -> None:
        """Validate geometry shapes/lengths."""
        n_stations = len(self.station_ids)

        if self.coords_xy.ndim != 2 or self.coords_xy.shape[1] != 2:
            raise ValueError("coords_xy must have shape (n_stations, 2)")

        if self.coords_xy.shape[0] != n_stations:
            raise ValueError("station_ids and coords_xy must have matching lengths")

        if self.elevations is not None:
            if self.elevations.ndim != 1:
                raise ValueError("elevations must be 1D when provided")
            if self.elevations.shape[0] != n_stations:
                raise ValueError("elevations length must match station_ids")
