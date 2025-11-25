from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .geometry import MamArrayGeometry


def _match_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        col = lookup.get(cand.lower())
        if col is not None:
            return col
    raise ValueError(f"Missing required column; tried {', '.join(candidates)}")


def load_geometry_from_csv(csv_path: Path) -> MamArrayGeometry:
    """
    Load MAM array geometry from a CSV file.

    Expected columns (case-insensitive):
      - station name: e.g. 'Name' or 'Station' (prefers 'Name')
      - 'Easting' (metres)
      - 'Northing' (metres)
      - 'Elevation' (metres)
    """
    df = pd.read_csv(csv_path)

    name_col = _match_column(df.columns, ["Name", "Station"])
    e_col = _match_column(df.columns, ["Easting"])
    n_col = _match_column(df.columns, ["Northing"])
    z_col = _match_column(df.columns, ["Elevation"])

    station_ids = df[name_col].astype(str).tolist()
    easting = df[e_col].to_numpy(dtype=float)
    northing = df[n_col].to_numpy(dtype=float)
    elevations = df[z_col].to_numpy(dtype=float)

    coords_xy = np.column_stack([easting, northing])

    geom = MamArrayGeometry(station_ids=station_ids, coords_xy=coords_xy, elevations=elevations)
    geom.validate()
    return geom
