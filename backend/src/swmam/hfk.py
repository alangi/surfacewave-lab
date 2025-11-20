from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .base import DispersionMethod, register_dispersion_method
from .dispersion import MamDispersionResult
from .geometry import MamArrayGeometry
from .timeseries import MamTimeSeries


@dataclass
class HFKConfig:
    fmin: float
    fmax: float
    df: float
    vmin: float
    vmax: float
    dv: float
    azimuths: np.ndarray | None = None
    window_length: float | None = None
    window_overlap: float = 0.5
    diagonal_loading: float = 0.01


@dataclass
class HFKMethod(DispersionMethod):
    name: ClassVar[str] = "hfk"
    default_config: ClassVar[HFKConfig] = HFKConfig(
        fmin=0.1,
        fmax=10.0,
        df=0.1,
        vmin=50.0,
        vmax=1000.0,
        dv=10.0,
        azimuths=None,
        window_length=None,
        window_overlap=0.5,
        diagonal_loading=0.01,
    )

    def estimate(
        self,
        ts: MamTimeSeries,
        geom: MamArrayGeometry,
        config: HFKConfig,
    ) -> MamDispersionResult:
        raise NotImplementedError("TODO: implement HFK/Capon dispersion")


register_dispersion_method(HFKMethod())
