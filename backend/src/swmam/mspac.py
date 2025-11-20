from dataclasses import dataclass
from typing import ClassVar

from .base import DispersionMethod, register_dispersion_method
from .dispersion import MamDispersionResult
from .geometry import MamArrayGeometry
from .timeseries import MamTimeSeries


@dataclass
class MSPACConfig:
    fmin: float
    fmax: float
    df: float
    vmin: float
    vmax: float
    dv: float
    n_radii: int
    window_length: float | None = None
    window_overlap: float = 0.5


@dataclass
class MSPACMethod(DispersionMethod):
    name: ClassVar[str] = "mspac"
    default_config: MSPACConfig = MSPACConfig(
        fmin=0.1,
        fmax=10.0,
        df=0.1,
        vmin=50.0,
        vmax=1000.0,
        dv=10.0,
        n_radii=8,
        window_length=None,
        window_overlap=0.5,
    )

    def estimate(
        self,
        ts: MamTimeSeries,
        geom: MamArrayGeometry,
        config: MSPACConfig,
    ) -> MamDispersionResult:
        raise NotImplementedError("TODO: implement MSPAC")


register_dispersion_method(MSPACMethod())
