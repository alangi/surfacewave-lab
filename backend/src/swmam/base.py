from typing import Any, ClassVar

import numpy as np

try:  # Python 3.9+ provides Protocol in typing
    from typing import Protocol
except ImportError:  # Fallback for older environments
    from typing_extensions import Protocol

from .dispersion import MamDispersionResult
from .geometry import MamArrayGeometry
from .timeseries import MamTimeSeries


class DispersionMethod(Protocol):
    name: ClassVar[str]

    def estimate(
        self,
        ts: MamTimeSeries,
        geom: MamArrayGeometry,
        config: Any,
    ) -> MamDispersionResult:
        ...


DISPERSION_REGISTRY: dict[str, DispersionMethod] = {}


def register_dispersion_method(method: DispersionMethod) -> None:
    DISPERSION_REGISTRY[method.name.lower()] = method


def get_dispersion_method(name: str) -> DispersionMethod:
    return DISPERSION_REGISTRY[name.lower()]
