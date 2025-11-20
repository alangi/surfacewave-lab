"""Surface-wave MAM processing package."""

from .base import (
    DISPERSION_REGISTRY,
    DispersionMethod,
    get_dispersion_method,
    register_dispersion_method,
)
from . import mspac as _mspac  # noqa: F401  # ensure MSPAC registers itself

__all__ = [
    "DispersionMethod",
    "DISPERSION_REGISTRY",
    "register_dispersion_method",
    "get_dispersion_method",
]
