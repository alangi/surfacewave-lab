"""Surface-wave MAM processing package."""

from .base import (
    DISPERSION_REGISTRY,
    DispersionMethod,
    get_dispersion_method,
    register_dispersion_method,
)

__all__ = [
    "DispersionMethod",
    "DISPERSION_REGISTRY",
    "register_dispersion_method",
    "get_dispersion_method",
]
