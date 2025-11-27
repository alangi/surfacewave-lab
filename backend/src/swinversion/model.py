from dataclasses import dataclass

import numpy as np


@dataclass
class LayerParamBounds:
    """Bounds for inversion layer parameters."""

    vs_min: float
    vs_max: float
    h_max: float
    h_min: float = 0.5
    vp_vs_min: float = 1.6
    vp_vs_max: float = 2.0
    rho_min: float = 1800.0
    rho_max: float = 2300.0


@dataclass
class ModelParametrization:
    """Parametrization settings for inversion runs."""

    nlayers: int
    layer_bounds: list[LayerParamBounds]

    def validate(self) -> None:
        if len(self.layer_bounds) != self.nlayers:
            raise ValueError("layer_bounds length must match nlayers")


@dataclass
class VsLayer:
    """Single shear-wave velocity layer."""

    vs: float
    h: float
    vp: float
    rho: float


@dataclass
class VsModel:
    """Shear-wave velocity model definition."""

    layers: list[VsLayer]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return layer properties as arrays (h, vp, vs, rho)."""
        h = np.asarray([layer.h for layer in self.layers], dtype=float)
        vp = np.asarray([layer.vp for layer in self.layers], dtype=float)
        vs = np.asarray([layer.vs for layer in self.layers], dtype=float)
        rho = np.asarray([layer.rho for layer in self.layers], dtype=float)
        return h, vp, vs, rho

    @classmethod
    def from_arrays(
        cls, h: np.ndarray, vp: np.ndarray, vs: np.ndarray, rho: np.ndarray
    ) -> "VsModel":
        h_arr = np.asarray(h, dtype=float)
        vp_arr = np.asarray(vp, dtype=float)
        vs_arr = np.asarray(vs, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)

        if not (
            h_arr.shape == vp_arr.shape == vs_arr.shape == rho_arr.shape
        ):
            raise ValueError("h, vp, vs, and rho must have matching shapes")

        layers = [
            VsLayer(vs=vs_value, h=h_value, vp=vp_value, rho=rho_value)
            for h_value, vp_value, vs_value, rho_value in zip(
                h_arr, vp_arr, vs_arr, rho_arr
            )
        ]
        return cls(layers=layers)
