from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MamDispersionResult:
    """Processed dispersion result."""

    method: str
    frequency: np.ndarray
    phase_velocity: np.ndarray
    uncertainty: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate arrays are 1D and matching length."""
        if self.frequency.ndim != 1 or self.phase_velocity.ndim != 1:
            raise ValueError("frequency and phase_velocity must be 1D arrays")
        if self.frequency.shape[0] != self.phase_velocity.shape[0]:
            raise ValueError("frequency and phase_velocity lengths must match")
        if self.uncertainty is not None:
            if self.uncertainty.ndim != 1:
                raise ValueError("uncertainty must be 1D when provided")
            if self.uncertainty.shape[0] != self.frequency.shape[0]:
                raise ValueError("uncertainty length must match frequency")

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "method": self.method,
            "frequency": self.frequency.tolist(),
            "phase_velocity": self.phase_velocity.tolist(),
            "uncertainty": None if self.uncertainty is None else self.uncertainty.tolist(),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MamDispersionResult":
        """Create an instance from JSON-safe dictionary."""
        return cls(
            method=data["method"],
            frequency=np.asarray(data["frequency"], dtype=float),
            phase_velocity=np.asarray(data["phase_velocity"], dtype=float),
            uncertainty=None
            if data.get("uncertainty") is None
            else np.asarray(data["uncertainty"], dtype=float),
            meta=dict(data.get("meta", {})),
        )
