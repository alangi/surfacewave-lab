from typing import Any, ClassVar, Protocol

import numpy as np

from .model import VsModel


class ForwardSolver(Protocol):
    name: ClassVar[str]

    def predict(self, model: VsModel, freqs: np.ndarray, mode: int = 0) -> np.ndarray:
        ...


FORWARD_SOLVERS: dict[str, ForwardSolver] = {}


def register_forward_solver(solver: ForwardSolver) -> None:
    FORWARD_SOLVERS[solver.name.lower()] = solver


def get_forward_solver(name: str) -> ForwardSolver:
    return FORWARD_SOLVERS[name.lower()]
