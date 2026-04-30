"""Ballistic trajectory r(t) = r0 + v * (t - t_start)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BallisticTrajectory:
    r0: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    t_start: float = 0.0

    def __post_init__(self) -> None:
        self.r0 = np.asarray(self.r0, dtype=np.float64).reshape(3)
        self.v = np.asarray(self.v, dtype=np.float64).reshape(3)

    def __call__(self, t: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.ndim == 0:
            return self.r0 + self.v * (float(t_arr) - self.t_start)
        return self.r0[None, :] + self.v[None, :] * (t_arr[:, None] - self.t_start)

    def t_at_z(self, z: float) -> float:
        """Time at which the trajectory reaches axial coordinate z (requires v[2] != 0)."""
        if self.v[2] == 0.0:
            raise ValueError("trajectory has zero v_z; cannot solve for t at given z")
        return self.t_start + (z - self.r0[2]) / self.v[2]
