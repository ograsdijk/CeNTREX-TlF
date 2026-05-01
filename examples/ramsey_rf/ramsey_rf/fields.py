"""Analytic field profiles for RF Ramsey benchmark.

Library code carries no hardcoded experiment numbers. All physical parameters
(field magnitudes, lengths, frequencies, amplitudes, phases) are required
keyword args on the convenience constructors. Direction can be either a fixed
unit 3-vector or a callable R -> (3,) for spatially varying polarization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

DirectionLike = Union[
    Sequence[float],
    npt.NDArray[np.floating],
    Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
]


def _normalize_direction(direction: DirectionLike) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Wrap a fixed 3-vector or a callable into a uniform R -> (3,) callable.

    Fixed vectors are normalized to unit length (otherwise the field magnitude
    would silently double-count); callables are assumed to return a vector with
    the desired direction-and-magnitude convention already (callable returns
    a unit vector OR a vector that scales the scalar profile).
    """
    if callable(direction):
        return direction
    arr = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        raise ValueError("direction vector has zero norm")
    unit = arr / norm
    return lambda R: unit


@dataclass
class AnalyticDCField:
    """DC E-field as a function of position R = (x, y, z).

    Field at R is `profile_z(R[2]) * direction(R)`. Direction is normalized to a
    unit vector (so the magnitude lives entirely in `profile_z`).
    """

    direction: DirectionLike
    profile_z: Callable[[float], float]

    def __post_init__(self) -> None:
        self._direction_fn = _normalize_direction(self.direction)

    def __call__(self, R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = float(R[2])
        magnitude = float(self.profile_z(z))
        return magnitude * self._direction_fn(R)

    @classmethod
    def tanh_ramp(
        cls,
        *,
        E_low: float,
        E_high: float,
        z_center: float,
        ramp_length: float,
        direction: DirectionLike,
    ) -> "AnalyticDCField":
        """Single tanh ramp: low at z << z_center, high at z >> z_center."""

        def profile(z: float) -> float:
            return E_low + 0.5 * (E_high - E_low) * (1.0 + np.tanh((z - z_center) / ramp_length))

        return cls(direction=direction, profile_z=profile)

    @classmethod
    def symmetric_plateau(
        cls,
        *,
        E_low: float,
        E_high: float,
        half_width: float,
        ramp_length: float,
        z_origin: float = 0.0,
        direction: DirectionLike,
    ) -> "AnalyticDCField":
        """Symmetric low/high/low plateau.

        Low outside |z - z_origin| > half_width, high inside. Smooth tanh
        transitions of width `ramp_length`. Ramp midpoints sit at
        z = z_origin +/- half_width.
        """

        def profile(z: float) -> float:
            zr = z - z_origin
            up = 0.5 * (1.0 + np.tanh((zr + half_width) / ramp_length))
            down = 0.5 * (1.0 - np.tanh((zr - half_width) / ramp_length))
            shape = up * down  # in [0, 1], ~1 inside, ~0 outside
            return E_low + (E_high - E_low) * shape

        return cls(direction=direction, profile_z=profile)


@dataclass
class AnalyticRFRegion:
    """One RF interaction region: spatial envelope x cos(omega t + phi) x direction.

    The library treats `omega` and `phi` as the carrier of the underlying RF E-field
    (no RWA, no rotating frame). The total E-field at R, t is:
        E_rf(R, t) = envelope_z(R[2]) * cos(omega t + phi) * direction(R)
    Multiple regions are summed by `FieldStack`.
    """

    direction: DirectionLike
    envelope_z: Callable[[float], float]
    omega: float
    phi: float

    def __post_init__(self) -> None:
        self._direction_fn = _normalize_direction(self.direction)

    def envelope_vec(self, R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = float(R[2])
        amp = float(self.envelope_z(z))
        return amp * self._direction_fn(R)

    @classmethod
    def gaussian(
        cls,
        *,
        z_center: float,
        width: float,
        amplitude: float,
        omega: float,
        phi: float,
        direction: DirectionLike,
    ) -> "AnalyticRFRegion":
        """Gaussian envelope with 1/e^2 half-width = `width`."""

        def envelope(z: float) -> float:
            return amplitude * np.exp(-2.0 * ((z - z_center) / width) ** 2)

        return cls(direction=direction, envelope_z=envelope, omega=omega, phi=phi)

    @classmethod
    def sin2(
        cls,
        *,
        z_center: float,
        width: float,
        amplitude: float,
        omega: float,
        phi: float,
        direction: DirectionLike,
    ) -> "AnalyticRFRegion":
        """sin^2 envelope of full width `width`, zero outside."""

        def envelope(z: float) -> float:
            zr = z - z_center
            if abs(zr) > 0.5 * width:
                return 0.0
            return amplitude * np.cos(np.pi * zr / width) ** 2

        return cls(direction=direction, envelope_z=envelope, omega=omega, phi=phi)

    @classmethod
    def rounded_rectangle(
        cls,
        *,
        z_center: float,
        width: float,
        edge_length: float,
        amplitude: float,
        omega: float,
        phi: float,
        direction: DirectionLike,
    ) -> "AnalyticRFRegion":
        """Flat-top of width `width` with smooth tanh edges of scale `edge_length`."""

        half = 0.5 * width

        def envelope(z: float) -> float:
            left = np.tanh((z - (z_center - half)) / edge_length)
            right = np.tanh((z - (z_center + half)) / edge_length)
            return amplitude * 0.5 * (left - right)

        return cls(direction=direction, envelope_z=envelope, omega=omega, phi=phi)


@dataclass
class MagneticRFRegion:
    """One MAGNETIC RF interaction region: spatial envelope x cos(omega t + phi) x direction.

    The total B-field at R, t is:
        B_rf(R, t) = envelope_z(R[2]) * cos(omega t + phi) * direction(R)
    Units: amplitude is in Gauss (matching centrex_tlf's H_func B convention).
    Multiple regions are summed by `FieldStack.B_total`.

    For the typical CeNTREX RF Ramsey geometry, RF is magnetic in x and drives
    the Tl nuclear spin via g_Tl·μ_N·**B**·**I_Tl**. Use this class instead of
    `AnalyticRFRegion` (which is for electric RF coupling via the molecular dipole).
    """

    direction: DirectionLike
    envelope_z: Callable[[float], float]
    omega: float
    phi: float

    def __post_init__(self) -> None:
        self._direction_fn = _normalize_direction(self.direction)

    def envelope_vec(self, R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = float(R[2])
        amp = float(self.envelope_z(z))
        return amp * self._direction_fn(R)

    @classmethod
    def gaussian(
        cls,
        *,
        z_center: float,
        width: float,
        amplitude: float,
        omega: float,
        phi: float,
        direction: DirectionLike,
    ) -> "MagneticRFRegion":
        """Gaussian envelope with 1/e^2 half-width = `width`. amplitude in Gauss."""

        def envelope(z: float) -> float:
            return amplitude * np.exp(-2.0 * ((z - z_center) / width) ** 2)

        return cls(direction=direction, envelope_z=envelope, omega=omega, phi=phi)

    @classmethod
    def sin2(
        cls,
        *,
        z_center: float,
        width: float,
        amplitude: float,
        omega: float,
        phi: float,
        direction: DirectionLike,
    ) -> "MagneticRFRegion":
        """sin^2 envelope of full width `width`, zero outside. amplitude in Gauss."""

        def envelope(z: float) -> float:
            zr = z - z_center
            if abs(zr) > 0.5 * width:
                return 0.0
            return amplitude * np.cos(np.pi * zr / width) ** 2

        return cls(direction=direction, envelope_z=envelope, omega=omega, phi=phi)

    @classmethod
    def rounded_rectangle(
        cls,
        *,
        z_center: float,
        width: float,
        edge_length: float,
        amplitude: float,
        omega: float,
        phi: float,
        direction: DirectionLike,
    ) -> "MagneticRFRegion":
        """Flat-top of width `width` with smooth tanh edges. amplitude in Gauss."""

        half = 0.5 * width

        def envelope(z: float) -> float:
            left = np.tanh((z - (z_center - half)) / edge_length)
            right = np.tanh((z - (z_center + half)) / edge_length)
            return amplitude * 0.5 * (left - right)

        return cls(direction=direction, envelope_z=envelope, omega=omega, phi=phi)


@dataclass
class FieldStack:
    """Container for the DC E-field, optional DC B-field, and RF regions of either type.

    `rf_regions` is for electric RF (units V/cm), `rf_regions_B` is for magnetic
    RF (units Gauss). Both add to the corresponding total at each (R, t).
    """

    E_dc: AnalyticDCField
    rf_regions: list[AnalyticRFRegion] = field(default_factory=list)
    B_dc: Optional[AnalyticDCField] = None
    rf_regions_B: list[MagneticRFRegion] = field(default_factory=list)

    def E_total(self, R: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
        E = self.E_dc(R)
        for region in self.rf_regions:
            E = E + np.cos(region.omega * t + region.phi) * region.envelope_vec(R)
        return E

    def B_total(self, R: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
        B = self.B_dc(R) if self.B_dc is not None else np.zeros(3, dtype=np.float64)
        for region in self.rf_regions_B:
            B = B + np.cos(region.omega * t + region.phi) * region.envelope_vec(R)
        return B
