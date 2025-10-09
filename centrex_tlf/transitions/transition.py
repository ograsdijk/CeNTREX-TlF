from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import sympy

import centrex_tlf.states as states

__all__: list[str] = [
    "OpticalTransitionType",
    "MicrowaveTransition",
    "OpticalTransition",
]


class OpticalTransitionType(IntEnum):
    O = -2  # noqa: E741
    P = -1
    Q = 0
    R = +1
    S = +2


@dataclass
class MicrowaveTransition:
    """
    Pure‐rotational (microwave) J→J′ transition within the same electronic manifold.

    Attributes:
        J_ground: Rotational quantum number of the ground state (non-negative)
        J_excited: Rotational quantum number of the excited state (non-negative)
        electronic_ground: Electronic state of the ground level (default: X)
        electronic_excited: Electronic state of the excited level (default: X)
    """

    J_ground: int
    J_excited: int
    electronic_ground: states.ElectronicState = states.ElectronicState.X
    electronic_excited: states.ElectronicState = states.ElectronicState.X

    def __post_init__(self):
        # ensure J’s are valid
        if self.J_ground < 0 or self.J_excited < 0:
            raise ValueError("J_ground and J_excited must be non‑negative")

    def __repr__(self) -> str:
        return f"MicrowaveTransition({self.name})"

    @property
    def name(self) -> str:
        return f"J={self.J_ground}→J={self.J_excited}"

    @property
    def Ω_ground(self) -> int:
        return 0

    @property
    def Ω_excited(self) -> int:
        return 0

    @property
    def P_ground(self) -> int:
        return (-1) ** self.J_ground

    @property
    def P_excited(self) -> int:
        return (-1) ** self.J_excited

    @property
    def qn_select_ground(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_ground,
            electronic=self.electronic_ground,
            Ω=self.Ω_ground,
            P=self.P_ground,
        )

    @property
    def qn_select_excited(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_excited,
            electronic=self.electronic_excited,
            Ω=self.Ω_excited,
            P=self.P_excited,
        )


@dataclass
class OpticalTransition:
    """
    Electronic (optical) transition with fine/hyperfine labels.
    J_excited is J_ground + branch shift t.value.

    Attributes:
        t: Transition type (O, P, Q, R, or S branch)
        J_ground: Rotational quantum number of the ground state (non-negative)
        F1_excited: Intermediate angular momentum quantum number F1 of excited state
                   (can be half-integer, e.g., 1.5 for 3/2)
        F_excited: Total angular momentum quantum number F of excited state (non-negative)
        electronic_ground: Electronic state of the ground level (default: X)
        electronic_excited: Electronic state of the excited level (default: B)

    Note:
        The F1_excited parameter accepts floats representing half-integer values.
        For example, use 1.5 or 3/2 for F1=3/2.
    """

    t: OpticalTransitionType
    J_ground: int
    F1_excited: float
    F_excited: int
    electronic_ground: states.ElectronicState = states.ElectronicState.X
    electronic_excited: states.ElectronicState = states.ElectronicState.B

    def __post_init__(self):
        # validate inputs
        if self.J_ground < 0 or self.F1_excited < 0 or self.F_excited < 0:
            raise ValueError("J_ground, F1, and F must be non‑negative")

        # ensure computed J_excited is non‑negative
        J_exc = self.J_ground + int(self.t.value)
        if J_exc < 0:
            raise ValueError(
                f"J_excited (J_ground + {self.t.value}) = {J_exc} must be non‑negative"
            )

    def __repr__(self) -> str:
        return f"OpticalTransition({self.name})"

    @property
    def J_excited(self) -> int:
        return self.J_ground + self.t.value

    @property
    def name(self) -> str:
        # Convert float to Rational for exact representation (e.g., 1.5 → 3/2)
        F1rat = sympy.Rational(self.F1_excited).limit_denominator()
        return f"{self.t.name}({self.J_ground}) F1'={F1rat} F'={self.F_excited}"

    @property
    def Ω_ground(self) -> int:
        return 0

    @property
    def Ω_excited(self) -> int:
        return 1

    @property
    def P_ground(self) -> int:
        return (-1) ** self.J_ground

    @property
    def P_excited(self) -> int:
        # optical parity flips sign
        return -self.P_ground

    @property
    def qn_select_ground(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_ground,
            F1=None,  # no F1 on ground selector
            F=None,  # no F on ground selector
            electronic=self.electronic_ground,
            Ω=self.Ω_ground,
            P=self.P_ground,
        )

    @property
    def qn_select_excited(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_excited,
            F1=self.F1_excited,
            F=self.F_excited,
            electronic=self.electronic_excited,
            Ω=self.Ω_excited,
            P=self.P_excited,
        )

    @property
    def ground_states(self) -> Sequence[states.CoupledBasisState]:
        return states.generate_coupled_states_X(self.qn_select_ground)

    @property
    def excited_states(self) -> Sequence[states.CoupledBasisState]:
        return states.generate_coupled_states_B(self.qn_select_excited)
