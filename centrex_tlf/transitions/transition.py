"""Transition definitions for optical and microwave spectroscopy.

This module provides dataclasses for representing molecular transitions including
optical (X→B electronic) and microwave (rotational) transitions with proper quantum
number labeling and state selectors.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import sympy as smp

from .. import states

__all__: list[str] = [
    "OpticalTransitionType",
    "MicrowaveTransition",
    "OpticalTransition",
]


class OpticalTransitionType(IntEnum):
    """Optical transition branch types following spectroscopic notation.

    Attributes:
        O (int): O-branch (ΔJ = -2)
        P (int): P-branch (ΔJ = -1)
        Q (int): Q-branch (ΔJ = 0)
        R (int): R-branch (ΔJ = +1)
        S (int): S-branch (ΔJ = +2)
    """

    O = -2  # noqa: E741
    P = -1
    Q = 0
    R = +1
    S = +2


@dataclass(frozen=True)
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
        """Human-readable transition name.

        Returns:
            str: Formatted string like "J=1→J=2"
        """
        return f"J={self.J_ground}→J={self.J_excited}"

    @property
    def Ω_ground(self) -> int:
        """Projection of electronic angular momentum for ground state (Ω=0 for X).

        Returns:
            int: Always 0 for X state
        """
        return 0

    @property
    def Ω_excited(self) -> int:
        """Projection of electronic angular momentum for excited state (Ω=0 for X).

        Returns:
            int: Always 0 for X state
        """
        return 0

    @property
    def P_ground(self) -> int:
        """Parity of ground state, P = (-1)^J.

        Returns:
            int: +1 for even J, -1 for odd J
        """
        return (-1) ** self.J_ground

    @property
    def P_excited(self) -> int:
        """Parity of excited state, P = (-1)^J.

        Returns:
            int: +1 for even J, -1 for odd J
        """
        return (-1) ** self.J_excited

    @property
    def qn_select_ground(self) -> states.QuantumSelector:
        """Quantum number selector for ground state manifold.

        Returns:
            states.QuantumSelector: Selector with J, electronic, Ω, P specified
        """
        return states.QuantumSelector(
            J=self.J_ground,
            electronic=self.electronic_ground,
            Ω=self.Ω_ground,
            P=self.P_ground,
        )

    @property
    def qn_select_excited(self) -> states.QuantumSelector:
        """Quantum number selector for excited state manifold.

        Returns:
            states.QuantumSelector: Selector with J, electronic, Ω, P specified
        """
        return states.QuantumSelector(
            J=self.J_excited,
            electronic=self.electronic_excited,
            Ω=self.Ω_excited,
            P=self.P_excited,
        )


@dataclass(frozen=True)
class OpticalTransition:
    """Electronic (optical) X→B transition with fine/hyperfine structure.

    Represents optical transitions between X (Ω=0) and B (Ω=1) electronic states,
    including hyperfine structure labels. The excited state J is determined by the
    ground state J and the branch type: J_excited = J_ground + t.value.

    Attributes:
        t (OpticalTransitionType): Transition branch type (O, P, Q, R, or S) determining
            ΔJ = t.value
        J_ground (int): Rotational quantum number of ground state (non-negative)
        F1_excited (float): Intermediate angular momentum F1 of excited state. Can be
            half-integer (e.g., 1.5 for 3/2)
        F_excited (int): Total angular momentum F of excited state (non-negative)
        electronic_ground (states.ElectronicState): Ground electronic state.
            Defaults to X.
        electronic_excited (states.ElectronicState): Excited electronic state.
            Defaults to B.

    Raises:
        ValueError: If J_ground, F1_excited, or F_excited is negative.
        ValueError: If computed J_excited = J_ground + t.value is negative.

    Note:
        F1_excited accepts floats for half-integer values. Use 1.5 or 3/2 for F1=3/2.
        Parity changes sign for optical transitions: P_excited = -P_ground.

    Example:
        >>> transition = OpticalTransition(
        ...     t=OpticalTransitionType.R,
        ...     J_ground=0,
        ...     F1_excited=0.5,
        ...     F_excited=1
        ... )
        >>> print(transition.name)
        R(0) F1'=1/2 F'=1
    """

    t: OpticalTransitionType
    J_ground: int
    F1_excited: float
    F_excited: int
    electronic_ground: states.ElectronicState = states.ElectronicState.X
    electronic_excited: states.ElectronicState = states.ElectronicState.B

    def __post_init__(self) -> None:
        """Validate quantum numbers and transition parameters after initialization."""
        # Validate non-negativity
        if self.J_ground < 0 or self.F1_excited < 0 or self.F_excited < 0:
            raise ValueError("J_ground, F1_excited, and F_excited must be non-negative")

        # Ensure computed J_excited is non-negative
        J_exc = self.J_ground + int(self.t.value)
        # Excited state B has Ω = 1, so J_excited must be >= 1
        if J_exc < 1:
            raise ValueError(
                f"J_excited (J_ground + {self.t.value}) = {J_exc} must be non-negative"
            )

    def __repr__(self) -> str:
        return f"OpticalTransition({self.name})"

    @property
    def J_excited(self) -> int:
        """Excited state rotational quantum number.

        Computed from ground state and branch type: J_excited = J_ground + ΔJ.

        Returns:
            int: Rotational quantum number of excited state
        """
        return self.J_ground + self.t.value

    @property
    def name(self) -> str:
        """Human-readable transition name with spectroscopic notation.

        Format: "Branch(J_ground) F1'=F1_excited F'=F_excited"
        F1 values are displayed as fractions (e.g., 1/2 instead of 0.5).

        Returns:
            str: Formatted transition name like "R(0) F1'=1/2 F'=1"
        """
        # Convert float to Rational for exact representation (e.g., 1.5 → 3/2)
        F1rat = smp.Rational(self.F1_excited).limit_denominator()
        return f"{self.t.name}({self.J_ground}) F1'={F1rat} F'={self.F_excited}"

    @property
    def Ω_ground(self) -> int:
        """Projection of electronic angular momentum for ground state (Ω=0 for X).

        Returns:
            int: Always 0 for X state
        """
        return 0

    @property
    def Ω_excited(self) -> int:
        """Projection of electronic angular momentum for excited state (Ω=1 for B).

        Returns:
            int: Always 1 for B state
        """
        return 1

    @property
    def P_ground(self) -> int:
        """Parity of ground state, P = (-1)^J.

        Returns:
            int: +1 for even J, -1 for odd J
        """
        return (-1) ** self.J_ground

    @property
    def P_excited(self) -> int:
        """Parity of excited state for optical transition.

        Optical transitions flip parity: P_excited = -P_ground.

        Returns:
            int: Opposite sign to P_ground
        """
        return -self.P_ground

    @property
    def qn_select_ground(self) -> states.QuantumSelector:
        """Quantum number selector for ground state manifold.

        Ground state selector excludes F1 and F (no hyperfine structure in X state).

        Returns:
            states.QuantumSelector: Selector with J, electronic, Ω, P specified
        """
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
        """Quantum number selector for excited state manifold.

        Excited state selector includes F1 and F for hyperfine structure in B state.

        Returns:
            states.QuantumSelector: Selector with J, F1, F, electronic, Ω, P specified
        """
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
        """Generate all ground state basis states matching the quantum selector.

        Returns:
            Sequence[states.CoupledBasisState]: List of ground X state basis states
        """
        return states.generate_coupled_states_X(self.qn_select_ground)

    @property
    def excited_states(self) -> Sequence[states.CoupledBasisState]:
        """Generate all excited state basis states matching the quantum selector.

        Returns:
            Sequence[states.CoupledBasisState]: List of excited B state basis states
        """
        return states.generate_coupled_states_B(self.qn_select_excited)
