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
    O = -2
    P = -1
    Q = 0
    R = +1
    S = +2


@dataclass
class MicrowaveTransition:
    J_ground: int
    J_excited: int
    electronic_ground: states.ElectronicState = states.ElectronicState.X
    electronic_excited: states.ElectronicState = states.ElectronicState.X

    def __repr__(self) -> str:
        return f"MicrowaveTransition({self.name})"

    @property
    def name(self) -> str:
        return f"J={self.J_ground} -> J={self.J_excited}"

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
            J=self.J_ground, electronic=self.electronic_ground, Ω=self.Ω_ground
        )

    @property
    def qn_select_excited(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_excited, electronic=self.electronic_excited, Ω=self.Ω_excited
        )


@dataclass
class OpticalTransition:
    t: OpticalTransitionType
    J_ground: int
    F1: float
    F: int
    electronic_ground: states.ElectronicState = states.ElectronicState.X
    electronic_excited: states.ElectronicState = states.ElectronicState.B

    def __repr__(self) -> str:
        return f"OpticalTransition({self.name})"

    @property
    def name(self) -> str:
        F1 = sympy.S(str(self.F1), rational=True)
        return f"{self.t.name}({self.J_ground}) F1'={F1} F'={int(self.F)}"

    @property
    def J_excited(self) -> int:
        return self.J_ground + self.t.value

    @property
    def P_excited(self) -> int:
        return self.P_ground * -1

    @property
    def P_ground(self) -> int:
        return (-1) ** self.J_ground

    @property
    def Ω_excited(self) -> int:
        return 1

    @property
    def Ω_ground(self) -> int:
        return 0

    @property
    def qn_select_ground(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_ground, electronic=self.electronic_ground, Ω=self.Ω_ground
        )

    @property
    def qn_select_excited(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_excited,
            F1=self.F1,
            F=self.F,
            electronic=self.electronic_excited,
            P=self.P_excited,
            Ω=self.Ω_excited,
        )

    @property
    def ground_states(self) -> Sequence[states.CoupledBasisState]:
        return states.generate_coupled_states_X(self.qn_select_ground)

    @property
    def excited_states(self) -> Sequence[states.CoupledBasisState]:
        return states.generate_coupled_states_B(self.qn_select_excited)
