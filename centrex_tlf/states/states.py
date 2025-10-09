from __future__ import annotations

import abc
import hashlib
import json
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np
import numpy.typing as npt
import sympy as sp

from .utils import CGc

__all__ = [
    "ElectronicState",
    "BasisState",
    "CoupledBasisState",
    "UncoupledBasisState",
    "State",
    "BasisStates_from_State",
    "Basis",
    "CoupledState",
    "UncoupledState",
]


class ElectronicState(Enum):
    X = auto()
    B = auto()


class Basis(Enum):
    Uncoupled = auto()
    Coupled = auto()
    CoupledP = auto()
    CoupledΩ = auto()


class Parity(Enum):
    Pos = +1
    Neg = -1


@dataclass
class BasisState(abc.ABC):
    J: int
    electronic_state: Optional[ElectronicState]
    isCoupled: bool
    isUncoupled: bool
    basis: Optional[Basis]

    # scalar product (psi * a)
    def __mul__(self, a: Union[float, complex, int]):
        raise NotImplementedError

    # scalar product (a * psi)
    def __rmul__(self, a: Union[float, complex, int]):
        raise NotImplementedError

    def __matmul__(self, other) -> int | float | complex:
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def print_quantum_numbers(self, printing: bool = False) -> str:
        raise NotImplementedError

    def transform_to_omega_basis(self):
        raise NotImplementedError

    def transform_to_parity_basis(self):
        raise NotImplementedError

    def state_string_custom(self, quantum_numbers: list[str]) -> str:
        raise NotImplementedError


class CoupledBasisState(BasisState):
    # constructor
    def __init__(
        self,
        F: int,
        mF: int,
        F1: float,
        J: int,
        I1: float,
        I2: float,
        Omega: int = 0,
        P: Optional[int] = None,
        electronic_state: Optional[ElectronicState] = None,
        energy: Optional[float] = None,
        Ω: Optional[int] = None,
        v: Optional[int] = None,
        basis: Optional[Basis] = None,
    ):
        self.F, self.mF = F, mF
        self.F1 = F1
        self.J = J
        self.I1 = I1
        self.I2 = I2

        if Ω is not None:
            self.Ω: int = Ω
            self.Omega = self.Ω
        elif Omega is not None:
            self.Omega = Omega
            self.Ω = self.Omega
        else:
            raise AssertionError("need to supply either Omega or Ω")
        if P is not None:
            self.P: Optional[int] = P
        else:
            self.P = None
        #     raise AssertionError("need to supply parity P")
        if electronic_state is not None and not isinstance(
            electronic_state, ElectronicState
        ):
            raise TypeError(
                f"Supply electronic state as ElectronicState enum, not {type(electronic_state)}"
            )
        self.electronic_state = electronic_state
        self.energy = energy
        self.isCoupled = True
        self.isUncoupled = False

        # determine which basis we are in
        if basis is not None:
            self.basis = basis
        elif (self.P is None) and (self.electronic_state == ElectronicState.B):
            self.basis = Basis.CoupledΩ
        elif self.electronic_state == ElectronicState.B:
            self.basis = Basis.CoupledP
        elif self.electronic_state == ElectronicState.X:
            self.basis = Basis.Coupled
        else:
            self.basis = None
        self.v = v

    # equality testing
    def __eq__(self, other: object) -> bool:
        # return self.__hash__() == other.__hash__()
        if not isinstance(other, CoupledBasisState):
            return False
        else:
            return (
                self.F == other.F
                and self.mF == other.mF
                and self.I1 == other.I1
                and self.I2 == other.I2
                and self.F1 == other.F1
                and self.J == other.J
                and self.Omega == other.Omega
                and self.P == other.P
                and self.electronic_state == other.electronic_state
                and self.v == other.v
                and self.basis == other.basis
            )

    @overload
    def __matmul__(self, other: CoupledBasisState) -> Literal[0, 1]: ...

    @overload
    def __matmul__(self, other: UncoupledBasisState) -> int | float | complex: ...

    # inner product
    def __matmul__(self, other):
        if not isinstance(other, BasisState):
            raise TypeError(
                "can only matmul CoupledBasisState with CoupledBasisState or "
                f"UncoupledBasisState (not {type(other)})"
            )
        if other.isCoupled:
            if self == other:
                return 1
            else:
                return 0
        else:
            # other is UncoupledBasisState, cast for type checker
            other_uncoupled = cast(UncoupledBasisState, other)
            return (
                UncoupledState([(1, other_uncoupled)]) @ self.transform_to_uncoupled()
            )

    # superposition: addition
    def __add__(self, other: Self) -> CoupledState:
        if self == other:
            return CoupledState([(2, self)])
        elif isinstance(other, CoupledBasisState):
            if self.basis == other.basis:
                return CoupledState([(1, self), (1, other)])
            else:
                raise TypeError("can only add BasisStates with the same basis")
        else:
            raise TypeError(
                f"can only add CoupledBasisState (not {type(other)}) "
                "to CoupledBasisState"
            )

    # superposition: subtraction
    def __sub__(self, other: Self) -> CoupledState | Literal[0]:
        if self == other:
            return 0
        elif isinstance(other, CoupledBasisState):
            if self.basis == other.basis:
                return CoupledState([(1, self), (-1, other)])
            else:
                raise TypeError("can only add BasisStates withe same basis")
        else:
            raise TypeError(
                f"can only subtract CoupledBasisState (not {type(other)}) "
                "from CoupledBasisState"
            )

    # scalar product (psi * a)
    def __mul__(self, a: Union[complex, float, int]) -> CoupledState:
        return CoupledState([(a, self)])

    # scalar product (a * psi)
    def __rmul__(self, a: Union[float, complex, int]) -> CoupledState:
        return self * a

    def __hash__(self) -> int:
        P = self.P if self.P is not None else 2
        ev = self.electronic_state.value if self.electronic_state is not None else 0
        v = self.v if self.v is not None else -1
        quantum_numbers = (
            int(self.J),
            float(self.F1),
            int(self.F),
            int(self.mF),
            float(self.I1),
            float(self.I2),
            int(P),
            int(self.Omega),
            ev,
            v,
        )
        return hash(quantum_numbers)

    def __repr__(self) -> str:
        return self.state_string()

    def _format_quantum_numbers_helper(
        self,
    ) -> tuple[
        sp.Rational,
        sp.Rational,
        sp.Rational,
        sp.Rational,
        sp.Rational,
        sp.Rational,
        Optional[str],
        Optional[int],
        Optional[int],
    ]:
        F = sp.S(str(self.F), rational=True)
        mF = sp.S(str(self.mF), rational=True)
        F1 = sp.S(str(self.F1), rational=True)
        J = sp.S(str(self.J), rational=True)
        I1 = sp.S(str(self.I1), rational=True)
        I2 = sp.S(str(self.I2), rational=True)
        P: Optional[str] = None
        if self.P is not None:
            if self.P == 1:
                P = "+"
            elif self.P == -1:
                P = "-"
        Omega = self.Omega
        v = self.v
        return F, mF, F1, J, I1, I2, P, Omega, v

    def state_string(self) -> str:
        F, mF, F1, J, I1, I2, P, Omega, v = self._format_quantum_numbers_helper()

        string = f"J = {J}, F₁ = {F1}, F = {F}, mF = {mF}, I₁ = {I1}, I₂ = {I2}"

        if self.electronic_state is not None:
            string = f"{self.electronic_state.name}, {string}"
        if self.P is not None:
            string = f"{string}, P = {P}"
        if Omega is not None:
            string = f"{string}, Ω = {Omega}"
        if v is not None:
            string = f"{string}, v = {v}"
        return "|" + string + ">"

    def state_string_custom(self, quantum_numbers: List[str]) -> str:
        F, mF, F1, J, I1, I2, P, Omega, v = self._format_quantum_numbers_helper()

        string = ""
        for name in quantum_numbers:
            if "electronic" in name and self.electronic_state:
                string += f"{self.electronic_state.name}, "
            else:
                val = getattr(self, name)
                if val is not None:
                    if name == "Ω":
                        string += f"{name} = {Omega}, "
                    else:
                        string += f"{name} = {eval(name)}, "
        string = string.strip(", ")
        return "|" + string + ">"

    def print_quantum_numbers(self, printing: bool = False) -> str:
        if printing:
            print(self.state_string())
        return self.state_string()

    # A method to transform from coupled to uncoupled basis
    def transform_to_uncoupled(self) -> UncoupledState:
        F = self.F
        mF = self.mF
        F1 = self.F1
        J = self.J
        I1 = self.I1
        I2 = self.I2
        electronic_state = self.electronic_state
        P = self.P
        Omega = self.Omega

        mF1s = np.arange(-F1, F1 + 1, 1)
        mJs = range(-round(J, 0), round(J, 0) + 1, 1)
        m1s = np.arange(-I1, I1 + 1, 1)
        m2s = np.arange(-I2, I2 + 1, 1)

        uncoupled_state = UncoupledState()

        for mF1 in mF1s:
            for mJ in mJs:
                for m1 in m1s:
                    for m2 in m2s:
                        amp = CGc(J, mJ, I1, m1, F1, mF1) * CGc(F1, mF1, I2, m2, F, mF)
                        basis_state = UncoupledBasisState(
                            J,
                            mJ,
                            I1,
                            float(m1),
                            I2,
                            float(m2),
                            P=P,
                            Omega=Omega,
                            electronic_state=electronic_state,
                        )
                        uncoupled_state = uncoupled_state + UncoupledState(
                            [(amp, basis_state)]
                        )

        return uncoupled_state.normalize()

    # Method for transforming parity eigenstate to Omega eigenstate basis
    def transform_to_omega_basis(self) -> CoupledState:
        F = self.F
        mF = self.mF
        F1 = self.F1
        J = self.J
        I1 = self.I1
        I2 = self.I2
        electronic_state = self.electronic_state
        P = self.P
        Omega = self.Omega

        assert self.basis is not None, "Unknown basis state, can't transform to Ω basis"
        assert (
            P is not None
        ), "Can't transform state to Omega basis if parity is not known"

        # Check that not already in omega basis
        if self.basis == Basis.CoupledP and self.electronic_state == ElectronicState.B:
            state_minus = 1 * CoupledBasisState(
                F,
                mF,
                F1,
                J,
                I1,
                I2,
                Omega=-1 * Omega,
                P=None,
                electronic_state=electronic_state,
                basis=Basis.CoupledΩ,
                v=self.v,
            )
            state_plus = 1 * CoupledBasisState(
                F,
                mF,
                F1,
                J,
                I1,
                I2,
                Omega=1 * Omega,
                P=None,
                electronic_state=electronic_state,
                basis=Basis.CoupledΩ,
                v=self.v,
            )
            state: CoupledState = (
                1 / np.sqrt(2) * (state_plus + P * (-1) ** (J) * state_minus)
            )
            return state

        elif self.basis == Basis.Coupled and self.electronic_state == ElectronicState.X:
            raise ValueError("Cannot transform X state to Omega basis")
        else:
            raise ValueError("Cannot transform to Omega basis")

    def transform_to_parity_basis(self) -> CoupledState:
        """
        Transforms self from Omega eigenstate basis (i.e. signed Omega) to
        parity eigenstate basis (unsigned Omega, P is good quantum number).
        Doing this is only defined for electronic state B.
        """
        F = self.F
        mF = self.mF
        F1 = self.F1
        J = self.J
        I1 = self.I1
        I2 = self.I2
        electronic_state = self.electronic_state
        P = self.P
        Omega = self.Omega
        S = 0

        # Check that not already in parity basis

        if (self.basis == Basis.CoupledΩ) or (
            P is None and not electronic_state == ElectronicState.X
        ):
            if np.sign(Omega) == 1:
                state = (
                    1
                    / np.sqrt(2)
                    * (
                        1
                        * CoupledBasisState(
                            F,
                            mF,
                            F1,
                            J,
                            I1,
                            I2,
                            Omega=np.abs(Omega),
                            P=+1,
                            electronic_state=electronic_state,
                        )
                        + 1
                        * CoupledBasisState(
                            F,
                            mF,
                            F1,
                            J,
                            I1,
                            I2,
                            Omega=np.abs(Omega),
                            P=-1,
                            electronic_state=electronic_state,
                        )
                    )
                )

            elif np.sign(Omega) == -1:
                state = (
                    1
                    / np.sqrt(2)
                    * (-1) ** (J - S)
                    * (
                        1
                        * CoupledBasisState(
                            F,
                            mF,
                            F1,
                            J,
                            I1,
                            I2,
                            Omega=np.abs(Omega),
                            P=+1,
                            electronic_state=electronic_state,
                        )
                        - 1
                        * CoupledBasisState(
                            F,
                            mF,
                            F1,
                            J,
                            I1,
                            I2,
                            Omega=np.abs(Omega),
                            P=-1,
                            electronic_state=electronic_state,
                        )
                    )
                )
        else:
            state = 1 * self

        return state  # type: ignore[return-value]


# Class for uncoupled basis states
class UncoupledBasisState(BasisState):
    # constructor
    def __init__(
        self,
        J: int,
        mJ: int,
        I1: float,
        m1: float,
        I2: float,
        m2: float,
        Omega: Optional[int] = None,
        P: Optional[int] = None,
        electronic_state: Optional[ElectronicState] = None,
        energy: Optional[float] = None,
        basis: Optional[Basis] = None,
        v: Optional[int] = None,
    ):
        self.J, self.mJ = J, mJ
        self.I1, self.m1 = I1, m1
        self.I2, self.m2 = I2, m2
        if Omega is not None:
            self.Omega = Omega
        else:
            raise ValueError("need to supply Omega")
        if P is not None:
            self.P = P
        else:
            raise ValueError("need to supply parity P")
        if electronic_state is not None:
            self.electronic_state = electronic_state
        else:
            raise ValueError("need to supply electronic state")
        self.isCoupled = False
        self.isUncoupled = True

        if basis is not None:
            self.basis = basis
        else:
            self.basis = Basis.Uncoupled

        self.v = v

        self.energy = energy

    # equality testing
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UncoupledBasisState):
            return False
        else:
            return (
                self.J == other.J
                and self.mJ == other.mJ
                and self.I1 == other.I1
                and self.I2 == other.I2
                and self.m1 == other.m1
                and self.m2 == other.m2
                and self.Omega == other.Omega
                and self.P == other.P
                and self.electronic_state == other.electronic_state
                and self.basis == other.basis
            )

    @overload
    def __matmul__(self, other: UncoupledBasisState) -> Literal[1, 0]: ...

    @overload
    def __matmul__(self, other: CoupledBasisState) -> int | float | complex: ...

    # inner product
    def __matmul__(self, other):
        if other.isUncoupled:
            if self == other:
                return 1
            else:
                return 0
        elif isinstance(other, CoupledBasisState):
            return UncoupledState([(1, self)]) @ other.transform_to_uncoupled()
        else:
            raise TypeError(
                "can only multiply UncoupledBasisState with UncoupledBasisState or "
                f"CoupledBasisState (not {type(other)}"
            )

    # superposition: addition
    def __add__(self, other: UncoupledBasisState) -> UncoupledState:
        if self == other:
            return UncoupledState([(2, self)])
        elif isinstance(other, UncoupledBasisState):
            return UncoupledState([(1, self), (1, other)])
        else:
            raise TypeError(
                f"can only add UncoupledBasisState (not {type(other)}) "
                "to UncoupledBasisState"
            )

    # superposition: subtraction
    def __sub__(self, other: UncoupledBasisState) -> Union[int, UncoupledState]:
        if self == other:
            return 0
        elif isinstance(other, UncoupledBasisState):
            return UncoupledState([(1, self), (-1, other)])
        else:
            raise TypeError(
                f"can only subtract UncoupledBasisState (not {type(other)}) "
                "from UncoupledBasisState"
            )

    # scalar product (psi * a)
    def __mul__(self, a: Union[float, complex, int]) -> UncoupledState:
        return UncoupledState([(a, self)])

    # scalar product (a * psi)
    def __rmul__(self, a: Union[float, complex, int]) -> UncoupledState:
        return self * a

    def __hash__(self) -> int:
        quantum_numbers = (
            int(self.J),
            int(self.mJ),
            float(self.I1),
            float(self.m1),
            float(self.I2),
            float(self.m2),
            int(self.P),
            int(self.Omega),
            self.electronic_state.value,
            self.v if self.v is not None else -1,
        )
        return hash(quantum_numbers)

    def __repr__(self) -> str:
        return self.state_string()

    def _format_quantum_numbers_helper(
        self, name: str
    ) -> tuple[
        str,
        Optional[sp.Rational | int | str],
    ]:
        if name == "J":
            return ("J", sp.S(str(self.J), rational=True))
        elif name == "mJ":
            return ("mJ", sp.S(str(self.mJ), rational=True))
        elif name == "I1":
            return ("I₁", sp.S(str(self.I1), rational=True))
        elif name == "I2":
            return ("I₂", sp.S(str(self.I2), rational=True))
        elif name == "m1":
            return ("m₁", sp.S(str(self.m1), rational=True))
        elif name == "m2":
            return ("m₂", sp.S(str(self.m2), rational=True))
        elif name == "P":
            P: str = ""
            if self.P == 1:
                P = "+"
            elif self.P == -1:
                P = "-"
            return ("P", P)
        elif name == "Omega" or name == "Ω":
            return ("Ω", self.Omega)
        else:
            return (name, None)

    def state_string(self) -> str:
        quantum_numbers = ["J", "mJ", "I1", "m1", "I2", "m2", "P", "Omega"]
        return self.state_string_custom(quantum_numbers)

    def state_string_custom(self, quantum_numbers: list[str]) -> str:
        string = ""
        for name in quantum_numbers:
            if "electronic" in name and self.electronic_state:
                string += f"{self.electronic_state.name}, "
            else:
                label, value = self._format_quantum_numbers_helper(name)
                if value is not None:
                    string += f"{label} = {value}, "
        string = string.strip(", ")
        return "|" + string + ">"

    def print_quantum_numbers(self, printing: bool = False) -> str:
        if printing:
            print(self.state_string())
        return self.state_string()

    # Method for converting to coupled basis
    def transform_to_coupled(self) -> CoupledState:
        # Determine quantum numbers
        J = self.J
        mJ = self.mJ
        I1 = self.I1
        m1 = self.m1
        I2 = self.I1
        m2 = self.m2
        Omega = self.Omega
        electronic_state = self.electronic_state
        P = self.P
        Omega = 0 if self.Omega is None else self.Omega
        v = self.v if self.v is not None else None

        # Determine what mF has to be
        mF = int(mJ + m1 + m2)

        uncoupled_state = self

        data: list[tuple[complex, CoupledBasisState]] = []

        # Loop over possible values of F1, F and m_F
        for F1 in np.arange(J - I1, J + I1 + 1):
            for F in range(int(round(F1 - I2, 0)), int(round(F1 + I2, 0)) + 1):
                if np.abs(mF) <= F:
                    coupled_state = CoupledBasisState(
                        F,
                        mF,
                        float(F1),
                        J,
                        I1,
                        I2,
                        Omega=Omega,
                        P=P,
                        electronic_state=electronic_state,
                        v=self.v,
                    )
                    amp = uncoupled_state @ coupled_state
                    data.append((amp, coupled_state))

        return CoupledState(data)

    # Method for transforming parity eigenstate to Omega eigenstate basis
    def transform_to_omega_basis(self) -> UncoupledState:
        # Determine quantum numbers
        J = self.J
        mJ = self.mJ
        I1 = self.I1
        m1 = self.m1
        I2 = self.I1
        m2 = self.m2
        Omega = self.Omega
        electronic_state = self.electronic_state
        P = self.P
        Omega = 0 if self.Omega is None else self.Omega

        # Check that not already in omega basis
        if P is not None and not electronic_state == ElectronicState.X:
            state_minus = 1 * UncoupledBasisState(
                J,
                mJ,
                I1,
                m1,
                I2,
                m2,
                P=P,
                Omega=-1 * Omega,
                electronic_state=electronic_state,
            )
            state_plus = 1 * UncoupledBasisState(
                J,
                mJ,
                I1,
                m1,
                I2,
                m2,
                P=P,
                Omega=Omega,
                electronic_state=electronic_state,
            )

            state = 1 / np.sqrt(2) * (state_plus + P * (-1) ** (J - 1) * state_minus)
        else:
            state = 1 * self

        return state


S = TypeVar("S", bound=BasisState)


class State(Generic[S]):
    def __init__(
        self,
        data: Sequence[
            Tuple[
                Union[int, float, complex],
                S,
            ]
        ],
        remove_zero_amp_cpts: bool = True,
    ):
        # remove components with zero amplitudes
        if remove_zero_amp_cpts:
            self.data = [(amp, cpt) for amp, cpt in data if amp != 0]
        else:
            self.data = list(data)

        # for iteration over the State
        self.index = len(self.data)

    def _create_new_instance(
        self,
        data: Sequence[
            Tuple[
                Union[int, float, complex],
                S,
            ]
        ],
        remove_zero_amp_cpts: bool = True,
    ) -> Self:
        return self.__class__(data, remove_zero_amp_cpts)

    # superposition: addition
    # (highly inefficient and ugly but should work)
    def __add__(self, other: Self) -> Self:
        data: List[Tuple[Union[float, complex], S]] = []
        # add components that are in self but not in other
        for amp1, cpt1 in self.data:
            only_in_self = True
            for amp2, cpt2 in other.data:
                if cpt2 == cpt1:
                    only_in_self = False
                    break
            if only_in_self:
                data.append((amp1, cpt1))
        # add components that are in other but not in self
        for amp1, cpt1 in other.data:
            only_in_other = True
            for amp2, cpt2 in self.data:
                if cpt2 == cpt1:
                    only_in_other = False
                    break
            if only_in_other:
                data.append((amp1, cpt1))
        # add components that are both in self and in other
        for amp1, cpt1 in self.data:
            for amp2, cpt2 in other.data:
                if cpt2 == cpt1:
                    data.append((amp1 + amp2, cpt1))
                    break
        return self._create_new_instance(data)

    # superposition: subtraction
    def __sub__(self, other: Self) -> Self:
        return self + -1 * other

    # scalar product (psi * a)
    def __mul__(self, a: Union[float, complex, int]) -> Self:
        return self._create_new_instance([(a * amp, psi) for amp, psi in self.data])

    # scalar product (a * psi)
    def __rmul__(self, a: Union[float, complex, int]) -> Self:
        return self * a

    # scalar division (psi / a)
    def __truediv__(self, a: Union[float, complex, int]) -> Self:
        return self * (1 / a)

    # negation
    def __neg__(self) -> Self:
        return -1 * self

    # inner product
    def __matmul__(self, other: Self) -> Union[int, float, complex]:
        result = 0
        for amp1, psi1 in self:
            for amp2, psi2 in other:
                result += amp1.conjugate() * amp2 * (psi1 @ psi2)  # type: ignore[operator]
        return result

    # iterator methods
    def __iter__(self):
        return ((amp, state) for amp, state in self.data)

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

    def __eq__(self, other: object) -> bool:
        """
        __eq__ method
        iterates through all amplitudes and states, can be slow

        Args:
            other (object): object to compare to

        Returns:
            bool: True if equal, False if not
        """
        if not isinstance(other, self.__class__):
            return False
        else:
            S1 = self.order_by_amp()
            S2 = other.order_by_amp()
            for (a1, s1), (a2, s2) in zip(S1, S2):
                if a1 != a2:
                    return False
                elif s1 != s2:
                    return False
            return True

    # direct access to a component
    def __getitem__(self, i: int) -> Tuple[Union[complex, float, int], BasisState]:
        return self.data[i]

    def __repr__(self) -> str:
        ordered = self.order_by_amp()
        idx = 0
        string = ""
        amp_max = np.max(np.abs(list(zip(*ordered))[0]))
        for amp, state in ordered:
            if np.abs(amp) < amp_max * 1e-3:
                continue
            string += f"{amp:.2f} x {state}"
            idx += 1
            if (idx > 5) or (idx == len(ordered.data)):
                break
            string += "\n"
        if idx == 0:
            return ""
        else:
            return string

    def state_string_custom(self, quantum_numbers: list[str]) -> str:
        ordered = self.order_by_amp()
        idx = 0
        string = ""
        amp_max = np.max(np.abs(list(zip(*ordered))[0]))
        for amp, state in ordered:
            if np.abs(amp) < amp_max * 1e-3:
                continue
            string += f"{amp:.2f} x {state.state_string_custom(quantum_numbers)}"
            idx += 1
            if (idx > 5) or (idx == len(ordered.data)):
                break
            string += "\n"
        if idx == 0:
            return ""
        else:
            return string

    def find_largest_component(self) -> S:
        # Order the state by amplitude
        state = self.order_by_amp()

        return state.data[0][1]

    @property
    def largest(self) -> S:
        return self.find_largest_component()

    def normalize(self) -> Self:
        data = []
        N = np.sqrt(self @ self)
        for amp, basis_state in self.data:
            data.append((amp / N, basis_state))

        return self._create_new_instance(data)

    # Method that removes components that are smaller than tolerance from the state
    def remove_small_components(self, tol: float = 1e-3) -> Self:
        purged_data = []
        for amp, basis_state in self.data:
            if np.abs(amp) > tol:
                purged_data.append((amp, basis_state))

        return self._create_new_instance(purged_data)

    # Method for ordering states in descending order of amp^2
    def order_by_amp(self) -> Self:
        data = self.data
        amp_array = np.zeros(len(data))

        # Make an numpy array of the amplitudes
        for i, d in enumerate(data):
            amp_array[i] = np.abs((data[i][0])) ** 2

        # Find ordering of array in descending order
        index = np.argsort(-1 * amp_array)

        # Reorder data
        reordered_data = data
        reordered_data = [reordered_data[i] for i in index]

        return self._create_new_instance(reordered_data)

    # Method for printing largest component basis states
    def print_largest_components(self, n: int = 1) -> str:
        # Order the state by amplitude
        state = self.order_by_amp()

        # Initialize an empty string
        string = ""

        for i in range(0, n):
            basis_state = state.data[i][1]
            basis_state.print_quantum_numbers()

        return string

    def state_vector(
        self,
        QN: Sequence[S],
    ) -> npt.NDArray[np.complex128]:
        state_vector = [1 * state @ self for state in QN]
        return np.array(state_vector, dtype=complex)

    # Method that generates a density matrix from state
    def density_matrix(
        self,
        QN: Sequence[S],
    ) -> npt.NDArray[np.complex128]:
        # Get state vector
        state_vec = self.state_vector(QN)

        # Generate density matrix from state vector
        density_matrix = np.tensordot(state_vec.conj(), state_vec, axes=0)

        return density_matrix.astype(np.complex128)

    # Method for transforming all basis states to omega basis
    def transform_to_omega_basis(self) -> Self:
        state = self._create_new_instance(data=[])
        for amp, basis_state in self.data:
            state += amp * basis_state.transform_to_omega_basis()

        return state

    # Method for transforming all basis states to parity basis
    def transform_to_parity_basis(self) -> Self:
        state = self._create_new_instance(data=[])
        for amp, basis_state in self.data:
            state += amp * basis_state.transform_to_parity_basis()

        return state


class CoupledState(State):
    def __init__(
        self,
        data: Sequence[Tuple[Union[int, float, complex], CoupledBasisState]] = [],
        remove_zero_amp_cpts: bool = True,
    ):
        super().__init__(data=data, remove_zero_amp_cpts=remove_zero_amp_cpts)

    def _create_new_instance(
        self,
        data: Sequence[Tuple[Union[int, float, complex], CoupledBasisState]],
        remove_zero_amp_cpts: bool = True,
    ):
        return CoupledState(data, remove_zero_amp_cpts)

    def find_largest_component(self) -> CoupledBasisState:
        return super().find_largest_component()

    @property
    def largest(self) -> CoupledBasisState:
        return super().largest

    # Function that returns state vector in given basis
    def state_vector(
        self,
        QN: Union[
            Sequence[CoupledBasisState],
            Sequence[CoupledState],
        ],
    ) -> npt.NDArray[np.complex128]:
        return super().state_vector(QN)

    # Method that generates a density matrix from state
    def density_matrix(
        self, QN: Union[Sequence[CoupledBasisState], Sequence[CoupledState]]
    ) -> npt.NDArray[np.complex128]:
        return super().density_matrix(QN)

    # Method for converting the state into the uncoupled basis
    def transform_to_uncoupled(self) -> UncoupledState:
        # Loop over the basis states, check if they are already in uncoupled
        # basis and if not convert to uncoupled basis, output state in new basis
        state_in_uncoupled_basis = UncoupledState()

        for amp, basis_state in self.data:
            if basis_state.isUncoupled:
                state_in_uncoupled_basis += UncoupledState([(amp, basis_state)])
            if basis_state.isCoupled:
                state_in_uncoupled_basis += amp * basis_state.transform_to_uncoupled()

        return state_in_uncoupled_basis


class UncoupledState(State):
    def __init__(
        self,
        data: Sequence[Tuple[Union[int, float, complex], UncoupledBasisState]] = [],
        remove_zero_amp_cpts: bool = True,
    ):
        super().__init__(data=data, remove_zero_amp_cpts=remove_zero_amp_cpts)

    def _create_new_instance(
        self,
        data: Sequence[Tuple[int | float | complex, UncoupledBasisState]],
        remove_zero_amp_cpts: bool = True,
    ) -> UncoupledState:
        return UncoupledState(data, remove_zero_amp_cpts)

    def find_largest_component(self) -> UncoupledBasisState:
        return super().find_largest_component()

    @property
    def largest(self) -> UncoupledBasisState:
        return super().largest

    # Function that returns state vector in given basis
    def state_vector(
        self,
        QN: Union[
            Sequence[UncoupledBasisState],
            Sequence[UncoupledState],
        ],
    ) -> npt.NDArray[np.complex128]:
        return super().state_vector(QN)

    # Method that generates a density matrix from state
    def density_matrix(
        self, QN: Union[Sequence[UncoupledBasisState], Sequence[UncoupledState]]
    ) -> npt.NDArray[np.complex128]:
        return super().density_matrix(QN)

    # Method for converting the state into the coupled basis
    def transform_to_coupled(self) -> CoupledState:
        # Loop over the basis states, check if they are already in uncoupled
        # basis and if not convert to coupled basis, output state in new basis
        state_in_coupled_basis = CoupledState()

        for amp, basis_state in self.data:
            if basis_state.isCoupled:
                state_in_coupled_basis += CoupledState([(amp, basis_state)])
            if basis_state.isUncoupled:
                state_in_coupled_basis += amp * basis_state.transform_to_coupled()

        return state_in_coupled_basis


def BasisStates_from_State(
    states: Union[Sequence[State], npt.NDArray[Any]],
) -> npt.NDArray[Any]:
    unique = []
    for state in states:
        for amp, basisstate in state:
            if basisstate not in unique:
                unique.append(basisstate)
    return np.array(unique)


@overload
def basisstate_to_state_list(
    states: Sequence[CoupledBasisState],
) -> List[CoupledState]: ...


@overload
def basisstate_to_state_list(
    states: Sequence[UncoupledBasisState],
) -> List[UncoupledState]: ...


def basisstate_to_state_list(states):
    return [1 * bs for bs in states]
