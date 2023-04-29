from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import sympy as smp

from centrex_tlf import states
from centrex_tlf.transitions import (
    MicrowaveTransition,
    OpticalTransition,
    OpticalTransitionType,
)

from .polarization import Polarization
from .utils import check_transition_coupled_allowed, select_main_states

__all__ = [
    "TransitionSelector",
    "generate_transition_selectors",
    "get_possible_optical_transitions",
]


@dataclass
class TransitionSelector:
    ground: Sequence[states.State]
    excited: Sequence[states.State]
    polarizations: Sequence[npt.NDArray[np.complex_]]
    polarization_symbols: List[smp.Symbol]
    Ω: smp.Symbol
    δ: smp.Symbol
    description: Optional[str] = None
    type: Optional[str] = None
    ground_main: Optional[states.State] = None
    excited_main: Optional[states.State] = None
    phase_modulation: bool = False

    def __repr__(self) -> str:
        if self.description is None:
            J_g = np.unique([g.largest.J for g in self.ground])[0]
            J_e = np.unique([e.largest.J for e in self.excited])[0]
            return f"TransitionSelector(J={J_g} -> J={J_e})"
        else:
            return f"TransitionSelector({self.description})"


def generate_transition_selectors(
    transitions: Sequence[Union[OpticalTransition, MicrowaveTransition]],
    polarizations: Sequence[Sequence[Polarization]],
    ground_mains: Optional[Sequence[states.State]] = None,
    excited_mains: Optional[Sequence[states.State]] = None,
    phase_modulations: Optional[Sequence[bool]] = None,
) -> List[TransitionSelector]:
    """
    Generate a list of TransitionSelectors from Transition(s) and Polarization(s).

    Args:
        transitions (Sequence[Union[OpticalTransition, MicrowaveTransition]]):
                                                    transitions to include in the system
        polarizations (Sequence[Sequence[Polarization]]): polarization, list of
                                                        polarizations per transition.
        ground_mains (Optional[Sequence[States]]): Sequence of a main ground state to
                                                    use per transition
        excited_mains (Optional[Sequence[States]]): Sequence of a a main excited state
                                                    to use per transition
        excited_mains:

    Returns:
        List[TransitionSelector]: List of TransitionSelectors
    """
    transition_selectors = []

    for idt, (transition, polarization) in enumerate(zip(transitions, polarizations)):
        ground_states_approx_qn_select = states.QuantumSelector(
            J=transition.J_ground,
            electronic=transition.electronic_ground,
            P=transition.P_ground,
            Ω=transition.Ω_ground,
        )
        ground_states_approx = list(
            [
                1 * s
                for s in states.generate_coupled_states_X(
                    ground_states_approx_qn_select
                )
            ]
        )

        if isinstance(transition, OpticalTransition):
            excited_states_approx = [
                1 * s
                for s in states.generate_coupled_states_B(transition.qn_select_excited)
            ]
        elif isinstance(transition, MicrowaveTransition):
            excited_states_approx = [
                1 * s
                for s in states.generate_coupled_states_X(transition.qn_select_excited)
            ]

        if phase_modulations is None:
            phase_modulation = False
        else:
            phase_modulation = phase_modulations[idt]
        if ground_mains is None:
            ground_main, excited_main = select_main_states(
                ground_states_approx, excited_states_approx, polarization[0].vector
            )

        transition_selectors.append(
            TransitionSelector(
                ground=ground_states_approx,
                excited=excited_states_approx,
                polarizations=[p.vector for p in polarization],
                polarization_symbols=[
                    smp.Symbol(f"P{p.name}{idt}") for p in polarization
                ],
                Ω=smp.Symbol(f"Ω{idt}", complex=True),
                δ=smp.Symbol(f"δ{idt}"),
                description=transition.name,
                ground_main=ground_main if ground_mains is None else ground_mains[idt],
                excited_main=excited_main
                if excited_mains is None
                else excited_mains[idt],
                phase_modulation=phase_modulation,
            )
        )
    return transition_selectors


def get_possible_optical_transitions(
    ground_state: states.CoupledBasisState,
    transition_types: Optional[Sequence[OpticalTransitionType]] = None,
):
    J = ground_state.J
    # F1 = ground_state.F1
    # F = ground_state.F
    I1 = ground_state.I1
    I2 = ground_state.I2

    if transition_types is None:
        transition_types = [t for t in OpticalTransitionType]

    transitions = []
    for transition_type in transition_types:
        ΔJ = transition_type.value
        J_excited = J + ΔJ
        _transitions = [
            OpticalTransition(transition_type, J, F1, F)
            for F1 in np.arange(np.abs(J_excited - I1), J_excited + I1 + 1)
            for F in np.arange(np.abs(F1 - I2), F1 + I2 + 1, dtype=int)
        ]
        _transitions = [
            t
            for t in _transitions
            if check_transition_coupled_allowed(ground_state, t.excited_states[0])
        ]
        transitions.append(_transitions)
    return transitions
