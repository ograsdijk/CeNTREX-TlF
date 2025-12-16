"""Transition selection and organization for optical Bloch equations.

This module provides tools for defining and organizing optical and microwave transitions
including their polarizations, symbolically-defined Rabi frequencies and detunings,
and the quantum states involved.
"""

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


def _sanitize_polarization_name(name: str) -> str:
    """Sanitize polarization name for use as a symbolic variable.

    Only keeps letters (a-z, A-Z). If the result is empty or contains
    mathematical symbols/numbers, returns None to trigger fallback naming.

    Args:
        name: Polarization name (e.g., "X", "1/√2 X + 1/√2 Z")

    Returns:
        Sanitized name containing only letters, or empty string if not simple

    Examples:
        >>> _sanitize_polarization_name("X")
        'X'
        >>> _sanitize_polarization_name("σp")
        'σp'
        >>> _sanitize_polarization_name("1/√2 X + 1/√2 Z")
        ''
    """
    # Check if name is simple (only letters and maybe σ, Ω, etc.)
    # Allow Greek letters but not math operators or numbers
    simple_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        "σΣ"
    )

    # If name contains operators, spaces, numbers, or other complex chars, return empty
    if any(c in name for c in "+-*/^√() 0123456789"):
        return ""

    # Keep only allowed characters
    sanitized = "".join(c for c in name if c in simple_chars)
    return sanitized


@dataclass
class TransitionSelector:
    """Describes a laser-driven transition with associated parameters.

    Attributes:
        ground (Sequence[CoupledState]): Ground states involved in the transition
        excited (Sequence[CoupledState]): Excited states involved in the transition
        polarizations (Sequence[npt.NDArray[np.complex128]]): Polarization vectors for
            each field component
        polarization_symbols (List[smp.Symbol]): Symbolic variables for polarization
            amplitudes
        Ω (smp.Symbol): Symbolic Rabi frequency
        δ (smp.Symbol): Symbolic detuning from resonance
        description (str | None): Human-readable description of the transition
        type (str | None): Type of transition (e.g., "optical", "microwave")
        ground_main (CoupledState | None): Main ground state for the transition
        excited_main (CoupledState | None): Main excited state for the transition
        phase_modulation (bool): Whether phase modulation is applied. Defaults to False.
    """

    ground: Sequence[states.CoupledState]
    excited: Sequence[states.CoupledState]
    polarizations: Sequence[npt.NDArray[np.complex128]]
    polarization_symbols: List[smp.Symbol]
    Ω: smp.Symbol
    δ: smp.Symbol
    description: Optional[str] = None
    type: Optional[str] = None
    ground_main: Optional[states.CoupledState] = None
    excited_main: Optional[states.CoupledState] = None
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
    ground_mains: Optional[Sequence[states.CoupledState]] = None,
    excited_mains: Optional[Sequence[states.CoupledState]] = None,
    phase_modulations: Optional[Sequence[bool]] = None,
) -> List[TransitionSelector]:
    """Generate TransitionSelector objects from transition and polarization specs.

    Creates a list of TransitionSelector objects that describe the laser-driven
    transitions in an optical Bloch equation system. Each selector includes the
    ground and excited states, polarization vectors, and symbolic parameters (Rabi
    frequency Ω and detuning δ) for use in symbolic calculations.

    Args:
        transitions (Sequence[OpticalTransition | MicrowaveTransition]): Transitions
            to include in the system
        polarizations (Sequence[Sequence[Polarization]]): List of polarizations for
            each transition. Inner sequence contains polarizations for a single
            transition (e.g., [pol_x, pol_y] for two-polarization transition).
        ground_mains (Sequence[CoupledState] | None): Optional main ground state for
            each transition. Used to identify the primary coupling. Defaults to None.
        excited_mains (Sequence[CoupledState] | None): Optional main excited state for
            each transition. Used to identify the primary coupling. Defaults to None.
        phase_modulations (Sequence[bool] | None): Whether each transition has phase
            modulation applied. Defaults to None (no modulation).

    Returns:
        List[TransitionSelector]: List of TransitionSelector objects, one per transition

    Example:
        >>> transitions = [OpticalTransition(J_ground=0, J_excited=1)]
        >>> pols = [[polarization_X, polarization_Y]]
        >>> selectors = generate_transition_selectors(transitions, pols)
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
        else:
            raise TypeError(
                f"Transition must be OpticalTransition or MicrowaveTransition, "
                f"got {type(transition).__name__}"
            )

        if phase_modulations is None:
            phase_modulation = False
        else:
            phase_modulation = phase_modulations[idt]

        # Initialize main states
        ground_main: Optional[states.CoupledState] = None
        excited_main: Optional[states.CoupledState] = None
        if ground_mains is None:
            ground_main, excited_main = select_main_states(
                ground_states_approx, excited_states_approx, polarization[0].vector
            )

        # Create polarization symbols with sanitized names
        pol_symbols = []
        for idx, p in enumerate(polarization):
            sanitized = _sanitize_polarization_name(p.name)
            if sanitized:
                # Use sanitized name if it's simple (e.g., "X", "Y", "σp")
                pol_symbols.append(smp.Symbol(f"P{sanitized}{idt}", complex=True))
            else:
                # Use generic naming for complex expressions (e.g., "PA0", "PB0")
                # A for first polarization, B for second, etc.
                pol_label = chr(ord("A") + idx)
                pol_symbols.append(smp.Symbol(f"P{pol_label}{idt}", complex=True))

        transition_selectors.append(
            TransitionSelector(
                ground=ground_states_approx,
                excited=excited_states_approx,
                polarizations=[p.vector for p in polarization],
                polarization_symbols=pol_symbols,
                Ω=smp.Symbol(f"Ω{idt}", complex=True),
                δ=smp.Symbol(f"δ{idt}", real=True),
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
    I1 = float(ground_state.I1)
    I2 = float(ground_state.I2)

    if transition_types is None:
        transition_types = [t for t in OpticalTransitionType]

    transitions = []
    for transition_type in transition_types:
        ΔJ = transition_type.value
        J_excited = J + ΔJ
        _transitions = [
            OpticalTransition(transition_type, J, float(F1), int(F))
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
