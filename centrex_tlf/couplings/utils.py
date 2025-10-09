from typing import Iterable, List, Literal, Sequence, Tuple, Union, cast, overload

import numpy as np
import numpy.typing as npt

from centrex_tlf import states
from centrex_tlf.states.states import CoupledBasisState

from .polarization import (
    polarization_X,
    polarization_Y,
    polarization_Z,
    polarization_σm,
    polarization_σp,
)

__all__: List[str] = []


def check_transition_coupled_allowed(
    ground: states.CoupledBasisState,
    excited: states.CoupledBasisState,
) -> bool:
    """
    Check if an electric-dipole (E1) transition between two coupled-basis states is allowed.

    Applies the following selection rules:
        • Parity must change: P_ground * P_excited == -1
        • Total angular momentum change: |ΔF| ≤ 1
        • The case F_ground = F_excited = 0 is forbidden.

    Args:
        ground (CoupledBasisState): The ground-state coupled basis.
        excited (CoupledBasisState): The excited-state coupled basis.

    Returns:
        bool: True if the transition is allowed, False otherwise.

    Raises:
        ValueError: If either state's parity (P) or total F is not set (None).
    """
    if ground.P is None or excited.P is None:
        raise ValueError("Both states must have parity P set")
    if ground.F is None or excited.F is None:
        raise ValueError("Both states must have total F set")

    # 1. Parity change
    if ground.P * excited.P != -1:
        return False

    # 2. Hyperfine selection rule
    dF = abs(excited.F - ground.F)
    if dF > 1:
        return False
    if dF == 0 and ground.F == 0:
        return False

    return True


@overload
def check_transition_coupled_allowed_polarization(
    ground_state: states.CoupledBasisState,
    excited_state: states.CoupledBasisState,
    ΔmF_allowed: Union[int, Iterable[int]],
    *,
    return_err: Literal[True] = ...,
    ΔmF_absolute: bool = ...,
) -> Tuple[bool, str]: ...


@overload
def check_transition_coupled_allowed_polarization(
    ground_state: states.CoupledBasisState,
    excited_state: states.CoupledBasisState,
    ΔmF_allowed: Union[int, Iterable[int]],
    *,
    return_err: Literal[False],
    ΔmF_absolute: bool = ...,
) -> bool: ...


def check_transition_coupled_allowed_polarization(
    ground_state: states.CoupledBasisState,
    excited_state: states.CoupledBasisState,
    ΔmF_allowed: Union[int, Iterable[int]],
    *,
    return_err: bool = True,
    ΔmF_absolute: bool = False,
) -> Union[bool, Tuple[bool, str]]:
    """
    Basic E1 selection-rule check for a hyperfine transition
    including photon-polarization (ΔmF) information.

    Parameters
    ----------
    ground_state, excited_state : states.CoupledBasisState
        Initial and final coupled basis states (must have P, F, mF set).
    ΔmF_allowed : int | Iterable[int]
        Allowed change in mF imposed by the photon polarization.
        Pass a single value (e.g. 0) or something like {-1, +1}.
    return_err : bool, default True
        If True, return (allowed, message); else return allowed only.
    ΔmF_absolute : bool, default False
        If True, compare |ΔmF| against |ΔmF_allowed|.

    Returns
    -------
    bool | (bool, str)
        Whether the transition is allowed and, optionally, an explanation.
    """
    # --- sanity checks -------------------------------------------------------
    if ground_state.P is None or excited_state.P is None:
        raise ValueError("Both states must have parity P set")
    if ground_state.F is None or excited_state.F is None:
        raise ValueError("Both states must have total F set")
    if ground_state.mF is None or excited_state.mF is None:
        raise ValueError("Both states must have magnetic quantum number mF set")

    # --- helper lambdas ------------------------------------------------------
    parity_ok = ground_state.P * excited_state.P == -1  # E1 requires P_i ≠ P_f
    ΔF = int(excited_state.F - ground_state.F)
    F_ok = (abs(ΔF) <= 1) and not (ground_state.F == excited_state.F == 0)

    ΔmF = int(excited_state.mF - ground_state.mF)
    if ΔmF_absolute:
        ΔmF = abs(ΔmF)

    # Allow an int or any iterable of ints for ΔmF_allowed
    if isinstance(ΔmF_allowed, int):
        allowed_set = {ΔmF_allowed}
    else:
        allowed_set = set(ΔmF_allowed)

    if ΔmF_absolute:
        allowed_set = {abs(v) for v in allowed_set}

    mF_ok = ΔmF in allowed_set

    # Special “stretched” prohibition: ΔF = 0, ΔmF = 0, and mF = 0
    stretched_forbidden = (ΔF == 0) and (ΔmF == 0) and (ground_state.mF == 0)

    # --- aggregate result ----------------------------------------------------
    allowed = parity_ok and F_ok and mF_ok and not stretched_forbidden

    if not return_err:
        return allowed

    # Build minimal error string only when needed
    if allowed:
        return True, ""

    errors = []
    if not parity_ok:
        errors.append("parity must change (P_i ≠ P_f)")
    if not F_ok:
        errors.append(f"|ΔF| must be ≤ 1 and F=0→0 forbidden (ΔF={ΔF})")
    if not mF_ok:
        errors.append(f"ΔmF={ΔmF} not in allowed set {sorted(allowed_set)}")
    if stretched_forbidden:
        errors.append("F=0, mF=0 → F'=0, mF'=0 transition is forbidden")

    return False, "transition not allowed; " + ", ".join(errors)


def assert_transition_coupled_allowed(
    ground_state: states.CoupledBasisState,
    excited_state: states.CoupledBasisState,
    ΔmF_allowed: int | Iterable[int],
) -> bool:
    """Check whether the transition is allowed based on the quantum numbers.
    Raises an AssertionError if the transition is not allowed.

    Args:
        ground_state (CoupledBasisState): ground CoupledBasisState
        excited_state (CoupledBasisState): excited CoupledBasisState

    Returns:
        tuple: allowed boolean
    """
    ret = check_transition_coupled_allowed_polarization(
        ground_state, excited_state, ΔmF_allowed, return_err=True
    )
    allowed, errors = ret
    assert allowed, errors
    return allowed


def ΔmF_allowed(
    polarization: npt.NDArray[np.complex128],
    tol: float = 1e-10,
) -> Union[int, npt.NDArray[np.int_]]:
    """
    Determine which hyperfine ΔmF transitions are allowed by a photon’s polarization.

    Args:
        polarization (npt.NDArray[np.complex128]): Normalized Jones vector
            [Ex, Ey, Ez] in the lab frame.
        tol (float, optional): Threshold below which a spherical component
            is considered zero. Defaults to 1e-10.

    Returns:
        int: the single allowed ΔmF value, if only one is nonzero.
        npt.NDArray[np.int_]: array of allowed ΔmF values, otherwise.

    Raises:
        ValueError: If `polarization` is not shape (3,) or if all spherical components
            are below `tol` (i.e. effectively zero).
    """
    polarization = np.asarray(polarization, dtype=np.complex128)
    if polarization.shape != (3,):
        raise ValueError("polarization must be a (3,) array in (Ex, Ey, Ez) order")

    # Convert (Ex,Ey,Ez) → spherical components (q = +1, 0, –1)
    ex, ey, ez = polarization
    e_p1 = -(ex + 1j * ey) / np.sqrt(2)  # σ⁺  (ΔmF = +1)
    e_m1 = (ex - 1j * ey) / np.sqrt(2)  # σ⁻  (ΔmF = –1)
    e_0 = ez  # π    (ΔmF =  0)

    allowed = []
    if abs(e_p1) > tol:
        allowed.append(+1)
    if abs(e_m1) > tol:
        allowed.append(-1)
    if abs(e_0) > tol:
        allowed.append(0)

    if not allowed:
        raise ValueError(
            "Polarization vector is zero or unrecognized (all components < tol)"
        )

    allowed_sorted = sorted(set(allowed))
    if len(allowed_sorted) == 1:
        return allowed_sorted[0]
    return np.array(allowed_sorted, dtype=np.int_)


def select_main_states(
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    polarization: npt.NDArray[np.complex128],
) -> Tuple[states.CoupledState, states.CoupledState]:
    """
    Select the main ground and excited states based on allowed transitions.

    Scans excited states first, ground states second, and builds the list of
    allowed transitions in that order. If any allowed transition has a ground-state
    mF = 0, returns the last one found. Otherwise, returns the middle element
    of the allowed transitions.

    Args:
        ground_states (Sequence[states.CoupledState]): List of ground states.
        excited_states (Sequence[states.CoupledState]): List of excited states.
        polarization (npt.NDArray[np.complex128]): Normalized Jones vector
            [Ex, Ey, Ez] in the lab frame.

    Returns:
        Tuple[states.CoupledState, states.CoupledState]: Selected ground and excited states.

    Raises:
        ValueError: If none of the supplied ground and excited states have allowed transitions.
    """
    # Get the ΔmF value(s) allowed by the photon polarization
    ΔmF_raw = ΔmF_allowed(polarization)
    if isinstance(ΔmF_raw, (int, np.integer)):
        # Scalar → wrap in a tuple
        ΔmF_allowed_iterable: Tuple[int, ...] = (int(ΔmF_raw),)
    else:
        # NumPy array → copy to a tuple of plain ints
        ΔmF_allowed_iterable = tuple(
            int(x) for x in cast(npt.NDArray[np.int64], ΔmF_raw).tolist()
        )

    # Build the list of allowed transitions in the original order
    allowed_transitions = []  # (g_idx, e_idx, mF_exc)
    indices_gnd_mF0 = []  # (g_idx, e_idx, mF_gnd)

    for e_idx, exc in enumerate(excited_states):
        exc_bs = cast(CoupledBasisState, exc.largest)
        for g_idx, gnd in enumerate(ground_states):
            gnd_bs = cast(CoupledBasisState, gnd.largest)

            if check_transition_coupled_allowed_polarization(
                gnd_bs,
                exc_bs,
                ΔmF_allowed_iterable,
                return_err=False,
            ):
                allowed_transitions.append((g_idx, e_idx, exc_bs.mF))
                if gnd_bs.mF == 0:
                    indices_gnd_mF0.append((g_idx, e_idx, gnd_bs.mF))

    if not allowed_transitions:
        raise ValueError(
            "None of the supplied ground and excited states have allowed transitions"
        )

    # Select the main pair
    if indices_gnd_mF0:
        g_idx, e_idx, _ = indices_gnd_mF0[-1]  # last one with mF_gnd = 0
    else:
        mid = len(allowed_transitions) // 2
        g_idx, e_idx, _ = allowed_transitions[mid]  # middle element

    return ground_states[g_idx], excited_states[e_idx]
