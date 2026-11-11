from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # pandas is only needed by generate_br_dataframe; see coupling_matrix.py
    import pandas as pd

from centrex_tlf import hamiltonian, states

__all__ = ["calculate_br", "generate_br_dataframe"]


def calculate_br(
    excited_state: states.CoupledState,
    ground_states: Sequence[states.CoupledState],
    tol: float = 1e-3,
) -> npt.NDArray[np.floating]:
    """Calculate branching ratios for spontaneous emission from an excited state.

    Computes the relative probability for an excited state to decay to each of the
    ground states via electric dipole transitions in isotropic free space. For each
    final state, sums |⟨g|D_a|e⟩|² over three orthogonal Cartesian polarizations
    before normalizing over the supplied ground states. Amplitudes remain coherent
    within each polarization component, but distinct components are summed
    incoherently. A common polarization-average factor of 1/3 cancels.

    Args:
        excited_state (CoupledState): Excited state that can decay
        ground_states (Sequence[CoupledState]): Possible ground states for decay
        tol (float): Remove state components with amplitude < tol before calculating
            matrix elements. Defaults to 1e-3.

    Returns:
        npt.NDArray[np.floating]: Array of branching ratios, length len(ground_states),
            normalized so that Σ BR[i] = 1

    Raises:
        ValueError: A nonempty ground-state list has zero total dipole strength.

    Example:
        >>> BRs = calculate_br(excited_state, ground_states)
        >>> print(f"Decay to state 0: {BRs[0]*100:.1f}%")
    """
    if not len(ground_states):
        return np.zeros(0, dtype=np.float64)
    return _branching_ratios_reduced(
        _reduce_for_branching(excited_state, tol),
        [_reduce_for_branching(ground_state, tol) for ground_state in ground_states],
    )


# Cartesian unit polarizations, as the tuples ED_ME_coupled is cached on.
_CARTESIAN_POLARIZATIONS = tuple(
    tuple(polarization) for polarization in np.eye(3, dtype=np.complex128)
)


def _reduce_for_branching(state: states.CoupledState, tol: float) -> states.CoupledState:
    """Drop small components and move to the Omega basis.

    Depends on one state only, so callers looping over excited states can do it
    once per ground state instead of once per (ground, excited) pair.
    """
    return hamiltonian.to_omega_basis(state.remove_small_components(tol=tol))


def _branching_ratios_reduced(
    excited_reduced: states.CoupledState,
    ground_states_reduced: Sequence[states.CoupledState],
) -> npt.NDArray[np.floating]:
    """Branching ratios for states already passed through `_reduce_for_branching`.

    Same sum as three `generate_ED_ME_mixed_state` calls per ground state (one
    per Cartesian polarization), accumulated in the same component order, but
    in a single pass over the basis-state pairs and skipping pairs the E1
    selection rules make exactly zero before the cached lookup.
    """
    strengths = np.zeros(len(ground_states_reduced), dtype=np.float64)
    ED_ME_coupled = hamiltonian.ED_ME_coupled
    pol_x, pol_y, pol_z = _CARTESIAN_POLARIZATIONS
    excited_data = excited_reduced.data
    for idg, ground_reduced in enumerate(ground_states_reduced):
        me_x = me_y = me_z = 0j
        for amp_bra, bra in ground_reduced.data:
            amp_bra_conj = amp_bra.conjugate()
            for amp_ket, ket in excited_data:
                if (
                    abs(bra.J - ket.J) > 1
                    or abs(bra.mF - ket.mF) > 1
                    or abs(bra.F - ket.F) > 1
                    or abs(bra.Omega - ket.Omega) > 1
                ):
                    continue
                weight = amp_bra_conj * amp_ket
                me_x += weight * ED_ME_coupled(bra, ket, pol_vec=pol_x, rme_only=False)
                me_y += weight * ED_ME_coupled(bra, ket, pol_vec=pol_y, rme_only=False)
                me_z += weight * ED_ME_coupled(bra, ket, pol_vec=pol_z, rme_only=False)
        strengths[idg] = sum(abs(me) ** 2 for me in (me_x, me_y, me_z))

    total_strength = strengths.sum()
    if total_strength == 0:
        raise ValueError(
            "Cannot normalize branching ratios: supplied ground states have zero total dipole strength"
        )
    return strengths / total_strength


def generate_br_dataframe(
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    group_ground: Optional[str] = None,
    group_excited: bool = True,
    remove_zeros: bool = True,
    tolerance: float = 1e-3,
) -> "pd.DataFrame":
    """Generate pandas DataFrame of branching ratios for optical transitions.

    Creates a table showing the branching ratios for spontaneous emission from excited
    states to ground states. Rows represent ground states (or groups thereof) and columns
    represent excited states (or groups thereof). Values are normalized branching ratios
    summing to 1 for each excited state column.

    Args:
        ground_states (Sequence[CoupledState]): Ground states (superpositions of
            CoupledBasisStates) that can be populated by decay
        excited_states (Sequence[CoupledState]): Excited states (superpositions of
            CoupledBasisStates) that can decay
        group_ground (str | None): How to group ground states for display. Options:
            - None: Show individual states
            - "J": Group by rotational quantum number J
            - "mF": Group by J, F1, F quantum numbers
            Defaults to None.
        group_excited (bool): If True, group excited states by J, F1, F into single
            columns. Defaults to True.
        remove_zeros (bool): If True, remove ground states with zero total branching
            ratio. Defaults to True.
        tolerance (float): Remove state components with amplitude < tolerance before
            calculating matrix elements. Defaults to 1e-3.

    Returns:
        pd.DataFrame: DataFrame with ground states as rows (indexed by state labels) and
            excited states as columns, containing normalized branching ratios

    Raises:
        AssertionError: If group_ground is not None, "J", or "mF"

    Example:
        >>> df = generate_br_dataframe(ground_states, excited_states, group_ground="J")
        >>> print(df)  # Shows branching ratios grouped by ground state J
    """
    import pandas as pd

    br: List[npt.NDArray[np.floating]] = []
    for es in excited_states:
        br.append(calculate_br(es, ground_states, tolerance))

    brs = np.sum(br, axis=0)

    if group_ground is not None:
        if group_ground == "J":
            J_unique = np.unique([s.largest.J for s in ground_states])
            indices_group = [
                states.QuantumSelector(
                    J=Ji, electronic=states.ElectronicState.X
                ).get_indices(ground_states)
                for Ji in J_unique
            ]
            data = {"states": [f"|X, J = {Ji}>" for Ji in J_unique]}
        elif group_ground == "mF":
            mF_selectors = np.unique(
                [(s.largest.J, s.largest.F1, s.largest.F) for s in ground_states],  # type: ignore # noqa: 203
                axis=0,
            )
            indices_group = [
                states.QuantumSelector(
                    J=J, F1=F1, F=F, electronic=states.ElectronicState.X
                ).get_indices(ground_states)
                for J, F1, F in mF_selectors
            ]
            data = {
                "states": [
                    [ground_states[idx] for idx in ind][0].largest.state_string_custom(
                        ["electronic", "J", "F1", "F"]
                    )
                    for ind in indices_group
                ]
            }
        else:
            raise AssertionError("group_ground not equal to either J or mF")
        br = [np.array([bri[ind].sum() for ind in indices_group]) for bri in br]
        brs = np.sum(br, axis=0)
        if remove_zeros:
            m = brs != 0
        else:
            m = np.ones(len(brs), dtype=bool)
        data["states"] = np.asarray(data["states"])[m].tolist()
    else:
        if remove_zeros:
            m = brs != 0
        else:
            m = np.ones(len(brs), dtype=bool)
        data = {
            "states": [
                qn.largest.state_string_custom(  # type: ignore
                    ["electronic", "J", "F1", "F", "mF"]
                )
                for qn in [s for ids, s in enumerate(ground_states) if m[ids]]
            ]
        }

    br_dataframe = pd.DataFrame(data=data)
    if group_excited:
        J_unique = np.unique([s.largest.J for s in excited_states])
        F1_unique: npt.NDArray[np.floating] = np.unique(
            [s.largest.F1 for s in excited_states]  # type: ignore
        )
        F_unique: npt.NDArray[np.int_] = np.unique(
            [s.largest.F for s in excited_states]  # type: ignore
        )
        quantum_selectors = [
            states.QuantumSelector(
                J=Ji, F1=F1i, F=Fi, electronic=states.ElectronicState.B
            )
            for Ji in J_unique
            for F1i in F1_unique
            for Fi in F_unique
        ]
        indices_group = [qs.get_indices(excited_states) for qs in quantum_selectors]
        indices_group = [ind for ind in indices_group if len(ind) > 0]
        for ind in indices_group:
            s = excited_states[ind[0]].largest
            bri = np.sum([br[i] for i in ind], axis=0)
            bri /= np.sum(bri)
            br_dataframe[
                s.state_string_custom(["electronic", "J", "F1", "F"])  # type: ignore
            ] = bri[m]
    else:
        for idb, brv in enumerate(br):
            br_dataframe[
                excited_states[idb].largest.state_string_custom(  # type: ignore
                    ["electronic", "J", "F1", "F", "mF"]
                )
            ] = np.asarray(brv)[m] / np.sum(brs)
    return br_dataframe.set_index("states")
