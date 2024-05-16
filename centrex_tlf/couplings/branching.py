from typing import List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from centrex_tlf import states

from .matrix_elements import calculate_ED_ME_mixed_state

__all__ = ["calculate_br", "generate_br_dataframe"]


def calculate_br(
    excited_state: states.CoupledState,
    ground_states: Sequence[states.CoupledState],
    tol: float = 1e-3,
) -> npt.NDArray[np.float_]:
    # matrix elements between the excited state and the ground states
    MEs = np.zeros((len(ground_states)), dtype=np.complex_)

    for idg, ground_state in enumerate(ground_states):
        MEs[idg] = calculate_ED_ME_mixed_state(
            ground_state.remove_small_components(tol=tol),
            excited_state.remove_small_components(tol=tol),
        )

    # Calculate branching ratios
    BRs = np.abs(MEs) ** 2 / (np.sum(np.abs(MEs) ** 2)).astype(np.float_)
    return BRs


def generate_br_dataframe(
    ground_states: Sequence[states.CoupledState],
    excited_states: Sequence[states.CoupledState],
    group_ground: Optional[str] = None,
    group_excited: bool = True,
    remove_zeros: bool = True,
    tolerance: float = 1e-3,
) -> pd.DataFrame:
    """
    Generate a pandas DataFrame of branching ratios given a set of ground states and
    excited states.

    Args:
        ground_states (Sequence[states.State]): superpositions of ground state
                                                CoupledBasisStates
        excited_states (Sequence[states.State]): superpositions of excited state
                                                CoupledBasisStates
        group_ground (Optional[str], optional): Group ground states with "J" or "mF".
                                                Defaults to None.
        group_excited (bool, optional): Group all excited states in a single column.
                                        Defaults to True.
        remove_zeros (bool, optional): Remove states with zero branching. Defaults to
                                        True.

    Raises:
        AssertionError: _description_

    Returns:
        _type_: _description_
    """
    br: List[npt.NDArray[np.float_]] = []
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
        data["states"] = np.asarray(data["states"])[m]
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
        F1_unique: npt.NDArray[np.float_] = np.unique(
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
