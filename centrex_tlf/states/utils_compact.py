from typing import List, Sequence, cast

import numpy as np
import numpy.typing as npt

from .states import CoupledBasisState, CoupledState

__all__ = ["compact_QN_coupled_indices"]


def compact_QN_coupled_indices(
    QN: Sequence[CoupledState], indices_compact: npt.NDArray[np.int_]
) -> List[CoupledState]:
    """Compact the states given by indices in indices_compact

    Args:
        QN (list): states
        indices_compact (list, array): indices to compact into a single state

    Returns:
        list: compacted states
    """
    QNc = [QN[idx] for idx in indices_compact]

    def slc(s):
        return s.largest

    Js = np.unique([slc(s).J for s in QNc if slc(s).J is not None])
    F1s = np.unique([slc(s).F1 for s in QNc if slc(s).F1 is not None])
    Fs = np.unique([slc(s).F for s in QNc if slc(s).F is not None])
    mFs = np.unique([slc(s).mF for s in QNc if slc(s).mF is not None])
    Ps = np.unique([slc(s).P for s in QNc if slc(s).P is not None])

    QNcompact = [qn for idx, qn in enumerate(QN) if idx not in indices_compact[1:]]

    state_rep = cast(CoupledBasisState, QNcompact[indices_compact[0]].largest)
    if len(Js) != 1:
        state_rep = CoupledBasisState(
            F=state_rep.F,
            mF=state_rep.mF,
            F1=state_rep.F1,
            J=None,
            I1=state_rep.I1,
            I2=state_rep.I2,
            Omega=state_rep.Omega,
            P=state_rep.P,
            electronic_state=state_rep.electronic_state,
            basis=state_rep.basis,
        )
        # state_rep.J = None
    if len(F1s) != 1:
        state_rep = CoupledBasisState(
            F=state_rep.F,
            mF=state_rep.mF,
            F1=None,
            J=state_rep.J,
            I1=state_rep.I1,
            I2=state_rep.I2,
            Omega=state_rep.Omega,
            P=state_rep.P,
            electronic_state=state_rep.electronic_state,
            basis=state_rep.basis,
        )
    if len(Fs) != 1:
        state_rep = CoupledBasisState(
            F=None,
            mF=state_rep.mF,
            F1=state_rep.F1,
            J=state_rep.J,
            I1=state_rep.I1,
            I2=state_rep.I2,
            Omega=state_rep.Omega,
            P=state_rep.P,
            electronic_state=state_rep.electronic_state,
        )
    if len(mFs) != 1:
        state_rep = CoupledBasisState(
            F=state_rep.F,
            mF=None,
            F1=state_rep.F1,
            J=state_rep.J,
            I1=state_rep.I1,
            I2=state_rep.I2,
            Omega=state_rep.Omega,
            P=state_rep.P,
            electronic_state=state_rep.electronic_state,
        )
    if len(Ps) != 1:
        state_rep = CoupledBasisState(
            F=state_rep.F,
            mF=state_rep.mF,
            F1=state_rep.F1,
            J=state_rep.J,
            I1=state_rep.I1,
            I2=state_rep.I2,
            Omega=state_rep.Omega,
            P=None,
            electronic_state=state_rep.electronic_state,
        )

    # make it a state again instead of uncoupled basisstate
    QNcompact[indices_compact[0]] = (1.0 + 0j) * state_rep

    return QNcompact
