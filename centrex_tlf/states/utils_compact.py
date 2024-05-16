from typing import List, Sequence

import numpy as np
import numpy.typing as npt

from .states import CoupledState

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

    state_rep = QNcompact[indices_compact[0]].largest
    if len(Js) != 1:
        state_rep.J = None
    if len(F1s) != 1:
        state_rep.F1 = None
    if len(Fs) != 1:
        state_rep.F = None
    if len(mFs) != 1:
        state_rep.mF = None
    if len(Ps) != 1:
        state_rep.P = None

    # make it a state again instead of uncoupled basisstate
    QNcompact[indices_compact[0]] = (1.0 + 0j) * state_rep

    return QNcompact
