"""Detection observables for the RF Ramsey simulation.

Survival probability projects each Psi_final column onto the closest dressed
eigenstate of H_at_end among detection targets. Per-J populations sum bare-basis
probabilities by J manifold.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .states import TargetSpec, dressed_initial_states


def survival_probability(
    Psi_final: npt.NDArray[np.complex128],
    H_at_end: npt.NDArray[np.complex128],
    QN: npt.NDArray,
    targets: TargetSpec,
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Per-initial-state projection onto the closest dressed eigenstate at the end.

    Args:
        Psi_final: (N, K) — columns are the propagated initial states.
        H_at_end: (N, N) Hermitian, diagonalized to find dressed targets at end.
        QN: basis ordering used to construct target state vectors.
        targets: K target specs (one per initial state, or a selector resolving
            to K targets).

    Returns:
        (probabilities of shape (K,), eigenstate indices used)
    """
    Psi_target, eig_idx, _overlap = dressed_initial_states(H_at_end, QN, targets)
    if Psi_target.shape[1] != Psi_final.shape[1]:
        raise ValueError(
            f"detection targets resolved to {Psi_target.shape[1]} states, "
            f"but Psi_final has {Psi_final.shape[1]} columns"
        )
    # |<target_k | psi_k>|^2 along columns
    overlaps = np.einsum("nk,nk->k", Psi_target.conj(), Psi_final)
    return np.abs(overlaps) ** 2, eig_idx


def per_j_populations(
    Psi: npt.NDArray[np.complex128],
    j_indices: dict[int, npt.NDArray[np.int_]],
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Bare-basis population per J manifold.

    Returns (probs of shape (K, n_J), list of J values in column order).
    """
    Js = list(j_indices.keys())
    K = Psi.shape[1]
    out = np.zeros((K, len(Js)), dtype=np.float64)
    pop = np.abs(Psi) ** 2  # (N, K)
    for j_col, J in enumerate(Js):
        idx = j_indices[J]
        out[:, j_col] = pop[idx, :].sum(axis=0)
    return out, Js
