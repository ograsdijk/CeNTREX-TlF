"""Initial-state preparation and basis-index helpers for the uncoupled X basis.

`QuantumSelector` in centrex_tlf only supports the coupled basis, so we provide
`select_uncoupled` here for filtering the uncoupled basis by quantum numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh as _scipy_eigh

from centrex_tlf.states import (
    ElectronicState,
    UncoupledBasisState,
    UncoupledState,
    find_closest_vector_idx,
)


@dataclass
class UncoupledSelector:
    """Filter UncoupledBasisStates by quantum numbers.

    Each field accepts None (matches all), a single value, or a sequence of values.
    Multiple fields combine with AND.
    """

    J: Union[int, Sequence[int], None] = None
    mJ: Union[int, Sequence[int], None] = None
    m1: Union[float, Sequence[float], None] = None
    m2: Union[float, Sequence[float], None] = None
    electronic: Union[ElectronicState, Sequence[ElectronicState], None] = None

    def matches(self, bs: UncoupledBasisState) -> bool:
        def _ok(value, allowed) -> bool:
            if allowed is None:
                return True
            if isinstance(allowed, (list, tuple, set, np.ndarray)):
                return value in allowed
            return value == allowed

        if not _ok(bs.J, self.J):
            return False
        if not _ok(bs.mJ, self.mJ):
            return False
        if not _ok(bs.m1, self.m1):
            return False
        if not _ok(bs.m2, self.m2):
            return False
        if not _ok(bs.electronic_state, self.electronic):
            return False
        return True

    def get_indices(self, QN: npt.NDArray) -> npt.NDArray[np.int_]:
        return np.array([i for i, bs in enumerate(QN) if self.matches(bs)], dtype=np.int_)


TargetSpec = Union[
    UncoupledSelector,
    Sequence[UncoupledState],
    Sequence[UncoupledBasisState],
]


def select_uncoupled(QN: npt.NDArray, **kwargs) -> npt.NDArray[np.int_]:
    """Convenience: select indices matching given keyword filters (J=, mJ=, ...)."""
    return UncoupledSelector(**kwargs).get_indices(QN)


def targets_to_state_vectors(
    targets: TargetSpec,
    QN: npt.NDArray,
) -> tuple[list[UncoupledState], npt.NDArray[np.complex128]]:
    """Resolve a TargetSpec into (list of UncoupledState, stacked state-vector matrix).

    Returns (states, vectors) where:
        states[k] is a normalized UncoupledState
        vectors has shape (N, K), each column a state vector in the QN ordering
    """
    if isinstance(targets, UncoupledSelector):
        idx = targets.get_indices(QN)
        if len(idx) == 0:
            raise ValueError("UncoupledSelector matched no basis states")
        states = [UncoupledState([(1.0 + 0j, QN[i])]) for i in idx]
    else:
        seq = list(targets)
        if len(seq) == 0:
            raise ValueError("targets sequence is empty")
        states = []
        for t in seq:
            if isinstance(t, UncoupledBasisState):
                states.append(UncoupledState([(1.0 + 0j, t)]))
            elif isinstance(t, UncoupledState):
                states.append(t)
            else:
                raise TypeError(
                    f"target must be UncoupledState, UncoupledBasisState, or UncoupledSelector; "
                    f"got {type(t).__name__}"
                )

    N = len(QN)
    K = len(states)
    vectors = np.zeros((N, K), dtype=np.complex128)
    for k, st in enumerate(states):
        v = st.state_vector(QN).astype(np.complex128, copy=False)
        n = np.linalg.norm(v)
        if n == 0.0:
            raise ValueError(f"target {k} projects to zero on QN")
        vectors[:, k] = v / n
    return states, vectors


def dressed_initial_states(
    H_at_start: npt.NDArray[np.complex128],
    QN: npt.NDArray,
    targets: TargetSpec,
) -> tuple[npt.NDArray[np.complex128], list[int], list[float]]:
    """Build Psi0 of shape (N, K) by selecting the dressed eigenstate of
    H_at_start that has maximum overlap with each target's state vector.

    Returns:
        Psi0: (N, K) complex array; columns are the chosen eigenvectors.
        eigenstate_indices: list of K column indices of V used.
        overlap_probs: list of K |<target|chosen_eigvec>|^2 values (sanity diagnostic).
    """
    _states, target_vecs = targets_to_state_vectors(targets, QN)
    K = target_vecs.shape[1]

    D, V = np.linalg.eigh(H_at_start)
    del D  # eigenvalues unused for state-selection

    Psi0 = np.empty((V.shape[0], K), dtype=np.complex128)
    indices: list[int] = []
    overlaps: list[float] = []
    used: set[int] = set()
    for k in range(K):
        tv = target_vecs[:, k]
        # Mask already-used eigenstates so two targets cannot grab the same eigvec
        candidate_mask = np.array([i not in used for i in range(V.shape[1])])
        if not candidate_mask.any():
            raise RuntimeError("ran out of eigenstates while assigning targets")
        candidate_vectors = V[:, candidate_mask]
        rel_idx = find_closest_vector_idx(tv, candidate_vectors)
        # Map back to absolute eigenstate index
        absolute_indices = np.flatnonzero(candidate_mask)
        idx = int(absolute_indices[rel_idx])
        used.add(idx)
        indices.append(idx)
        Psi0[:, k] = V[:, idx]
        overlaps.append(float(np.abs(tv.conj() @ V[:, idx]) ** 2))

    return Psi0, indices, overlaps


def adiabatic_dressed_initial_states(
    H_func: Callable[..., npt.NDArray[np.complex128]],
    E_init: Sequence[float] | npt.NDArray[np.float64],
    E_target: Sequence[float] | npt.NDArray[np.float64],
    QN: npt.NDArray,
    targets: TargetSpec,
    *,
    n_steps: int = 50,
    B: Sequence[float] | npt.NDArray[np.float64] = (0.0, 0.0, 0.0),
) -> tuple[npt.NDArray[np.complex128], list[int], list[float]]:
    """Adiabatic-ancestor state preparation.

    For each target bare state:
      1. Diagonalize H(E_target, B) and pick the eigenvector with maximum
         overlap to the target (this is the dressed eigenstate at the target
         field that "is" mostly the bare target state).
      2. Walk E linearly from E_target back to E_init in `n_steps` interior
         steps. At each step, diagonalize H and pick the eigenvector with
         maximum overlap to the previous-step eigenvector (adiabatic tracking).
      3. The final eigenvector at E_init is the column of Psi0 to use.

    Use this when the experimentally-prepared state is best characterized by
    its bare quantum numbers AT THE HIGH-FIELD plateau (e.g. mJ=-1, m1=-1/2,
    m2=-1/2 in the polarized molecule), but the simulation must START at the
    low-field entrance (where the eigenstate is heavily mixed in a different
    way and cannot be picked by simple bare-state matching).

    Returns (Psi0 of shape (N, K), final eigenstate indices at E_init,
    overlap probabilities at E_target).
    """
    _states, target_vecs = targets_to_state_vectors(targets, QN)
    K = target_vecs.shape[1]
    E_init_arr = np.asarray(E_init, dtype=np.float64).reshape(3)
    E_target_arr = np.asarray(E_target, dtype=np.float64).reshape(3)
    B_arr = np.asarray(B, dtype=np.float64).reshape(3)

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    # Step 1: pick eigenvectors at E_target.
    H_target = H_func(E_target_arr, B_arr)
    D_target, V_target = _scipy_eigh(H_target, driver="evr")
    del D_target

    used: set[int] = set()
    current_vectors = np.empty((V_target.shape[0], K), dtype=np.complex128)
    overlap_probs: list[float] = []
    for k in range(K):
        cand_mask = np.array([i not in used for i in range(V_target.shape[1])])
        cand = V_target[:, cand_mask]
        rel = find_closest_vector_idx(target_vecs[:, k], cand)
        abs_idx = int(np.flatnonzero(cand_mask)[rel])
        used.add(abs_idx)
        current_vectors[:, k] = V_target[:, abs_idx]
        overlap_probs.append(float(np.abs(target_vecs[:, k].conj() @ V_target[:, abs_idx]) ** 2))

    # Step 2: walk back to E_init, tracking each column independently.
    # (Independent tracking is fine because eigenstates can be re-used across columns
    # — the columns represent distinct adiabatic branches that may legitimately
    # share an intermediate-step eigenvector if they cross.)
    fractions = np.linspace(1.0, 0.0, n_steps + 1)[1:]  # skip s=1 (already done)
    for s in fractions:
        E_step = E_init_arr + s * (E_target_arr - E_init_arr)
        H_step = H_func(E_step, B_arr)
        D_step, V_step = _scipy_eigh(H_step, driver="evr")
        del D_step
        new_vectors = np.empty_like(current_vectors)
        for k in range(K):
            idx = find_closest_vector_idx(current_vectors[:, k], V_step)
            v = V_step[:, idx]
            # Phase-align so the dot product is real-positive (continuous gauge)
            phase = np.vdot(current_vectors[:, k], v)
            if abs(phase) > 0:
                v = v * np.conj(phase) / abs(phase)
            new_vectors[:, k] = v
        current_vectors = new_vectors

    # Step 3: at E_init, look up which eigenvector of H(E_init) each column
    # corresponds to (purely diagnostic — the simulator uses current_vectors directly).
    H_init = H_func(E_init_arr, B_arr)
    _D_init, V_init = _scipy_eigh(H_init, driver="evr")
    init_indices: list[int] = []
    for k in range(K):
        idx = find_closest_vector_idx(current_vectors[:, k], V_init)
        init_indices.append(int(idx))

    return current_vectors, init_indices, overlap_probs


def j_manifold_indices(QN: npt.NDArray) -> dict[int, npt.NDArray[np.int_]]:
    """Return {J: indices in QN with that J value} for all J present."""
    out: dict[int, list[int]] = {}
    for i, bs in enumerate(QN):
        out.setdefault(int(bs.J), []).append(i)
    return {J: np.array(idx, dtype=np.int_) for J, idx in sorted(out.items())}
