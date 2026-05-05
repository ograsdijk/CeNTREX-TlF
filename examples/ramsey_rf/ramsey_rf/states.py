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


# Tiny B field added during adiabatic walks to lift the (mJ, m_Tl, m_F) →
# (-mJ, -m_Tl, -m_F) Kramers-style degeneracy of the static Stark Hamiltonian.
# At B=0, opposite-sign-m branches are exactly degenerate (time-reversal
# symmetry); numpy.linalg.eigh returns arbitrary linear combinations within
# the degenerate subspace, which breaks max-overlap branch tracking.
# Adding ~ μ_Tl·B = 0.12 Hz Zeeman splitting at B_z = 1e-4 G separates the
# branches without affecting the state composition meaningfully.
_DEFAULT_WALK_B_Z_GAUSS = 1.0e-4


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
    walk_B_z: float = _DEFAULT_WALK_B_Z_GAUSS,
) -> tuple[npt.NDArray[np.complex128], list[int], list[float]]:
    """Adiabatic-ancestor state preparation.

    Returns the dressed eigenstate at `E_init` (the simulator's starting field,
    typically LOW so dressed states are nearly bare) that adiabatically connects
    to the dressed state at `E_target` (typically HIGH, where dressed states
    are heavily Stark-mixed) with the supplied bare quantum numbers.

    Algorithm (always walks LOW → HIGH so bare-state matching anchors where it
    is reliable):

      1. Diagonalize H(E_init, B) and pick the eigenvector with max overlap to
         the bare target. At low field the dressed states are nearly bare so
         this is unambiguous (overlap ~0.99). This eigenvector IS Psi0. Note
         Psi0 is computed at the USER-SUPPLIED B (default zero) so the simulator
         sees the right physical initial state.
      2. Walk E linearly from E_init to E_target in `n_steps` interior steps,
         tracking each branch by max overlap with the previous step's
         eigenvector and phase-aligning. The walk uses `B + walk_B_z·ẑ` to
         lift the (mJ → -mJ) degeneracy that otherwise causes branch hopping
         at B=0 (see _DEFAULT_WALK_B_Z_GAUSS comment). Used only for the
         diagnostic returns: eigenvector indices and overlap probabilities at
         E_target.

    NOTE: prior to fix-2026-05-05, this function walked HIGH → LOW with the
    initial pick at E_target. That is unreliable when the polarized branch at
    high field is heavily Stark-mixed (~33% bare-J=1 weight at 30 kV/cm,
    Jmax=6) and the m → -m degeneracy puts the bare-J=1-mJ=-1 weight in an
    arbitrary 50/50 mix of mJ=+1 and mJ=-1 polarized branches. The OLD
    function returned a Psi0 with arbitrary mJ character. Always walk
    low → high; the low-E pick at B=0 has overlap ~0.99 to the right branch
    even with the residual degeneracy, since the bare target itself selects
    a definite mJ.

    Returns (Psi0 at E_init shape (N, K), eigenstate indices at E_target,
    overlap probabilities |<bare|tracked_eigvec>|^2 at E_target).
    """
    _states, target_vecs = targets_to_state_vectors(targets, QN)
    K = target_vecs.shape[1]
    E_init_arr = np.asarray(E_init, dtype=np.float64).reshape(3)
    E_target_arr = np.asarray(E_target, dtype=np.float64).reshape(3)
    B_user = np.asarray(B, dtype=np.float64).reshape(3)
    B_walk = B_user + np.array([0.0, 0.0, walk_B_z])

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    # Step 1: pick Psi0 at E_INIT under the user's B (default 0). Reliable
    # max-overlap-to-bare here even with the residual degeneracy because the
    # bare target itself selects a definite mJ.
    H_init = H_func(E_init_arr, B_user)
    _D_init, V_init = _scipy_eigh(H_init, driver="evr")

    used: set[int] = set()
    Psi0 = np.empty((V_init.shape[0], K), dtype=np.complex128)
    for k in range(K):
        cand_mask = np.array([i not in used for i in range(V_init.shape[1])])
        cand = V_init[:, cand_mask]
        rel = find_closest_vector_idx(target_vecs[:, k], cand)
        abs_idx = int(np.flatnonzero(cand_mask)[rel])
        used.add(abs_idx)
        Psi0[:, k] = V_init[:, abs_idx]

    # Step 2 (diagnostic): walk E_init → E_target with the symmetry-breaking
    # walk_B_z, starting from the eigvec at E_init under B_walk (= the closest
    # match to Psi0 at the slightly-perturbed B).
    H_init_walk = H_func(E_init_arr, B_walk)
    _D_init_walk, V_init_walk = _scipy_eigh(H_init_walk, driver="evr")
    used_walk: set[int] = set()
    current_vectors = np.empty_like(Psi0)
    for k in range(K):
        cand_mask = np.array([i not in used_walk for i in range(V_init_walk.shape[1])])
        cand = V_init_walk[:, cand_mask]
        rel = find_closest_vector_idx(Psi0[:, k], cand)
        abs_idx = int(np.flatnonzero(cand_mask)[rel])
        used_walk.add(abs_idx)
        current_vectors[:, k] = V_init_walk[:, abs_idx]

    fractions = np.linspace(0.0, 1.0, n_steps + 1)[1:]
    for s in fractions:
        E_step = E_init_arr + s * (E_target_arr - E_init_arr)
        H_step = H_func(E_step, B_walk)
        _D_step, V_step = _scipy_eigh(H_step, driver="evr")
        new_vectors = np.empty_like(current_vectors)
        for k in range(K):
            idx = find_closest_vector_idx(current_vectors[:, k], V_step)
            v = V_step[:, idx]
            phase = np.vdot(current_vectors[:, k], v)
            if abs(phase) > 0:
                v = v * np.conj(phase) / abs(phase)
            new_vectors[:, k] = v
        current_vectors = new_vectors

    # Step 3 diagnostics: indices and overlap with bare target at E_target.
    H_target = H_func(E_target_arr, B_walk)
    _D_target, V_target = _scipy_eigh(H_target, driver="evr")
    target_indices: list[int] = []
    overlap_probs_at_target: list[float] = []
    for k in range(K):
        idx = find_closest_vector_idx(current_vectors[:, k], V_target)
        target_indices.append(int(idx))
        overlap_probs_at_target.append(
            float(np.abs(target_vecs[:, k].conj() @ current_vectors[:, k]) ** 2)
        )

    return Psi0, target_indices, overlap_probs_at_target


def adiabatic_branch_to_field(
    H_func: Callable[..., npt.NDArray[np.complex128]],
    targets: TargetSpec,
    QN: npt.NDArray,
    *,
    E_anchor: Sequence[float] | npt.NDArray[np.float64] = (0.0, 0.0, 0.0),
    E_final: Sequence[float] | npt.NDArray[np.float64],
    n_steps: int = 80,
    B: Sequence[float] | npt.NDArray[np.float64] = (0.0, 0.0, 0.0),
    walk_B_z: float = _DEFAULT_WALK_B_Z_GAUSS,
) -> tuple[npt.NDArray[np.complex128], list[int], list[float]]:
    """Identify the dressed eigenstate at `E_final` that adiabatically connects
    to a bare state at the (low-field) `E_anchor`.

    Mirror of `adiabatic_dressed_initial_states` but walking ANCHOR → FINAL
    instead of TARGET → INIT. Use this when the bare state matches reliably
    at low/zero field but is heavily mixed at the final (high) field — the
    common case for polarized molecules at strong DC Stark fields, where
    `find_closest_vector_idx(bare_target, V_high)` is unreliable because the
    bare-state weight is spread across multiple polarized eigenstates.

    The function:
      1. Picks the eigenvector at E_anchor with maximum overlap to the target
         (reliable because the dressed states are nearly bare there).
      2. Walks E linearly from E_anchor to E_final in `n_steps` interior
         steps, picking the eigenvector with maximum overlap to the previous
         step's eigenvector at each step (continuous-branch tracking with
         phase-alignment). A tiny `walk_B_z` field is added during the walk
         to lift the (mJ → -mJ) degeneracy that otherwise causes branch
         hopping at B=0; default 1e-4 G is small enough to be physically
         negligible while large enough to separate eigenvalues numerically.
      3. The final eigenvector at E_final is the polarized branch.

    Returns (branches of shape (N, K) at E_final, indices into the eigenvalue
    ordering of H(E_final, B + walk_B_z·ẑ), overlap probabilities at E_anchor).
    """
    _states, target_vecs = targets_to_state_vectors(targets, QN)
    K = target_vecs.shape[1]
    E_anchor_arr = np.asarray(E_anchor, dtype=np.float64).reshape(3)
    E_final_arr = np.asarray(E_final, dtype=np.float64).reshape(3)
    B_walk = np.asarray(B, dtype=np.float64).reshape(3) + np.array([0.0, 0.0, walk_B_z])

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    # Step 1: pick eigenvectors at E_anchor (reliable max-overlap-to-bare).
    H_anchor = H_func(E_anchor_arr, B_walk)
    _D_anchor, V_anchor = _scipy_eigh(H_anchor, driver="evr")

    used: set[int] = set()
    current_vectors = np.empty((V_anchor.shape[0], K), dtype=np.complex128)
    overlap_probs: list[float] = []
    for k in range(K):
        cand_mask = np.array([i not in used for i in range(V_anchor.shape[1])])
        cand = V_anchor[:, cand_mask]
        rel = find_closest_vector_idx(target_vecs[:, k], cand)
        abs_idx = int(np.flatnonzero(cand_mask)[rel])
        used.add(abs_idx)
        current_vectors[:, k] = V_anchor[:, abs_idx]
        overlap_probs.append(
            float(np.abs(target_vecs[:, k].conj() @ V_anchor[:, abs_idx]) ** 2)
        )

    # Step 2: walk anchor → final, tracking each column independently.
    fractions = np.linspace(0.0, 1.0, n_steps + 1)[1:]  # skip s=0 (already done)
    for s in fractions:
        E_step = E_anchor_arr + s * (E_final_arr - E_anchor_arr)
        H_step = H_func(E_step, B_walk)
        _D_step, V_step = _scipy_eigh(H_step, driver="evr")
        new_vectors = np.empty_like(current_vectors)
        for k in range(K):
            idx = find_closest_vector_idx(current_vectors[:, k], V_step)
            v = V_step[:, idx]
            phase = np.vdot(current_vectors[:, k], v)
            if abs(phase) > 0:
                v = v * np.conj(phase) / abs(phase)
            new_vectors[:, k] = v
        current_vectors = new_vectors

    # Diagnostic: which eigenvector index of H(E_final) does each branch correspond to?
    H_final = H_func(E_final_arr, B_walk)
    _D_final, V_final = _scipy_eigh(H_final, driver="evr")
    final_indices: list[int] = []
    for k in range(K):
        idx = find_closest_vector_idx(current_vectors[:, k], V_final)
        final_indices.append(int(idx))

    return current_vectors, final_indices, overlap_probs


def j_manifold_indices(QN: npt.NDArray) -> dict[int, npt.NDArray[np.int_]]:
    """Return {J: indices in QN with that J value} for all J present."""
    out: dict[int, list[int]] = {}
    for i, bs in enumerate(QN):
        out.setdefault(int(bs.J), []).append(i)
    return {J: np.array(idx, dtype=np.int_) for J, idx in sorted(out.items())}
