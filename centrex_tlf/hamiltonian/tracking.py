"""Adiabatically tracked X and B levels versus electric field.

The fields are along z (`E` in V/cm, `B_z` in Gauss), so mF is conserved and
every calculation runs per mF block. Working per block has a second benefit:
it never has to resolve the eigenvectors of a nearly degenerate +/-mF pair,
which are ill defined at the `B = 1e-5 G` placeholder field (see AGENTS.md).

Each block is followed from zero field to the requested fields in small steps,
matching every new set of eigenvectors to the previous set with a Hungarian
assignment on the overlap matrix and keeping a continuous phase convention.
This keeps the zero-field labels right well past the fields where one-shot
overlap matching mislabels states.

Units
-----
Everything in this module is a frequency in **Hz** (not rad/s): the matrices
returned by `field_matrices_X` / `field_matrices_B` and the energies returned
by `track_levels`. Multiply by `2 * np.pi` once if you need angular units.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from centrex_tlf import states
from centrex_tlf.states import ElectronicState, QuantumSelector

from .basis_transformations import generate_transform_matrix
from .generate_hamiltonian import (
    generate_coupled_hamiltonian_B,
    generate_uncoupled_hamiltonian_X,
)

__all__ = [
    "FieldMatrices",
    "LevelParent",
    "TrackedBlock",
    "TrackedLevels",
    "field_matrices_B",
    "field_matrices_X",
    "track_levels",
]


# ---------------------------------------------------------------------------
# field matrices
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FieldMatrices:
    """A basis and its field-free, Stark and Zeeman matrices, all in **Hz**.

    The Hamiltonian at fields `E` (V/cm) and `B_z` (Gauss) along z is
    ``H0 + E * HSz + B_z * HZz`` in Hz. The arrays are read-only; they may be
    shared between calls through an in-process cache.

    Attributes
    ----------
    electronic_state
        X or B.
    basis
        Coupled basis states. X uses the coupled basis; B uses the parity basis
        (`P = -1, +1`, `Omega = 1`).
    H0
        Field-free Hamiltonian, Hz.
    HSz
        Stark matrix for E along z, Hz per V/cm.
    HZz
        Zeeman matrix for B along z, Hz per Gauss.
    """

    electronic_state: ElectronicState
    basis: tuple[states.CoupledBasisState, ...]
    H0: npt.NDArray[Any]
    HSz: npt.NDArray[Any]
    HZz: npt.NDArray[Any]
    Jmin: int
    Jmax: int

    def __len__(self) -> int:
        return len(self.basis)


def _readonly(M: npt.NDArray[Any]) -> npt.NDArray[Any]:
    M = np.ascontiguousarray(M)
    # Drop an identically zero imaginary part: the X and B matrices are real in
    # these bases, and real eigenproblems are both faster and better defined.
    if np.iscomplexobj(M) and not np.any(M.imag):
        M = np.ascontiguousarray(M.real)
    M.setflags(write=False)
    return M


@lru_cache(maxsize=8)
def field_matrices_X(Jmin: int, Jmax: int) -> FieldMatrices:
    """X-state basis and field-free, Stark and Zeeman matrices in Hz.

    The matrices are built in the uncoupled basis and transformed to the
    coupled basis `|J, F1, F, mF>`.
    """
    Js = list(range(int(Jmin), int(Jmax) + 1))
    qn_uncoupled = list(states.generate_uncoupled_states_ground(Js=Js))
    qn_coupled = list(states.generate_coupled_states_ground(Js=Js))
    H = generate_uncoupled_hamiltonian_X(qn_uncoupled)
    S = generate_transform_matrix(qn_uncoupled, qn_coupled)

    def to_coupled(M: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return _readonly(np.asarray(S.conj().T @ M @ S))

    return FieldMatrices(
        electronic_state=ElectronicState.X,
        basis=tuple(qn_coupled),
        H0=to_coupled(H.Hff),
        HSz=to_coupled(H.HSz),
        HZz=to_coupled(H.HZz),
        Jmin=int(Jmin),
        Jmax=int(Jmax),
    )


@lru_cache(maxsize=8)
def field_matrices_B(Jmin: int, Jmax: int) -> FieldMatrices:
    """B-state basis (parity basis) and field-free, Stark and Zeeman matrices in Hz."""
    selector = QuantumSelector(
        J=np.arange(int(Jmin), int(Jmax) + 1),
        F1=None,
        F=None,
        mF=None,
        electronic=ElectronicState.B,
        P=[-1, +1],
        Ω=1,
    )
    qn = list(states.generate_coupled_states_B(selector))
    H = generate_coupled_hamiltonian_B(qn)
    H0 = H.Hrot + H.H_mhf_Tl + H.H_mhf_F + H.H_LD + H.H_cp1_Tl + H.H_c_Tl
    return FieldMatrices(
        electronic_state=ElectronicState.B,
        basis=tuple(qn),
        H0=_readonly(np.asarray(H0)),
        HSz=_readonly(np.asarray(H.HSz)),
        HZz=_readonly(np.asarray(H.HZz)),
        Jmin=int(Jmin),
        Jmax=int(Jmax),
    )


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class LevelParent:
    """Zero-field parent of a tracked level.

    `P` is the parity of a B parent and None for X, where parity is `(-1)**J`.
    """

    electronic_state: ElectronicState = field(compare=False)
    J: int
    F1: float
    F: int
    mF: int
    P: int | None = None

    @property
    def key(self) -> tuple:
        """`(J, F1, F, mF)` for X, `(J, F1, F, mF, P)` for B."""
        if self.P is None:
            return (self.J, self.F1, self.F, self.mF)
        return (self.J, self.F1, self.F, self.mF, self.P)

    def __str__(self) -> str:
        from fractions import Fraction

        F1 = Fraction(self.F1).limit_denominator(2)
        txt = f"{self.electronic_state.name}(J={self.J}, F1={F1}, F={self.F}, mF={self.mF}"
        if self.P is not None:
            txt += f", P={self.P:+d}"
        return txt + ")"


def _parent_of(state: states.CoupledBasisState) -> LevelParent:
    electronic = state.electronic_state
    P = getattr(state, "P", None)
    if electronic is ElectronicState.X:
        P = None
    return LevelParent(
        electronic_state=electronic,
        J=int(state.J),
        F1=float(state.F1),
        F=int(state.F),
        mF=int(state.mF),
        P=None if P is None else int(P),
    )


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------
def _n_tracking_steps(E: float, B: float, max_step_V_cm: float, max_step_G: float) -> int:
    """Number of ramp steps needed to track both fields on from zero.

    Both fields have to be stepped: sizing the ramp on `E` alone leaves a large
    `B` (e.g. `E=0, B=500`) crossing its whole range in the floor of 8 steps,
    which can hand a level the wrong zero-field parent with no warning.
    """
    steps_E = int(np.ceil(abs(float(E)) / max(float(max_step_V_cm), 1e-12)))
    steps_B = int(np.ceil(abs(float(B)) / max(float(max_step_G), 1e-12)))
    return max(steps_E, steps_B, 8 if B else 1, 1)


class _BlockTracker:
    """Follow the eigenvectors of one mF block as the fields change."""

    def __init__(self, H0: np.ndarray, HS: np.ndarray, HZ: np.ndarray) -> None:
        self.H0, self.HS, self.HZ = H0, HS, HZ
        self.w0, self.V0 = eigh(H0)
        self.w, self.V = self.w0.copy(), self.V0.copy()
        self.E = 0.0
        self.B = 0.0

    def _step(self, E: float, B: float) -> None:
        w, V = eigh(self.H0 + E * self.HS + B * self.HZ)
        overlap = np.abs(self.V.conj().T @ V) ** 2
        rows, cols = linear_sum_assignment(-overlap)
        order = np.empty(len(cols), dtype=int)
        order[rows] = cols
        w, V = w[order], V[:, order]
        # continuous phase convention, so overlaps with earlier steps stay meaningful
        phase = np.sum(self.V.conj() * V, axis=0)
        nonzero = np.abs(phase) > 0
        phase[nonzero] /= np.abs(phase[nonzero])
        phase[~nonzero] = 1.0
        self.w, self.V = w, V * phase.conj()
        self.E, self.B = E, B

    def ramp(self, E: float, B: float, n_steps: int) -> None:
        """Move linearly from the current fields to `(E, B)` in `n_steps` steps."""
        E0, B0 = self.E, self.B
        for s in np.linspace(0.0, 1.0, int(n_steps) + 1)[1:]:
            self._step(E0 + s * (E - E0), B0 + s * (B - B0))
        self.E, self.B = float(E), float(B)

    @property
    def purity(self) -> np.ndarray:
        """|<zero-field eigenstate|dressed state>|^2 for each column."""
        return np.abs(np.sum(self.V0.conj() * self.V, axis=0)) ** 2


@dataclass(frozen=True, eq=False)
class TrackedBlock:
    """Tracked eigenvectors of one mF block.

    Column `k` of every eigenvector array is the level `level_indices[k]`.

    Attributes
    ----------
    basis_indices
        Rows of the block in `FieldMatrices.basis`.
    level_indices
        Global level index of each column.
    zero_field_energies
        Zero-field eigenvalues of the columns, Hz.
    zero_field_vectors
        Zero-field eigenvectors, shape `(n_block, n_block)`.
    vectors
        Eigenvectors at each requested field, shape `(n_E, n_block, n_block)`.
    """

    mF: int
    basis_indices: npt.NDArray[np.int_]
    level_indices: npt.NDArray[np.int_]
    zero_field_energies: npt.NDArray[np.floating]
    zero_field_vectors: npt.NDArray[Any]
    vectors: npt.NDArray[Any]


@dataclass(frozen=True, eq=False)
class TrackedLevels:
    """Levels of one manifold, tracked adiabatically from zero field.

    Attributes
    ----------
    matrices
        The `FieldMatrices` the levels were computed from.
    E_values
        Requested electric fields, V/cm, in the order given.
    B_z
        Magnetic field along z, Gauss.
    parents
        Zero-field parent of each level, sorted by `(J, F1, F, mF, P)`.
    energies_Hz
        Eigenvalue of each level at each field, Hz, shape `(n_E, n_levels)`.
        Absolute within the manifold; X and B do not share an energy origin.
    purity
        `|<zero-field parent eigenstate|dressed state>|^2`, shape
        `(n_E, n_levels)`. The parent eigenstate is the zero-field eigenvector
        (which already contains the small zero-field hyperfine mixing), not the
        bare basis state.
    blocks
        Eigenvectors per mF block.
    """

    matrices: FieldMatrices
    E_values: npt.NDArray[np.floating]
    B_z: float
    parents: tuple[LevelParent, ...]
    energies_Hz: npt.NDArray[np.floating]
    purity: npt.NDArray[np.floating]
    blocks: tuple[TrackedBlock, ...]
    _index: dict[tuple, int] = field(default_factory=dict, repr=False)

    def index(self, parent: LevelParent | Sequence[float]) -> int:
        """Level index of a parent, given as a `LevelParent` or its `key` tuple."""
        key = parent.key if isinstance(parent, LevelParent) else tuple(parent)
        try:
            return self._index[_normalize_key(key)]
        except KeyError:
            raise KeyError(f"no tracked level with parent {key}") from None

    def block(self, mF: int) -> TrackedBlock:
        for blk in self.blocks:
            if blk.mF == int(mF):
                return blk
        raise KeyError(f"no tracked block with mF={mF}")

    def vector(self, level: int, field_index: int) -> npt.NDArray[Any]:
        """Dressed eigenvector of `level` at `E_values[field_index]` in the full basis."""
        blk = self.block(self.parents[level].mF)
        col = int(np.flatnonzero(blk.level_indices == level)[0])
        out = np.zeros(len(self.matrices), dtype=blk.vectors.dtype)
        out[blk.basis_indices] = blk.vectors[field_index][:, col]
        return out


def _normalize_key(key: tuple) -> tuple:
    J, F1, F, mF = key[:4]
    out: tuple = (int(J), float(F1), int(F), int(mF))
    if len(key) > 4 and key[4] is not None:
        out += (int(key[4]),)
    return out


class _ManifoldTracker:
    """Follow every mF block of a manifold through a sequence of fields.

    Internal building block shared by `track_levels` and the transition field
    grids. Fields are visited in the order requested; the first move is a
    straight ramp from zero field to `(E_first, B_z)`, after which `E` is
    stepped at fixed `B_z`.
    """

    def __init__(
        self,
        matrices: FieldMatrices,
        *,
        B_z: float = 0.0,
        max_step_V_cm: float = 2.0,
        max_step_G: float = 1.0,
        mF: Iterable[int] | None = None,
    ) -> None:
        self.matrices = matrices
        self.B_z = float(B_z)
        self.max_step_V_cm = float(max_step_V_cm)
        self.max_step_G = float(max_step_G)
        self._started = False

        basis = matrices.basis
        mF_all = np.asarray([int(s.mF) for s in basis])
        wanted = sorted(set(mF_all.tolist()) if mF is None else {int(m) for m in mF})

        self.block_mF: list[int] = []
        self.block_basis: list[np.ndarray] = []
        self.trackers: list[_BlockTracker] = []
        block_parents: list[list[LevelParent]] = []
        for m in wanted:
            ix = np.flatnonzero(mF_all == m)
            if ix.size == 0:
                continue
            sub = np.ix_(ix, ix)
            tr = _BlockTracker(matrices.H0[sub], matrices.HSz[sub], matrices.HZz[sub])
            # zero-field label of each eigenvector: Hungarian on |<basis|eigvec>|^2
            rows, cols = linear_sum_assignment(-(np.abs(tr.V0) ** 2))
            labels: list[LevelParent | None] = [None] * ix.size
            for r, c in zip(rows, cols, strict=False):
                labels[int(c)] = _parent_of(basis[int(ix[r])])
            self.block_mF.append(m)
            self.block_basis.append(ix)
            self.trackers.append(tr)
            block_parents.append([lbl for lbl in labels if lbl is not None])

        # global level order: sorted by parent label
        flat = [(p, b, k) for b, ps in enumerate(block_parents) for k, p in enumerate(ps)]
        flat.sort(key=lambda t: _sort_key(t[0]))
        self.parents: tuple[LevelParent, ...] = tuple(p for p, _, _ in flat)
        self.block_levels = [np.empty(len(ps), dtype=int) for ps in block_parents]
        for level, (_, b, k) in enumerate(flat):
            self.block_levels[b][k] = level

    @property
    def n_levels(self) -> int:
        return len(self.parents)

    def advance(self, E: float) -> None:
        """Move all blocks to electric field `E` (V/cm) at the fixed `B_z`."""
        E = float(E)
        if not self._started:
            n = _n_tracking_steps(E, self.B_z, self.max_step_V_cm, self.max_step_G)
            for tr in self.trackers:
                tr.ramp(E, self.B_z, n)
            self._started = True
            return
        current = self.trackers[0].E if self.trackers else E
        n = int(np.ceil(abs(E - current) / max(self.max_step_V_cm, 1e-12)))
        if n == 0:
            return
        for tr in self.trackers:
            tr.ramp(E, self.B_z, n)

    def energies_zero_field(self) -> np.ndarray:
        out = np.empty(self.n_levels)
        for tr, lv in zip(self.trackers, self.block_levels, strict=True):
            out[lv] = tr.w0
        return out

    def energies(self) -> np.ndarray:
        out = np.empty(self.n_levels)
        for tr, lv in zip(self.trackers, self.block_levels, strict=False):
            out[lv] = tr.w
        return out

    def purity(self) -> np.ndarray:
        out = np.empty(self.n_levels)
        for tr, lv in zip(self.trackers, self.block_levels, strict=False):
            out[lv] = tr.purity
        return out


def _sort_key(p: LevelParent) -> tuple:
    return (p.J, p.F1, p.F, p.mF, 0 if p.P is None else p.P)


def track_levels(
    matrices: FieldMatrices,
    E_values: float | Sequence[float] | npt.NDArray[np.floating],
    B_z: float = 0.0,
    max_step_V_cm: float = 2.0,
    *,
    max_step_G: float = 1.0,
    mF: Iterable[int] | None = None,
) -> TrackedLevels:
    """Track every level of a manifold from zero field to the requested fields.

    Parameters
    ----------
    matrices
        From `field_matrices_X` or `field_matrices_B` (Hz).
    E_values
        Electric fields along z, V/cm. Any order; results come back in the
        order given. Internally the fields are visited in ascending order.
    B_z
        Magnetic field along z, Gauss. It is ramped on together with the first
        (smallest) field, then held fixed while `E` is stepped.
    max_step_V_cm, max_step_G
        Largest field step between successive diagonalizations. Smaller is
        safer near avoided crossings; a converged result is step-size
        independent.
    mF
        Optional subset of mF blocks to track (all blocks by default).

    Returns
    -------
    TrackedLevels
        Energies in Hz, eigenvectors per mF block, each level's zero-field
        parent and its purity.
    """
    E_req = np.atleast_1d(np.asarray(E_values, dtype=float))
    if E_req.ndim != 1:
        raise ValueError("E_values must be a scalar or 1-D")
    order = np.argsort(E_req, kind="stable")

    tracker = _ManifoldTracker(
        matrices,
        B_z=B_z,
        max_step_V_cm=max_step_V_cm,
        max_step_G=max_step_G,
        mF=mF,
    )
    nE = E_req.size
    energies = np.empty((nE, tracker.n_levels))
    purity = np.empty((nE, tracker.n_levels))
    vectors = [
        np.empty((nE, ix.size, ix.size), dtype=tr.V0.dtype)
        for ix, tr in zip(tracker.block_basis, tracker.trackers, strict=False)
    ]

    for i in order:
        tracker.advance(E_req[i])
        energies[i] = tracker.energies()
        purity[i] = tracker.purity()
        for b, tr in enumerate(tracker.trackers):
            vectors[b][i] = tr.V

    blocks = tuple(
        TrackedBlock(
            mF=m,
            basis_indices=ix,
            level_indices=lv,
            zero_field_energies=tr.w0,
            zero_field_vectors=tr.V0,
            vectors=V,
        )
        for m, ix, lv, tr, V in zip(
            tracker.block_mF,
            tracker.block_basis,
            tracker.block_levels,
            tracker.trackers,
            vectors,
            strict=False,
        )
    )
    index = {_normalize_key(p.key): i for i, p in enumerate(tracker.parents)}
    return TrackedLevels(
        matrices=matrices,
        E_values=E_req,
        B_z=float(B_z),
        parents=tracker.parents,
        energies_Hz=energies,
        purity=purity,
        blocks=blocks,
        _index=index,
    )
