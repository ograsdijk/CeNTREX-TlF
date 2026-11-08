"""Transition components on a grid of electric fields.

A `TransitionFieldGrid` stores, at a set of electric fields along z, the
frequency and dressed coupling strength of every component that connects a
tracked *lower* manifold to a tracked *upper* manifold. Values between the
stored fields come from PCHIP interpolation; the stored fields are chosen
adaptively so that interpolation stays within the build tolerances.

The core knows nothing about X -> B transitions: it takes two tracked manifolds
(`FieldMatrices`, possibly the same one), bare-basis operator matrices `D_q`
(q = -1, 0, +1), a frequency origin, a pair filter and a pair-labelling
function. `build_optical_field_grid` is the X -> B builder.

Physics notes
-------------
* E is along z, so mF is exact and a component with polarization index `q`
  always has `mF_upper - mF_lower = q`. Nothing else is assumed: every pair
  whose dressed `|d_q|^2` exceeds a small relative cutoff anywhere on the grid
  is kept, including nominally forbidden (Delta F, Delta J, parity) pairs that
  only become driveable through Stark mixing.
* `q` follows the package convention for `<upper| D . eps |lower>`
  (`hamiltonian.ED_ME_coupled`): `eps_{+1} = -(Ex + i Ey)/sqrt(2)`,
  `eps_{-1} = (Ex - i Ey)/sqrt(2)`, `eps_0 = Ez`, so
  the predefined sigma-plus polarization in `couplings` drives `q = +1`.
* Strengths are `|d_q|^2` in units of the operator matrices (for the optical
  builder, the dimensionless angular/reduced dipole factor used throughout the
  package). They are hints: real line intensities need populations and the
  optical Bloch equations (dark states, saturation).
* Frequencies are in Hz. Levels come from `hamiltonian.tracking`, so labels are
  zero-field parents found by adiabatic tracking, not by one-shot matching.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from centrex_tlf.states import ElectronicState

from .transition import OpticalTransition, OpticalTransitionType

__all__ = [
    "CrossingFlag",
    "TransitionComponent",
    "TransitionFieldGrid",
    "ZeroFieldComponents",
    "build_optical_field_grid",
    "optical_dipole_operators",
    "spherical_components",
]

FORMAT_VERSION = 1
Q_VALUES = (-1, 0, 1)

# Cartesian polarization vectors that give a unit spherical component for q in
# the package convention (see the module docstring).
_UNIT_POLARIZATION = {
    -1: np.array([1.0, 1.0j, 0.0], dtype=np.complex128) / np.sqrt(2),
    0: np.array([0.0, 0.0, 1.0], dtype=np.complex128),
    +1: np.array([-1.0, 1.0j, 0.0], dtype=np.complex128) / np.sqrt(2),
}


def _package_version() -> str:
    try:
        return metadata.version("centrex-tlf")
    except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
        return "unknown"


def spherical_components(polarization: Any, normalize: bool = True) -> dict[int, complex]:
    """Spherical components `eps_q` of a Cartesian polarization `[Ex, Ey, Ez]`.

    Accepts a `couplings.Polarization` or a 3-vector. Uses the package
    convention, in which `|eps_q|^2 |d_q|^2` is the squared matrix element
    `|generate_ED_ME_mixed_state(upper, lower, pol)|^2` of a component.
    """
    vec = np.asarray(getattr(polarization, "vector", polarization), dtype=np.complex128)
    if vec.shape != (3,):
        raise ValueError("polarization must be a 3-vector [Ex, Ey, Ez]")
    if normalize:
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("polarization vector is zero")
        vec = vec / norm
    Ex, Ey, Ez = vec
    return {
        +1: complex(-(Ex + 1j * Ey) / np.sqrt(2)),
        -1: complex((Ex - 1j * Ey) / np.sqrt(2)),
        0: complex(Ez),
    }


# ---------------------------------------------------------------------------
# result records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossingFlag:
    """A stored-field interval that needs care.

    `reason` is one of
    * ``"purity"``: some level's purity jumps by more than the threshold
      between consecutive fine tracking points (avoided crossing),
    * ``"strength"``: some component's strength changes fast between
      consecutive fine points,
    * ``"min_spacing"``: refinement reached the finest spacing and the last
      midpoint check there still failed.
    """

    E_low: float
    E_high: float
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class TransitionComponent:
    """One component at one field."""

    label: Hashable
    lower: Any  # LevelParent
    upper: Any  # LevelParent
    q: int
    E: float
    frequency_Hz: float
    strength: float
    lower_purity: float
    upper_purity: float


@dataclass(frozen=True)
class ZeroFieldComponents:
    """Candidate components at zero field, passed to a callable frequency origin.

    `energy_difference_Hz` is `E_upper - E_lower` without any origin.
    """

    lower: tuple[Any, ...]
    upper: tuple[Any, ...]
    q: npt.NDArray[np.int_]
    labels: tuple[Hashable, ...]
    energy_difference_Hz: npt.NDArray[np.floating]
    strength: npt.NDArray[np.floating]


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class TransitionFieldGrid:
    """Transition components stored on an adaptive grid of electric fields.

    Build with `TransitionFieldGrid.build` (generic) or
    `build_optical_field_grid`; reload with `TransitionFieldGrid.load`.

    Array attributes (`nS` stored fields, `nC` components, `nL`/`nU` levels):

    * `E_fields` `(nS,)`, V/cm
    * `lower_energies_Hz` `(nS, nL)`, `upper_energies_Hz` `(nS, nU)`
    * `lower_purity` `(nS, nL)`, `upper_purity` `(nS, nU)`
    * `component_lower`, `component_upper`, `component_q` `(nC,)`
    * `strengths` `(nS, nC)`, dressed `|d_q|^2`

    A component's frequency is
    ``frequency_origin_Hz + upper_energy - lower_energy``.
    """

    kind: str
    E_fields: npt.NDArray[np.floating]
    B_z: float
    frequency_origin_Hz: float
    lower_parents: tuple[Any, ...]
    upper_parents: tuple[Any, ...]
    lower_energies_Hz: npt.NDArray[np.floating]
    upper_energies_Hz: npt.NDArray[np.floating]
    lower_purity: npt.NDArray[np.floating]
    upper_purity: npt.NDArray[np.floating]
    component_lower: npt.NDArray[np.int_]
    component_upper: npt.NDArray[np.int_]
    component_q: npt.NDArray[np.int_]
    labels: tuple[Hashable, ...]
    strengths: npt.NDArray[np.floating]
    flags: tuple[CrossingFlag, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    package_version: str = field(default_factory=_package_version)
    _interp: dict[str, PchipInterpolator] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        lower: Any,
        upper: Any,
        operators: Mapping[int, npt.NDArray[Any]],
        *,
        E_max_V_cm: float,
        E_min_V_cm: float = 0.0,
        B_z: float = 0.0,
        frequency_origin_Hz: float | Callable[[ZeroFieldComponents], float] = 0.0,
        pair_filter: Callable[[Any, Any, int], bool] | None = None,
        pair_label: Callable[[Any, Any], Hashable] | None = None,
        kind: str = "generic",
        frequency_tol_Hz: float = 1e5,
        strength_rtol: float = 0.02,
        strength_floor: float = 1e-2,
        coupling_cutoff: float = 1e-6,
        base_spacing_V_cm: float = 25.0,
        min_spacing_V_cm: float | None = None,
        max_step_V_cm: float = 2.0,
        purity_jump: float = 0.05,
        strength_jump: float = 0.3,
        parameters: Mapping[str, Any] | None = None,
    ) -> TransitionFieldGrid:
        """Track both manifolds and build an adaptive component grid.

        Parameters
        ----------
        lower, upper
            `hamiltonian.FieldMatrices`; may be the same object (e.g. X -> X).
        operators
            `{q: D_q}` for q in (-1, 0, +1), bare-basis matrices of shape
            `(len(upper.basis), len(lower.basis))`, element `<upper|D_q|lower>`.
        E_min_V_cm, E_max_V_cm, B_z
            Field range (V/cm) and fixed magnetic field along z (Gauss).
        frequency_origin_Hz
            Added to every `E_upper - E_lower`. Either a number or a callable
            that receives the zero-field `ZeroFieldComponents` and returns it
            (used to pin a reference line).
        pair_filter
            `pair_filter(lower_parent, upper_parent, q) -> bool`; pairs it
            rejects are never considered. `mF_upper - mF_lower = q` is always
            enforced, and in a single manifold a level is never paired with
            itself.
        pair_label
            `pair_label(lower_parent, upper_parent) -> label`. Defaults to the
            tuple of the two parents.
        frequency_tol_Hz, strength_rtol, strength_floor
            Refinement tolerances. A midpoint fails if a frequency is off by
            more than `frequency_tol_Hz`, or a strength by more than
            `strength_rtol * max(strength, strength_floor * global_max)`.
        coupling_cutoff
            Keep a pair if its `|d_q|^2` exceeds `coupling_cutoff * global_max`
            at any fine field.
        base_spacing_V_cm, min_spacing_V_cm
            Initial stored spacing and the finest spacing refinement may reach
            (default `base / 16`). The exact reference data are computed on the
            finest grid.
        max_step_V_cm
            Largest tracking step, independent of the stored spacing.
        purity_jump, strength_jump
            Thresholds for `CrossingFlag` (per fine step; strength relative).
        """
        from centrex_tlf.hamiltonian.tracking import _ManifoldTracker

        E_min, E_max = float(E_min_V_cm), float(E_max_V_cm)
        if not E_max > E_min:
            raise ValueError("E_max_V_cm must be larger than E_min_V_cm")
        n_base = max(1, int(np.ceil((E_max - E_min) / float(base_spacing_V_cm) - 1e-9)))
        base = (E_max - E_min) / n_base
        min_spacing = base / 16 if min_spacing_V_cm is None else float(min_spacing_V_cm)
        levels = max(0, int(np.ceil(np.log2(base / min_spacing) - 1e-9)))
        n_fine = n_base * 2**levels + 1
        E_fine = np.linspace(E_min, E_max, n_fine)

        same = lower is upper
        tl = _ManifoldTracker(lower, B_z=B_z, max_step_V_cm=max_step_V_cm)
        tu = tl if same else _ManifoldTracker(upper, B_z=B_z, max_step_V_cm=max_step_V_cm)
        if pair_label is None:
            pair_label = _default_label

        # ---- candidate pairs, grouped by (q, lower block, upper block)
        groups = []  # (q, bl, bu, D_sub, rows, cols, pair_ids)
        pairs_l: list[int] = []
        pairs_u: list[int] = []
        pairs_q: list[int] = []
        for q in Q_VALUES:
            D = operators.get(q)
            if D is None:
                continue
            D = np.asarray(D)
            if D.shape != (len(upper.basis), len(lower.basis)):
                raise ValueError(f"operator q={q} has shape {D.shape}")
            for bl, m_l in enumerate(tl.block_mF):
                if (m_l + q) not in tu.block_mF:
                    continue
                bu = tu.block_mF.index(m_l + q)
                D_sub = D[np.ix_(tu.block_basis[bu], tl.block_basis[bl])]
                if not np.any(D_sub):
                    continue
                rows, cols, ids = [], [], []
                for c, lv_l in enumerate(tl.block_levels[bl]):
                    par_l = tl.parents[lv_l]
                    for r, lv_u in enumerate(tu.block_levels[bu]):
                        if same and lv_u == lv_l:
                            continue
                        par_u = tu.parents[lv_u]
                        if pair_filter is not None and not pair_filter(par_l, par_u, q):
                            continue
                        rows.append(r)
                        cols.append(c)
                        ids.append(len(pairs_q))
                        pairs_l.append(int(lv_l))
                        pairs_u.append(int(lv_u))
                        pairs_q.append(q)
                if ids:
                    groups.append(
                        (bl, bu, D_sub, np.asarray(rows), np.asarray(cols), np.asarray(ids))
                    )
        n_pairs = len(pairs_q)
        if n_pairs == 0:
            raise ValueError("no candidate pairs; check operators and pair_filter")
        pl = np.asarray(pairs_l)
        pu = np.asarray(pairs_u)
        pq = np.asarray(pairs_q)

        def dressed_strengths(Vl: list[np.ndarray], Vu: list[np.ndarray]) -> np.ndarray:
            out = np.empty(n_pairs)
            for bl, bu, D_sub, rows, cols, ids in groups:
                d = Vu[bu].conj().T @ D_sub @ Vl[bl]
                out[ids] = np.abs(d[rows, cols]) ** 2
            return out

        # ---- zero field (before any ramp) for the frequency origin
        zf_l = tl.energies_zero_field()
        zf_u = tu.energies_zero_field()
        s0 = dressed_strengths([t.V0 for t in tl.trackers], [t.V0 for t in tu.trackers])
        if callable(frequency_origin_Hz):
            zmask = s0 > 0
            zf = ZeroFieldComponents(
                lower=tuple(tl.parents[i] for i in pl[zmask]),
                upper=tuple(tu.parents[i] for i in pu[zmask]),
                q=pq[zmask],
                labels=tuple(
                    pair_label(tl.parents[i], tu.parents[j])
                    for i, j in zip(pl[zmask], pu[zmask], strict=True)
                ),
                energy_difference_Hz=(zf_u[pu] - zf_l[pl])[zmask],
                strength=s0[zmask],
            )
            origin = float(frequency_origin_Hz(zf))
        else:
            origin = float(frequency_origin_Hz)

        # ---- exact data on the fine grid
        nL, nU = tl.n_levels, tu.n_levels
        e_l = np.empty((n_fine, nL))
        e_u = np.empty((n_fine, nU))
        p_l = np.empty((n_fine, nL))
        p_u = np.empty((n_fine, nU))
        s_fine = np.empty((n_fine, n_pairs))
        for i, E in enumerate(E_fine):
            tl.advance(E)
            if not same:
                tu.advance(E)
            e_l[i], p_l[i] = tl.energies(), tl.purity()
            e_u[i], p_u[i] = tu.energies(), tu.purity()
            s_fine[i] = dressed_strengths([t.V for t in tl.trackers], [t.V for t in tu.trackers])

        # ---- keep pairs above the cutoff anywhere
        global_max = float(s_fine.max())
        keep = np.flatnonzero(s_fine.max(axis=0) > coupling_cutoff * global_max)
        s_fine = s_fine[:, keep]
        pl, pu, pq = pl[keep], pu[keep], pq[keep]
        used_l = np.unique(pl)
        used_u = np.unique(pu)
        cl = np.searchsorted(used_l, pl)
        cu = np.searchsorted(used_u, pu)
        e_l, p_l = e_l[:, used_l], p_l[:, used_l]
        e_u, p_u = e_u[:, used_u], p_u[:, used_u]
        freq_fine = e_u[:, cu] - e_l[:, cl]

        # ---- adaptive selection of stored fields
        stride = 2**levels
        stored = set(range(0, n_fine, stride))
        floor = strength_floor * global_max
        failed_at_min: set[tuple[int, int]] = set()
        while True:
            idx = np.asarray(sorted(stored))
            x = E_fine[idx]
            if idx.size < 2:
                break
            interp_l = PchipInterpolator(x, e_l[idx], axis=0)
            interp_u = PchipInterpolator(x, e_u[idx], axis=0)
            interp_s = PchipInterpolator(x, s_fine[idx], axis=0)
            a, b = idx[:-1], idx[1:]
            check = (b - a) >= 2
            mids = (a[check] + b[check]) // 2
            if mids.size == 0:
                break
            Em = E_fine[mids]
            f_int = interp_u(Em)[:, cu] - interp_l(Em)[:, cl]
            f_err = np.abs(f_int - freq_fine[mids]).max(axis=1)
            s_ex = s_fine[mids]
            s_err = np.abs(interp_s(Em) - s_ex) / np.maximum(s_ex, floor)
            bad = (f_err > frequency_tol_Hz) | (s_err.max(axis=1) > strength_rtol)
            if not np.any(bad):
                break
            for lo, hi, m in zip(a[check][bad], b[check][bad], mids[bad], strict=True):
                stored.add(int(m))
                if hi - lo == 2:
                    failed_at_min.add((int(lo), int(hi)))

        idx = np.asarray(sorted(stored))

        # ---- flags (adjacent intervals with the same reason are merged)
        raw: list[tuple[int, int, str, float]] = []
        for lo, hi in pairwise(idx):
            seg = slice(int(lo), int(hi) + 1)
            dp = max(
                np.abs(np.diff(p_l[seg], axis=0)).max(initial=0.0),
                np.abs(np.diff(p_u[seg], axis=0)).max(initial=0.0),
            )
            if dp > purity_jump:
                raw.append((int(lo), int(hi), "purity", float(dp)))
            s_seg = s_fine[seg]
            big = s_seg.max(axis=0) > floor
            if np.any(big):
                ds = np.abs(np.diff(s_seg[:, big], axis=0)) / np.maximum(
                    np.maximum(s_seg[:-1, big], s_seg[1:, big]), floor
                )
                rel = float(ds.max(initial=0.0))
                if rel > strength_jump:
                    raw.append((int(lo), int(hi), "strength", rel))
        for lo, hi in failed_at_min:
            raw.append((lo, hi, "min_spacing", 0.0))
        flags = _merge_flags(raw, E_fine)

        lower_parents = tuple(tl.parents[i] for i in used_l)
        upper_parents = tuple(tu.parents[i] for i in used_u)
        labels = tuple(
            pair_label(lower_parents[i], upper_parents[j]) for i, j in zip(cl, cu, strict=True)
        )
        params = dict(parameters or {})
        params.update(
            kind=kind,
            lower=[lower.electronic_state.name, lower.Jmin, lower.Jmax],
            upper=[upper.electronic_state.name, upper.Jmin, upper.Jmax],
            same_manifold=same,
            E_min_V_cm=E_min,
            E_max_V_cm=E_max,
            B_z=float(B_z),
            frequency_tol_Hz=float(frequency_tol_Hz),
            strength_rtol=float(strength_rtol),
            strength_floor=float(strength_floor),
            coupling_cutoff=float(coupling_cutoff),
            base_spacing_V_cm=float(base_spacing_V_cm),
            min_spacing_V_cm=float(min_spacing),
            max_step_V_cm=float(max_step_V_cm),
        )
        return cls(
            kind=kind,
            E_fields=E_fine[idx].copy(),
            B_z=float(B_z),
            frequency_origin_Hz=origin,
            lower_parents=lower_parents,
            upper_parents=upper_parents,
            lower_energies_Hz=e_l[idx].copy(),
            upper_energies_Hz=e_u[idx].copy(),
            lower_purity=p_l[idx].copy(),
            upper_purity=p_u[idx].copy(),
            component_lower=cl,
            component_upper=cu,
            component_q=pq,
            labels=labels,
            strengths=s_fine[idx].copy(),
            flags=flags,
            parameters=params,
        )

    # ---------------------------------------------------------------- queries
    @property
    def n_components(self) -> int:
        return int(self.component_q.size)

    def frequencies_at_stored_fields(self) -> npt.NDArray[np.floating]:
        """Exact component frequencies at `E_fields`, Hz, shape `(nS, nC)`."""
        return (
            self.frequency_origin_Hz
            + self.upper_energies_Hz[:, self.component_upper]
            - self.lower_energies_Hz[:, self.component_lower]
        )

    def _interpolator(self, name: str) -> PchipInterpolator:
        if name not in self._interp:
            self._interp[name] = PchipInterpolator(self.E_fields, getattr(self, name), axis=0)
        return self._interp[name]

    def _check_E(self, E: float) -> float:
        E = float(E)
        lo, hi = float(self.E_fields[0]), float(self.E_fields[-1])
        if not (lo - 1e-9 <= E <= hi + 1e-9):
            raise ValueError(f"E = {E} V/cm is outside the grid [{lo}, {hi}] V/cm")
        return min(max(E, lo), hi)

    def _values(self, E: float) -> dict[str, np.ndarray]:
        E = self._check_E(E)
        e_l = self._interpolator("lower_energies_Hz")(E)
        e_u = self._interpolator("upper_energies_Hz")(E)
        pur_l = self._interpolator("lower_purity")(E)
        pur_u = self._interpolator("upper_purity")(E)
        return {
            "frequency_Hz": self.frequency_origin_Hz
            + e_u[self.component_upper]
            - e_l[self.component_lower],
            "strength": np.maximum(self._interpolator("strengths")(E), 0.0),
            "lower_purity": pur_l[self.component_lower],
            "upper_purity": pur_u[self.component_upper],
        }

    def level_energies(self, E: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated `(lower, upper)` level energies at `E`, Hz."""
        E = self._check_E(E)
        return (
            self._interpolator("lower_energies_Hz")(E),
            self._interpolator("upper_energies_Hz")(E),
        )

    def at(self, E: float) -> Any:
        """Every component at field `E` (V/cm), interpolated, as a DataFrame.

        One row per component, in grid order, with the label, both parents'
        quantum numbers, `q`, `frequency_Hz`, `strength` (`|d_q|^2`) and both
        purities.
        """
        import pandas as pd

        v = self._values(E)
        low = [self.lower_parents[i] for i in self.component_lower]
        up = [self.upper_parents[i] for i in self.component_upper]
        return pd.DataFrame(
            {
                "label": [_label_name(lbl) for lbl in self.labels],
                "J": [p.J for p in low],
                "F1": [p.F1 for p in low],
                "F": [p.F for p in low],
                "mF": [p.mF for p in low],
                "P": [p.P for p in low],
                "J'": [p.J for p in up],
                "F1'": [p.F1 for p in up],
                "F'": [p.F for p in up],
                "mF'": [p.mF for p in up],
                "P'": [p.P for p in up],
                "q": self.component_q,
                "frequency_Hz": v["frequency_Hz"],
                "strength": v["strength"],
                "lower_purity": v["lower_purity"],
                "upper_purity": v["upper_purity"],
            }
        )

    def components(self, label: Hashable, E: float) -> list[TransitionComponent]:
        """Components carrying `label` (e.g. an `OpticalTransition`) at `E`.

        For optical grids the label is the line, so both parent parities of
        the upper level are returned; the parity-allowed ones have
        `upper.P == label.P_excited`.
        """
        v = self._values(E)
        out = []
        for k, lbl in enumerate(self.labels):
            if lbl != label:
                continue
            out.append(
                TransitionComponent(
                    label=lbl,
                    lower=self.lower_parents[self.component_lower[k]],
                    upper=self.upper_parents[self.component_upper[k]],
                    q=int(self.component_q[k]),
                    E=float(E),
                    frequency_Hz=float(v["frequency_Hz"][k]),
                    strength=float(v["strength"][k]),
                    lower_purity=float(v["lower_purity"][k]),
                    upper_purity=float(v["upper_purity"][k]),
                )
            )
        return out

    def strength(
        self,
        polarization: Any,
        E: float,
        ground_population: Mapping[Any, float] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Population-weighted strength of every component for a polarization.

        Returns `w(lower) * |eps_q|^2 * |d_q|^2` per component (array aligned
        with `at(E)`); summing over the components of a line gives that line's
        strength. `ground_population` maps a lower level (`LevelParent` or its
        `key` tuple) to a weight; levels it does not name get 0. The default is
        equal population (weight 1 for every lower level).
        """
        eps = spherical_components(polarization)
        pol_w = np.array([abs(eps[int(q)]) ** 2 for q in self.component_q])
        weights = self._lower_weights(ground_population)
        return weights[self.component_lower] * pol_w * self._values(E)["strength"]

    def _lower_weights(self, population: Mapping[Any, float] | None) -> np.ndarray:
        n = len(self.lower_parents)
        if population is None:
            return np.ones(n)
        by_key = {}
        for k, w in population.items():
            key = k.key if hasattr(k, "key") else tuple(k)
            by_key[_norm_key(key)] = float(w)
        return np.array([by_key.get(_norm_key(p.key), 0.0) for p in self.lower_parents])

    # ------------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> Path:
        """Write the grid to a compressed `.npz` with JSON metadata."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        meta = {
            "format_version": FORMAT_VERSION,
            "package_version": self.package_version,
            "kind": self.kind,
            "B_z": self.B_z,
            "frequency_origin_Hz": self.frequency_origin_Hz,
            "parameters": self.parameters,
            "label_names": [_label_name(lbl) for lbl in self.labels],
            "flags": [[f.E_low, f.E_high, f.reason, f.detail] for f in self.flags],
        }
        np.savez_compressed(
            path,
            meta=np.asarray(json.dumps(meta)),
            E_fields=self.E_fields,
            lower_parents=_parents_to_array(self.lower_parents),
            upper_parents=_parents_to_array(self.upper_parents),
            lower_electronic=np.asarray([p.electronic_state.name for p in self.lower_parents]),
            upper_electronic=np.asarray([p.electronic_state.name for p in self.upper_parents]),
            lower_energies_Hz=self.lower_energies_Hz,
            upper_energies_Hz=self.upper_energies_Hz,
            lower_purity=self.lower_purity,
            upper_purity=self.upper_purity,
            component_lower=self.component_lower,
            component_upper=self.component_upper,
            component_q=self.component_q,
            strengths=self.strengths,
        )
        return path

    @classmethod
    def load(
        cls, path: str | Path, *, expected: Mapping[str, Any] | None = None
    ) -> TransitionFieldGrid:
        """Read a grid written by `save`.

        Refuses (``ValueError``) a file written by a different package version
        or file format, since the Hamiltonians may have changed (see
        AGENTS.md on stale cached Hamiltonians), and any file whose build
        parameters differ from `expected`.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"{path}: file format {meta.get('format_version')} is not "
                    f"{FORMAT_VERSION}; rebuild the grid"
                )
            current = _package_version()
            if meta.get("package_version") != current:
                raise ValueError(
                    f"{path}: built with centrex-tlf {meta.get('package_version')}, "
                    f"this is {current}; rebuild the grid"
                )
            params = meta["parameters"]
            for key, want in (expected or {}).items():
                have = params.get(key)
                if _jsonable(want) != have:
                    raise ValueError(f"{path}: parameter {key!r} is {have!r}, expected {want!r}")
            lower_parents = _parents_from_array(data["lower_parents"], data["lower_electronic"])
            upper_parents = _parents_from_array(data["upper_parents"], data["upper_electronic"])
            cl = data["component_lower"]
            cu = data["component_upper"]
            label_fn = _LABELLERS.get(meta["kind"])
            if label_fn is None:
                labels = tuple(meta["label_names"])
            else:
                labels = tuple(
                    label_fn(lower_parents[i], upper_parents[j])
                    for i, j in zip(cl, cu, strict=True)
                )
            return cls(
                kind=meta["kind"],
                E_fields=data["E_fields"],
                B_z=float(meta["B_z"]),
                frequency_origin_Hz=float(meta["frequency_origin_Hz"]),
                lower_parents=lower_parents,
                upper_parents=upper_parents,
                lower_energies_Hz=data["lower_energies_Hz"],
                upper_energies_Hz=data["upper_energies_Hz"],
                lower_purity=data["lower_purity"],
                upper_purity=data["upper_purity"],
                component_lower=cl,
                component_upper=cu,
                component_q=data["component_q"],
                labels=labels,
                strengths=data["strengths"],
                flags=tuple(CrossingFlag(*f) for f in meta["flags"]),
                parameters=params,
                package_version=meta["package_version"],
            )


def _merge_flags(
    raw: Sequence[tuple[int, int, str, float]], E_fine: np.ndarray
) -> tuple[CrossingFlag, ...]:
    describe = {
        "purity": "max purity jump per fine step {:.3f}",
        "strength": "max relative strength change per fine step {:.3f}",
        "min_spacing": "midpoint check still failed at the finest spacing",
    }
    out: list[CrossingFlag] = []
    for reason in ("purity", "strength", "min_spacing"):
        spans = sorted((lo, hi, v) for lo, hi, r, v in raw if r == reason)
        merged: list[list[float]] = []
        for lo, hi, v in spans:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
                merged[-1][2] = max(merged[-1][2], v)
            else:
                merged.append([lo, hi, v])
        out += [
            CrossingFlag(
                float(E_fine[int(lo)]),
                float(E_fine[int(hi)]),
                reason,
                describe[reason].format(v),
            )
            for lo, hi, v in merged
        ]
    return tuple(sorted(out, key=lambda f: (f.E_low, f.reason)))


def _label_name(label: Hashable) -> str:
    return str(getattr(label, "name", label))


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _norm_key(key: tuple) -> tuple:
    J, F1, F, mF = key[:4]
    out: tuple = (int(J), float(F1), int(F), int(mF))
    if len(key) > 4 and key[4] is not None:
        out += (int(key[4]),)
    return out


def _default_label(lower: Any, upper: Any) -> Hashable:
    return (lower.key, upper.key)


def _parents_to_array(parents: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [[p.J, p.F1, p.F, p.mF, np.nan if p.P is None else p.P] for p in parents],
        dtype=float,
    ).reshape(-1, 5)


def _parents_from_array(arr: np.ndarray, electronic: np.ndarray) -> tuple[Any, ...]:
    from centrex_tlf.hamiltonian.tracking import LevelParent

    return tuple(
        LevelParent(
            electronic_state=ElectronicState[str(name)],
            J=int(row[0]),
            F1=float(row[1]),
            F=int(row[2]),
            mF=int(row[3]),
            P=None if np.isnan(row[4]) else int(row[4]),
        )
        for row, name in zip(arr, electronic, strict=True)
    )


# ---------------------------------------------------------------------------
# optical X -> B builder
# ---------------------------------------------------------------------------
def _optical_label(lower: Any, upper: Any) -> OpticalTransition:
    return OpticalTransition(
        t=OpticalTransitionType(int(upper.J) - int(lower.J)),
        J_ground=int(lower.J),
        F1_excited=float(upper.F1),
        F_excited=int(upper.F),
    )


_LABELLERS: dict[str, Callable[[Any, Any], Hashable]] = {"optical": _optical_label}


def optical_dipole_operators(lower: Any, upper: Any) -> dict[int, np.ndarray]:
    """Bare X -> B electric-dipole matrices `D_q[upper, lower]`.

    `lower` is an X `FieldMatrices`, `upper` a B one in the parity basis. Each
    parity basis state is converted to the Omega basis (`to_omega_basis`)
    and the elements come from `ED_ME_coupled(excited, ground)`, exactly as in
    `generate_ED_ME_mixed_state`, with a polarization of unit spherical
    component `q`.
    """
    from centrex_tlf.hamiltonian.matrix_elements_electric_dipole import (
        ED_ME_coupled,
        to_omega_basis,
    )

    X = lower.basis
    B = upper.basis
    D = {q: np.zeros((len(B), len(X)), dtype=np.complex128) for q in Q_VALUES}
    pols = {q: tuple(_UNIT_POLARIZATION[q]) for q in Q_VALUES}
    X_by_mF: dict[int, list[int]] = {}
    for j, x in enumerate(X):
        X_by_mF.setdefault(int(x.mF), []).append(j)

    for i, b in enumerate(B):
        omega = to_omega_basis(1 * b)
        for q in Q_VALUES:
            for j in X_by_mF.get(int(b.mF) - q, []):
                x = X[j]
                if abs(int(x.J) - int(b.J)) > 1 or abs(int(x.F) - int(b.F)) > 1:
                    continue
                me = 0j
                for amp, basis_b in omega.data:
                    me += np.conj(amp) * ED_ME_coupled(basis_b, x, pol_vec=pols[q])
                D[q][i, j] = me
    return D


def _r2_reference_origin(reference_Hz: float) -> Callable[[ZeroFieldComponents], float]:
    from .definitions import R2_F1_7o2_F3

    def origin(zf: ZeroFieldComponents) -> float:
        sel = np.asarray(
            [
                lbl == R2_F1_7o2_F3 and R2_F1_7o2_F3.P_excited == up.P
                for lbl, up in zip(zf.labels, zf.upper, strict=True)
            ]
        )
        if not np.any(sel):
            raise ValueError("R(2) F1'=7/2 F'=3 is not in the grid; need J_ground_max >= 2")
        w = zf.strength[sel]
        centroid = float(np.sum(w * zf.energy_difference_Hz[sel]) / np.sum(w))
        return reference_Hz - centroid

    return origin


def build_optical_field_grid(
    J_ground_max: int,
    E_max_V_cm: float,
    *,
    E_min_V_cm: float = 0.0,
    B_z: float = 0.0,
    frequency_tol_Hz: float = 1e5,
    reference_frequency_Hz: float | None = None,
    **kwargs: Any,
) -> TransitionFieldGrid:
    """X -> B transition components for ground `J <= J_ground_max`.

    Lower manifold: X, `J = 0 ... J_ground_max + 2`. Upper manifold: B in the
    parity basis, `J = 1 ... J_ground_max + 3`. Components are labelled by
    their `OpticalTransition` line (branch O to S from the parents' Delta J);
    pairs with `|Delta J| > 2` have no line label and are dropped.

    Frequencies are absolute UV frequencies in Hz. X and B do not share an
    energy origin, so the grid is shifted to put the zero-field R(2)
    F1'=7/2 F'=3 line at `reference_frequency_Hz` (default
    `constants.X_B_REFERENCE_FREQUENCY_HZ`, 1103.435082708 THz). Because the
    X J=2 hyperfine structure is unresolved in that line, "the line" means the
    strength-weighted centroid of its parity-allowed zero-field components
    (equal ground population, summed over q).

    Remaining keyword arguments go to `TransitionFieldGrid.build`.
    """
    from centrex_tlf.constants import X_B_REFERENCE_FREQUENCY_HZ
    from centrex_tlf.hamiltonian.tracking import field_matrices_B, field_matrices_X

    J_ground_max = int(J_ground_max)
    lower = field_matrices_X(0, J_ground_max + 2)
    upper = field_matrices_B(1, J_ground_max + 3)
    operators = optical_dipole_operators(lower, upper)
    ref = X_B_REFERENCE_FREQUENCY_HZ if reference_frequency_Hz is None else reference_frequency_Hz

    def pair_filter(lo: Any, up: Any, q: int) -> bool:
        return J_ground_max >= lo.J and abs(up.J - lo.J) <= 2 and up.J >= 1

    params = dict(kwargs.pop("parameters", {}) or {})
    params.update(J_ground_max=J_ground_max, reference_frequency_Hz=float(ref))
    return TransitionFieldGrid.build(
        lower,
        upper,
        operators,
        E_max_V_cm=E_max_V_cm,
        E_min_V_cm=E_min_V_cm,
        B_z=B_z,
        frequency_origin_Hz=_r2_reference_origin(float(ref)),
        pair_filter=pair_filter,
        pair_label=_optical_label,
        kind="optical",
        frequency_tol_Hz=frequency_tol_Hz,
        parameters=params,
        **kwargs,
    )
