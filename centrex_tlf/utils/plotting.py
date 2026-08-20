from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from typing import Any, Mapping, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from centrex_tlf.hamiltonian import (
    generate_coupled_hamiltonian_B,
    generate_transform_matrix,
    generate_uncoupled_hamiltonian_X,
)
from centrex_tlf.states import (
    ElectronicState,
    QuantumSelector,
    generate_coupled_states_B,
    generate_coupled_states_ground,
    generate_coupled_states_X,
    generate_uncoupled_states_ground,
)
from centrex_tlf.transitions import OpticalTransition, OpticalTransitionType

__all__ = [
    "plot_level_diagram",
    "filter_levels_with_decay_or_coupling",
    "combine_decay_only_states",
    "DressedLevel",
    "TransitionLevelStructure",
    "TransitionLevelDiagram",
    "calculate_transition_level_structure",
    "plot_transition_level_diagram",
]


def filter_levels_with_decay_or_coupling(
    states: Sequence[Any],
    coupling_mats: Sequence[np.ndarray] | None = None,
    branching_ratio: np.ndarray | None = None,
    *,
    coupling_threshold: float = 0.0,
    decay_threshold: float = 0.0,
    br_is_final_initial: bool = True,
    only_for_J: Sequence[float] | None = None,
) -> tuple[list[Any], list[np.ndarray] | None, np.ndarray | None, np.ndarray]:
    """
    Remove levels (states) that have neither couplings nor decays above threshold.

    A state is kept if:
      - it has |coupling| > coupling_threshold with any other state, OR
      - it participates in a decay with BR > decay_threshold, OR
      - its J is NOT in `only_for_J` (if specified)

    Parameters
    ----------
    states
        List of CoupledBasisState objects.
    coupling_mats
        List of coupling matrices M[i,j].
    branching_ratio
        Branching-ratio matrix.
    coupling_threshold
        Minimum |M[i,j]| to count as a coupling.
    decay_threshold
        Minimum BR to count as a decay.
    br_is_final_initial
        If True, BR[final, initial] corresponds to decay initial → final.
        If False, BR[initial, final] corresponds to decay initial → final.
    only_for_J
        If provided, filtering is applied ONLY to states whose J is in this list.
        All other states are always kept.

    Returns
    -------
    states_kept
    coupling_mats_kept
    branching_ratio_kept
    kept_indices
        Indices into the original arrays.
    """
    n = len(states)

    # Which states are subject to filtering
    if only_for_J is None:
        filter_mask = np.ones(n, dtype=bool)
    else:
        Jset = {float(J) for J in only_for_J}
        filter_mask = np.array([float(st.J) in Jset for st in states], dtype=bool)

    coupling_mats_arr: list[np.ndarray] | None = None
    if coupling_mats is not None:
        coupling_mats_arr = []
        for k, M in enumerate(coupling_mats):
            A = np.asarray(M)
            if A.shape != (n, n):
                raise ValueError(
                    f"coupling_mats[{k}] has shape {A.shape}, expected ({n},{n})"
                )
            coupling_mats_arr.append(A)

    BR = None
    if branching_ratio is not None:
        BR = np.asarray(branching_ratio, dtype=float)
        if BR.shape != (n, n):
            raise ValueError(
                f"`branching_ratio` has shape {BR.shape}, expected ({n},{n})"
            )

    keep = np.zeros(n, dtype=bool)

    # States not subject to filtering are always kept
    keep |= ~filter_mask

    coupled, decays = _compute_involvement(
        n,
        coupling_mats_arr,
        BR,
        coupling_threshold=coupling_threshold,
        decay_threshold=decay_threshold,
        br_is_final_initial=br_is_final_initial,
    )
    keep |= (coupled | decays) & filter_mask

    kept_idx = np.nonzero(keep)[0]

    states_kept = [states[i] for i in kept_idx]

    coupling_kept = None
    if coupling_mats_arr is not None:
        coupling_kept = [
            np.asarray(M)[np.ix_(kept_idx, kept_idx)] for M in coupling_mats_arr
        ]

    br_kept = None
    if BR is not None:
        br_kept = BR[np.ix_(kept_idx, kept_idx)]

    return states_kept, coupling_kept, br_kept, kept_idx


@dataclass(frozen=True)
class _CombinedState:
    electronic_state: ElectronicState
    J: float
    F1: None = None
    F: None = None
    mF: float = 0.0
    is_combined: bool = True


# ---------------- helper utilities (shared) ----------------
def f_maybe(v: Any) -> float | None:
    return float(v) if v is not None else None


def as_frac2(v: float) -> str:
    fr = Fraction(v).limit_denominator(2)
    if fr.denominator == 1:
        return f"{fr.numerator}"
    return f"{fr.numerator}/{fr.denominator}"


def as_signed_frac2(v: float) -> str:
    fr = Fraction(v).limit_denominator(2)
    if fr.denominator == 1:
        return f"{fr.numerator:+d}"
    num = fr.numerator
    sign = "+" if num >= 0 else "-"
    return f"{sign}{abs(num)}/{fr.denominator}"


def sort_none_last(vals: Sequence[float | None]) -> list[float | None]:
    return sorted(vals, key=lambda x: (x is None, x if x is not None else 0.0))


def j_in_list(Jval: float, Jlist: Sequence[float], tol: float) -> bool:
    return any(abs(Jval - float(Jx)) <= tol for Jx in Jlist)


def _compute_involvement(
    n: int,
    coupling_mats: Sequence[np.ndarray] | None,
    branching_ratio: np.ndarray | None,
    *,
    coupling_threshold: float,
    decay_threshold: float,
    br_is_final_initial: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean arrays (coupled, decays) of length n.

    `coupled[i]` is True if state i has any |coupling| > coupling_threshold.
    `decays[i]` is True if state i participates in any decay above decay_threshold.
    """

    coupled = np.zeros(n, dtype=bool)
    if coupling_mats is not None:
        for M in coupling_mats:
            A = np.abs(np.asarray(M))
            mask = A > coupling_threshold
            np.fill_diagonal(mask, False)
            coupled |= mask.any(axis=0) | mask.any(axis=1)

    decays = np.zeros(n, dtype=bool)
    if branching_ratio is not None:
        mask = np.asarray(branching_ratio, dtype=float) > decay_threshold
        np.fill_diagonal(mask, False)
        if br_is_final_initial:
            # BR[final, initial]
            initial_involved = mask.any(axis=0)
            final_involved = mask.any(axis=1)
        else:
            # BR[initial, final]
            initial_involved = mask.any(axis=1)
            final_involved = mask.any(axis=0)
        decays |= initial_involved | final_involved

    return coupled, decays


# ---------------- combine / collapse helper ----------------
def combine_decay_only_states(
    states: Sequence[Any],
    coupling_mats: Sequence[np.ndarray] | None = None,
    branching_ratio: np.ndarray | None = None,
    *,
    combine_for_J: Sequence[float] | None = None,
    combine_for_electronic: Sequence[ElectronicState] | None = None,
    drop_isolated_when_combining: bool = True,
    drop_isolated_for_J: Sequence[float] | None = None,
    coupling_threshold: float = 0.0,
    decay_threshold: float = 0.0,
    br_is_final_initial: bool = True,
    j_match_tol: float = 1e-9,
) -> tuple[list[Any], list[np.ndarray] | None, np.ndarray | None]:
    """Return (states_out, coupling_out, BR_out) after optionally dropping isolated
    and collapsing groups of decay-only states per (electronic_state, J).

    This extracts the previous inlined logic so callers can perform the collapsing
    once before calling `plot_level_diagram`.
    """
    n_pre = len(states)

    coupling_mats_arr: list[np.ndarray] | None = None
    if coupling_mats is not None:
        coupling_mats_arr = []
        for k, M in enumerate(coupling_mats):
            A = np.asarray(M)
            if A.shape != (n_pre, n_pre):
                raise ValueError(
                    f"coupling_mats[{k}] has shape {A.shape}, expected ({n_pre},{n_pre})"
                )
            coupling_mats_arr.append(A)

    BR0 = None
    if branching_ratio is not None:
        BR0 = np.asarray(branching_ratio, dtype=float)
        if BR0.shape != (n_pre, n_pre):
            raise ValueError(
                f"`branching_ratio` has shape {BR0.shape}, expected ({n_pre},{n_pre})"
            )

    # ---- optionally drop isolated before combining ----
    if drop_isolated_when_combining:
        coupled_pre, decays_pre = _compute_involvement(
            n_pre,
            coupling_mats_arr,
            BR0,
            coupling_threshold=coupling_threshold,
            decay_threshold=decay_threshold,
            br_is_final_initial=br_is_final_initial,
        )
        isolated = (~coupled_pre) & (~decays_pre)

        if drop_isolated_for_J is None:
            drop_mask = isolated
        else:
            drop_mask = np.zeros(n_pre, dtype=bool)
            for i, st in enumerate(states):
                if isolated[i] and j_in_list(
                    float(st.J), drop_isolated_for_J, j_match_tol
                ):
                    drop_mask[i] = True

        keep_idx = np.nonzero(~drop_mask)[0]
        states = [states[i] for i in keep_idx]

        if coupling_mats_arr is not None:
            coupling_mats_arr = [
                M[np.ix_(keep_idx, keep_idx)] for M in coupling_mats_arr
            ]
        if BR0 is not None:
            BR0 = np.asarray(BR0)[np.ix_(keep_idx, keep_idx)]

    # ---- identify decay-only groups eligible for collapsing ----
    n_post = len(states)
    coupled, decays = _compute_involvement(
        n_post,
        coupling_mats_arr,
        BR0,
        coupling_threshold=coupling_threshold,
        decay_threshold=decay_threshold,
        br_is_final_initial=br_is_final_initial,
    )
    decay_only = decays & (~coupled)

    # apply optional J filter for collapsing
    if combine_for_J is not None:
        decay_only = np.array(
            [
                bool(decay_only[i])
                and j_in_list(float(states[i].J), combine_for_J, j_match_tol)
                for i in range(n_post)
            ],
            dtype=bool,
        )

    # restrict collapsing to requested electronic manifolds
    if combine_for_electronic is None:
        allowed_elec = {ElectronicState.X}
    else:
        allowed_elec = set(combine_for_electronic)

    groups: dict[tuple[ElectronicState, float], list[int]] = defaultdict(list)
    for i, st in enumerate(states):
        if decay_only[i] and (st.electronic_state in allowed_elec):
            groups[(st.electronic_state, float(st.J))].append(i)

    # build explicit boolean mask of indices that will be collapsed
    to_collapse = np.zeros(n_post, dtype=bool)
    for idxs in groups.values():
        to_collapse[idxs] = True

    if not to_collapse.any():
        return (
            list(states),
            None if coupling_mats_arr is None else list(coupling_mats_arr),
            BR0,
        )

    old_to_new = np.full(n_post, -1, dtype=int)
    new_states: list[Any] = []

    # keep all states that are NOT being collapsed
    for i, st in enumerate(states):
        if not to_collapse[i]:
            old_to_new[i] = len(new_states)
            new_states.append(st)

    # create combined states for each group and map old indices
    for (elec, J), idxs in sorted(
        groups.items(), key=lambda kv: (kv[0][0].value, kv[0][1])
    ):
        new_i = len(new_states)
        new_states.append(_CombinedState(electronic_state=elec, J=J, mF=0.0))
        for old_i in idxs:
            old_to_new[old_i] = new_i

    n1 = len(new_states)

    BR1 = None
    if BR0 is not None:
        BR1 = np.zeros((n1, n1), dtype=float)
        rr, cc = np.nonzero(BR0)
        if rr.size:
            np.add.at(BR1, (old_to_new[rr], old_to_new[cc]), BR0[rr, cc])

    coupling1 = None
    if coupling_mats_arr is not None:
        coupling1 = []
        for M in coupling_mats_arr:
            M1 = np.zeros((n1, n1), dtype=M.dtype)
            rr, cc = np.nonzero(M)
            if rr.size:
                np.add.at(M1, (old_to_new[rr], old_to_new[cc]), M[rr, cc])
            coupling1.append(M1)

    return new_states, coupling1, BR1


# ---------------- main plotting function (assumes combining done externally) ----------------
def plot_level_diagram(
    states: Sequence[Any],  # CoupledBasisState, imported elsewhere
    coupling_mats: Sequence[np.ndarray] | None = None,
    branching_ratio: np.ndarray | None = None,
    *,
    ax: Axes | None = None,
    coupling_threshold: float = 0.0,
    decay_threshold: float = 0.0,
    # label sizes
    mf_label_fontsize: float | str | None = None,
    j_label_fontsize: float | str | None = None,
    right_label_fontsize: float | str | None = None,
    isolated_level_alpha: float = 0.7,
    collapse_decay_to_J: bool = False,
    collapse_couplings_to_J: bool = False,
    # layout
    electronic_gap_y: float = 10.0,
    j_gap_y: float = 5.0,
    f1_gap_y: float = 3.8,
    f_gap_y: float = 2.2,
    j_gap_x: float = 3.0,
    mf_spacing_x: float = 1.0,
    level_halfwidth: float = 0.35,
    level_lw: float = 2.0,
    coupling_lw: float = 1.5,
    decay_lw: float = 1.2,
    coupling_alpha: float = 0.9,
    decay_alpha: float = 0.7,
) -> Axes:
    """Plot a simplified level diagram grouped by electronic manifold and $J$.

    Assumed convention: `branching_ratio[final, initial]` corresponds to decay
    initial → final.
    """

    n0 = len(states)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))

    # ---------------- label styling (derived from Matplotlib defaults) ----------------
    def _fontsize_to_points(size: object, *, default: float) -> float:
        try:
            return float(FontProperties(size=cast(Any, size)).get_size_in_points())
        except Exception:
            try:
                return float(size)  # type: ignore[arg-type]
            except Exception:
                return float(default)

    base_fs = _fontsize_to_points(plt.rcParams.get("font.size", 10.0), default=10.0)
    label_fs = _fontsize_to_points(
        plt.rcParams.get("axes.labelsize", base_fs), default=base_fs
    )
    tick_fs = _fontsize_to_points(
        plt.rcParams.get("xtick.labelsize", base_fs), default=base_fs
    )
    mf_fs_default = float(min(tick_fs, base_fs))
    j_fs_default = float(max(label_fs, base_fs))
    right_fs_default = float(base_fs)

    mf_label_fontsize = (
        _fontsize_to_points(mf_label_fontsize, default=mf_fs_default)
        if mf_label_fontsize is not None
        else mf_fs_default
    )
    j_label_fontsize = (
        _fontsize_to_points(j_label_fontsize, default=j_fs_default)
        if j_label_fontsize is not None
        else j_fs_default
    )
    right_label_fontsize = (
        _fontsize_to_points(right_label_fontsize, default=right_fs_default)
        if right_label_fontsize is not None
        else right_fs_default
    )

    def _measure_text_points(text: str, fontsize: float) -> tuple[float, float] | None:
        try:
            t = ax.text(
                0.0,
                0.0,
                text,
                fontsize=fontsize,
                alpha=0.0,
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
            ax.figure.canvas.draw()
            canvas = ax.figure.canvas
            get_renderer = getattr(canvas, "get_renderer", None)
            renderer = get_renderer() if callable(get_renderer) else None
            if renderer is None:
                t.remove()
                return None
            bbox = t.get_window_extent(renderer=cast(Any, renderer))
            t.remove()
            w_pts = float(bbox.width) * 72.0 / float(ax.figure.dpi)
            h_pts = float(bbox.height) * 72.0 / float(ax.figure.dpi)
            return w_pts, h_pts
        except Exception:
            return None

    mf_size = _measure_text_points(r"$-1/2$", mf_label_fontsize)
    j_size = _measure_text_points(r"$J=10$", j_label_fontsize)
    one_char = _measure_text_points(r"$F$", right_label_fontsize)

    mf_h_pts = float(mf_size[1]) if mf_size is not None else float(mf_label_fontsize)
    j_h_pts = float(j_size[1]) if j_size is not None else float(j_label_fontsize)
    one_char_w_pts = (
        float(one_char[0]) if one_char is not None else float(right_label_fontsize)
    )

    gap_pts = 0.25 * min(mf_h_pts, j_h_pts)
    mf_dy_pts_base = mf_h_pts + gap_pts
    j_dy_pts_base = mf_dy_pts_base + mf_h_pts + gap_pts
    pad_F_pts = one_char_w_pts
    pad_F1_extra_pts = one_char_w_pts

    # ---------------- validation + normalization ----------------
    coupling_mats_arr: list[np.ndarray] | None = None
    if coupling_mats is not None:
        coupling_mats_arr = []
        for k, M in enumerate(coupling_mats):
            A = np.asarray(M)
            if A.shape != (n0, n0):
                raise ValueError(
                    f"coupling_mats[{k}] has shape {A.shape}, expected ({n0},{n0})"
                )
            coupling_mats_arr.append(A)

    BR0: np.ndarray | None = None
    if branching_ratio is not None:
        BR0 = np.asarray(branching_ratio, dtype=float)
        if BR0.shape != (n0, n0):
            raise ValueError(
                f"`branching_ratio` has shape {BR0.shape}, expected ({n0},{n0})"
            )

    n = len(states)

    coupled, decays = _compute_involvement(
        n,
        coupling_mats_arr,
        BR0,
        coupling_threshold=coupling_threshold,
        decay_threshold=decay_threshold,
        br_is_final_initial=True,
    )
    participates = coupled | decays

    by_row: dict[
        tuple[ElectronicState, float, float | None, float | None], list[int]
    ] = defaultdict(list)
    by_elec_J: dict[tuple[ElectronicState, float], list[int]] = defaultdict(list)
    all_Js_set: set[float] = set()

    for i, st in enumerate(states):
        elec = st.electronic_state
        J = float(st.J)
        F1 = f_maybe(getattr(st, "F1", None))
        F = f_maybe(getattr(st, "F", None))
        by_row[(elec, J, F1, F)].append(i)
        by_elec_J[(elec, J)].append(i)
        all_Js_set.add(J)

    elec_order = [ElectronicState.X, ElectronicState.B]

    y_base: dict[tuple[ElectronicState, float, float | None, float | None], float] = {}
    current_y = 0.0
    for elec in elec_order:
        Js = sorted({J for (e, J, _, _) in by_row if e == elec})
        for J in Js:
            F1s = sort_none_last(
                list({F1 for (e, JJ, F1, _) in by_row if e == elec and JJ == J})
            )
            for F1 in F1s:
                Fs = sort_none_last(
                    [
                        F
                        for (e, JJ, FF1, F) in by_row
                        if e == elec and JJ == J and FF1 == F1
                    ]
                )

                for k, F in enumerate(Fs):
                    y_base[(elec, J, F1, F)] = current_y + k * f_gap_y

                if Fs:
                    current_y += (len(Fs) - 1) * f_gap_y
                current_y += f1_gap_y
            current_y += j_gap_y
        if elec == ElectronicState.X:
            current_y += electronic_gap_y

    all_Js = sorted(all_Js_set)

    def _mf_vals_from_max(mf_max: float) -> np.ndarray:
        mf2 = int(round(2.0 * mf_max))
        vals2 = np.arange(-mf2, mf2 + 1, 2, dtype=int)
        return vals2.astype(float) / 2.0

    mf_range_by_J: dict[float, np.ndarray] = {}
    ncols_by_J: dict[float, int] = {}
    for J in all_Js:
        idx_j = by_elec_J.get((ElectronicState.X, J), []) + by_elec_J.get(
            (ElectronicState.B, J), []
        )

        F_vals_j: list[float] = []
        for i in idx_j:
            F_i = getattr(states[i], "F", None)
            if F_i is None:
                continue
            F_vals_j.append(float(F_i))

        if F_vals_j:
            mf_max = max(F_vals_j)
        else:
            mf_abs_vals_j: list[float] = []
            for i in idx_j:
                mf_i = getattr(states[i], "mF", None)
                if mf_i is None:
                    continue
                mf_abs_vals_j.append(abs(float(mf_i)))
            mf_max = max(mf_abs_vals_j) if mf_abs_vals_j else float(J + 1)

        mf_vals = _mf_vals_from_max(float(mf_max))
        mf_range_by_J[J] = mf_vals
        ncols_by_J[J] = len(mf_vals)

    J_to_x0: dict[float, float] = {}
    x_cursor = 0.0
    for J in all_Js:
        J_to_x0[J] = x_cursor
        width = (ncols_by_J[J] - 1) * mf_spacing_x
        x_cursor += width + j_gap_x

    mf_grid_by_J: dict[float, dict[float, float]] = {}
    for J in all_Js:
        mf_grid_by_J[J] = {
            float(mf): k * mf_spacing_x for k, mf in enumerate(mf_range_by_J[J])
        }

    cmap = plt.get_cmap("tab10")
    J_to_color: dict[float, tuple] = {}
    for idx, J in enumerate(all_Js):
        J_to_color[J] = cmap(idx % getattr(cmap, "N", 10))

    x = np.zeros(n)
    y = np.zeros(n)

    for (elec, J, F1, F), idx in by_row.items():
        base_y = y_base[(elec, J, F1, F)]
        x0 = J_to_x0[J]
        mf_to_dx = mf_grid_by_J[J]

        for i in idx:
            mf_val = getattr(states[i], "mF", None)
            mf = float(mf_val) if mf_val is not None else 0.0

            if mf not in mf_to_dx:
                existing = np.array(sorted(mf_to_dx.keys()), dtype=float)
                new = np.sort(np.unique(np.append(existing, mf)))
                mf_grid_by_J[J] = {
                    float(v): k * mf_spacing_x for k, v in enumerate(new)
                }
                mf_to_dx = mf_grid_by_J[J]

            x[i] = x0 + mf_to_dx[mf]
            y[i] = base_y

    for i in range(n):
        J_i = float(states[i].J)
        color = J_to_color.get(J_i, "k")
        alpha_level = 1.0 if participates[i] else float(isolated_level_alpha)
        ax.plot(
            [x[i] - level_halfwidth, x[i] + level_halfwidth],
            [y[i], y[i]],
            lw=level_lw,
            color=color,
            alpha=alpha_level,
        )

    for elec in elec_order:
        Js = sorted({J for (e, J) in by_elec_J.keys() if e == elec})
        for J in Js:
            idx_ej = by_elec_J.get((elec, J), [])
            if not idx_ej:
                continue

            has_real_levels = any(
                not getattr(states[i], "is_combined", False) for i in idx_ej
            )

            x0 = J_to_x0[J]
            mf_to_dx = mf_grid_by_J[J]
            x_center = x0 + 0.5 * (min(mf_to_dx.values()) + max(mf_to_dx.values()))

            y_min = float(np.min(y[idx_ej]))
            y_max = float(np.max(y[idx_ej]))
            place_above = elec == ElectronicState.B

            y_anchor = y_max if place_above else y_min
            va_mf = "bottom" if place_above else "top"
            va_J = "bottom" if place_above else "top"
            mf_dy_pts = mf_dy_pts_base if place_above else -mf_dy_pts_base
            j_dy_pts = j_dy_pts_base if place_above else -j_dy_pts_base

            if has_real_levels:
                idx_real = [
                    i for i in idx_ej if not getattr(states[i], "is_combined", False)
                ]

                F_vals_ej: list[float] = []
                for i in idx_real:
                    F_i = getattr(states[i], "F", None)
                    if F_i is None:
                        continue
                    F_vals_ej.append(float(F_i))

                if F_vals_ej:
                    mf_max_label = max(F_vals_ej)
                else:
                    mf_abs_vals_ej: list[float] = []
                    for i in idx_real:
                        mf_i = getattr(states[i], "mF", None)
                        if mf_i is None:
                            continue
                        mf_abs_vals_ej.append(abs(float(mf_i)))
                    mf_max_label = max(mf_abs_vals_ej) if mf_abs_vals_ej else 0.0

                mf_labels = _mf_vals_from_max(float(mf_max_label))
                for mf in mf_labels:
                    ax.annotate(
                        f"${as_signed_frac2(mf)}$",
                        xy=(x0 + mf_to_dx[mf], y_anchor),
                        xycoords="data",
                        xytext=(0.0, mf_dy_pts),
                        textcoords="offset points",
                        ha="center",
                        va=va_mf,
                        fontsize=mf_label_fontsize,
                    )

            ax.annotate(
                f"$J={as_frac2(J)}$",
                xy=(x_center, y_anchor),
                xycoords="data",
                xytext=(0.0, j_dy_pts),
                textcoords="offset points",
                ha="center",
                va=va_J,
                fontsize=j_label_fontsize,
            )

    for elec in elec_order:
        Js = sorted({J for (e, J, _, _) in by_row if e == elec})
        for J in Js:
            idx_ej = by_elec_J.get((elec, J), [])
            if not idx_ej:
                continue

            row_keys = [
                (e, JJ, F1, F)
                for (e, JJ, F1, F) in by_row.keys()
                if e == elec and JJ == J
            ]
            if not row_keys:
                continue

            x_anchor = float(np.max(x[idx_ej])) + level_halfwidth
            rows_by_F1: dict[float, list[float]] = defaultdict(list)
            f_texts = []

            for e, JJ, F1, F in row_keys:
                if F1 is None or F is None:
                    continue
                y_row = y_base[(e, JJ, F1, F)]
                t = ax.annotate(
                    f"$F={as_frac2(F)}$",
                    xy=(x_anchor, y_row),
                    xycoords="data",
                    xytext=(pad_F_pts, 0.0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=right_label_fontsize,
                )
                f_texts.append(t)
                rows_by_F1[float(F1)].append(y_row)

            x1_max_px = None
            try:
                ax.figure.canvas.draw()
                canvas = ax.figure.canvas
                get_renderer = getattr(canvas, "get_renderer", None)
                renderer = get_renderer() if callable(get_renderer) else None
                if renderer is not None:
                    x1_max_px = max(
                        float(t.get_window_extent(renderer=cast(Any, renderer)).x1)
                        for t in f_texts
                    )
            except Exception:
                x1_max_px = None

            try:
                ax.figure.canvas.draw()
                x_anchor_px = float(ax.transData.transform((x_anchor, 0.0))[0])
            except Exception:
                x_anchor_px = None

            for F1, ys in rows_by_F1.items():
                ys_sorted = sorted(ys)
                y_f1 = 0.5 * (ys_sorted[0] + ys_sorted[-1])

                if x1_max_px is not None and x_anchor_px is not None:
                    dx_px = (x1_max_px - x_anchor_px) + (
                        pad_F1_extra_pts * ax.figure.dpi / 72.0
                    )
                    dx_pts = dx_px * 72.0 / ax.figure.dpi
                else:
                    dx_pts = pad_F_pts + pad_F1_extra_pts

                ax.annotate(
                    f"$F_1={as_frac2(F1)}$",
                    xy=(x_anchor, y_f1),
                    xycoords="data",
                    xytext=(dx_pts, 0.0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=right_label_fontsize,
                )

    # ---------------- couplings ----------------
    if coupling_mats_arr:
        for M in coupling_mats_arr:
            A = np.abs(M)
            iu, ju = np.triu_indices(n, k=1)
            keep = A[iu, ju] > coupling_threshold
            if not np.any(keep):
                continue

            if not collapse_couplings_to_J:
                for i, j in zip(iu[keep], ju[keep]):
                    ax.annotate(
                        "",
                        xy=(float(x[j]), float(y[j])),
                        xytext=(float(x[i]), float(y[i])),
                        arrowprops=dict(
                            arrowstyle="<->",
                            linestyle="-",
                            lw=coupling_lw,
                            alpha=coupling_alpha,
                            color="k",
                            shrinkA=0.0,
                            shrinkB=0.0,
                        ),
                    )
            else:
                cpl_groups: dict[
                    tuple[ElectronicState, float, ElectronicState, float],
                    list[tuple[int, int, float]],
                ] = defaultdict(list)

                node_w = np.zeros(n, dtype=float)
                for i, j in zip(iu[keep], ju[keep]):
                    w = float(A[i, j])
                    if w <= 0.0:
                        continue
                    node_w[int(i)] += w
                    node_w[int(j)] += w

                for i, j in zip(iu[keep], ju[keep]):
                    w = float(A[i, j])
                    ei = states[i].electronic_state
                    ej = states[j].electronic_state
                    Ji = float(states[i].J)
                    Jj = float(states[j].J)

                    left = (ei.value, Ji)
                    right = (ej.value, Jj)
                    if left <= right:
                        key = (ei, Ji, ej, Jj)
                        cpl_groups[key].append((int(i), int(j), w))
                    else:
                        key = (ej, Jj, ei, Ji)
                        cpl_groups[key].append((int(j), int(i), w))

                anchors: dict[tuple[ElectronicState, float], tuple[float, float]] = {}
                for eJ, idxs in by_elec_J.items():
                    idx_arr = np.asarray(idxs, dtype=int)
                    if idx_arr.size == 0:
                        continue
                    w_arr = node_w[idx_arr]
                    wsum = float(np.sum(w_arr))
                    if wsum > 0.0:
                        xa = float(np.sum(x[idx_arr] * w_arr) / wsum)
                        ya = float(np.sum(y[idx_arr] * w_arr) / wsum)
                    else:
                        xa = float(np.mean(x[idx_arr]))
                        ya = float(np.mean(y[idx_arr]))
                    anchors[eJ] = (xa, ya)

                for (ei, Ji, ej, Jj), edges in cpl_groups.items():
                    if not edges:
                        continue
                    w = np.array([ww for _, _, ww in edges], dtype=float)
                    wmax = float(np.max(w))

                    x1, y1 = anchors.get(
                        (ei, Ji), (float(np.mean(x)), float(np.mean(y)))
                    )
                    x2, y2 = anchors.get(
                        (ej, Jj), (float(np.mean(x)), float(np.mean(y)))
                    )

                    ax.annotate(
                        "",
                        xy=(float(x2), float(y2)),
                        xytext=(float(x1), float(y1)),
                        arrowprops=dict(
                            arrowstyle="<->",
                            linestyle="-",
                            lw=coupling_lw * (0.5 + 2.0 * wmax),
                            alpha=coupling_alpha,
                            color="k",
                            shrinkA=0.0,
                            shrinkB=0.0,
                        ),
                    )

    # ---------------- decays ----------------
    if BR0 is not None:
        BR = BR0
        ij = np.argwhere(BR > decay_threshold)

        decay_arrowprops = dict(
            arrowstyle="->",
            linestyle="--",
            lw=decay_lw,
            alpha=decay_alpha,
            color="k",
        )

        if not collapse_decay_to_J:
            for j, i in ij:
                br = float(BR[j, i])
                if br <= decay_threshold:
                    continue
                if y[i] <= y[j]:
                    continue
                ax.annotate(
                    "",
                    xy=(float(x[j]), float(y[j])),
                    xytext=(float(x[i]), float(y[i])),
                    arrowprops=decay_arrowprops,
                )
        else:
            manifold_x: dict[tuple[ElectronicState, float], float] = {}
            manifold_y_top: dict[tuple[ElectronicState, float], float] = {}
            manifold_y_bottom: dict[tuple[ElectronicState, float], float] = {}
            for key, idxs in by_elec_J.items():
                idx_arr = np.asarray(idxs, dtype=int)
                if idx_arr.size == 0:
                    continue
                manifold_x[key] = float(np.mean(x[idx_arr]))
                manifold_y_top[key] = float(np.max(y[idx_arr]))
                manifold_y_bottom[key] = float(np.min(y[idx_arr]))

            decay_pairs: set[tuple[ElectronicState, float, ElectronicState, float]] = (
                set()
            )
            for j, i in ij:
                br = float(BR[j, i])
                if br <= decay_threshold:
                    continue
                if y[i] <= y[j]:
                    continue
                ei = states[i].electronic_state
                ej = states[j].electronic_state
                Ji = float(states[i].J)
                Jj = float(states[j].J)
                decay_pairs.add((ei, Ji, ej, Jj))

            for ei, Ji, ej, Jj in decay_pairs:
                key_i = (ei, Ji)
                key_f = (ej, Jj)
                if key_i not in manifold_x or key_f not in manifold_x:
                    continue

                x_start = manifold_x[key_i]
                x_end = manifold_x[key_f]
                y_start = manifold_y_top[key_i]
                y_end = manifold_y_bottom[key_f]
                if y_start <= y_end:
                    continue

                ax.annotate(
                    "",
                    xy=(x_end, y_end),
                    xytext=(x_start, y_start),
                    arrowprops=decay_arrowprops,
                )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axis("off")

    return ax


# =============================================================================
# Field-dressed X -> B transition level diagram
# =============================================================================
#
# The electric and magnetic fields are taken along z, so mF stays a good quantum
# number and the Hamiltonian block-diagonalizes by mF. Every block is
# non-degenerate at zero field, which is what makes the zero-field parent
# assignment well posed -- unlike a full diagonalization this needs no
# `B=[0,0,1e-5]` placeholder to lift the +/-mF degeneracies.
#
# States are matched to their zero-field parents by ramping the field up from
# zero in small steps and tracking eigenvectors (the adiabatic-continuation rule
# from AGENTS.md), so the assignment stays correct well above the fields where
# one-shot overlap matching fails.

_GROUND_PARENT_COLORS: tuple[str, ...] = (
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
)

# Fractions of the main axes width given to the left header gutter, the levels
# themselves, and the right F/F1 label gutter.
_FRAC_LEFT, _FRAC_LEVELS, _FRAC_RIGHT = 0.20, 0.62, 0.18

# Vertical layout of the main axes, in axes fractions (ylim is fixed to (0, 1)).
_Y_EXCITED_TICKS = 0.945
_Y_EXCITED_BAND = (0.660, 0.925)
_Y_ARROW = (0.460, 0.615)
_Y_GROUND_BAND = (0.115, 0.380)
_Y_GROUND_HEADERS = 0.425
_Y_GROUND_TICKS = 0.075


@dataclass(frozen=True)
class DressedLevel:
    """A single field-dressed level, labelled by the zero-field state it
    adiabatically connects to.

    Attributes
    ----------
    electronic_state
        X or B.
    F1, F, mF, P
        Quantum numbers of the *zero-field parent*, not of the dressed state.
        `P` is None for X levels, where parity is fixed by J.
    energy_MHz
        Eigenvalue at the requested field, in MHz. Absolute within its own
        manifold; the two manifolds do not share an energy origin.
    character
        Fraction of the dressed state on each plotted zero-field parent,
        renormalized to sum to 1. Keys are `(F1, F)` tuples for X levels and
        the parent parity `P` (an int) for B levels.
    residual
        Weight *outside* the plotted parent space (other J, other F', ...),
        before renormalization. A small residual is what justifies the
        renormalization; a large one means the plotted parents are not a good
        description of the dressed state.
    """

    electronic_state: ElectronicState
    F1: float
    F: int
    mF: int
    P: int | None
    energy_MHz: float
    # `hash=False` keeps the dict out of the generated __hash__; without it a
    # frozen dataclass advertises hashability and then raises on the dict. Hashing
    # on a subset of the __eq__ fields is contract-valid.
    character: dict[Any, float] = field(default_factory=dict, hash=False)
    residual: float = 0.0


@dataclass(frozen=True)
class TransitionLevelStructure:
    """Field-dressed level structure of an optical transition."""

    transition: OpticalTransition
    E: float
    B: float
    ground: tuple[DressedLevel, ...]
    excited: tuple[DressedLevel, ...]
    ground_parents: tuple[tuple[float, int], ...]
    # `hash=False`: see the note on `DressedLevel.character`.
    zero_field_parity_splitting_MHz: dict[int, float] = field(hash=False)

    @property
    def max_ground_residual(self) -> float:
        return max((lv.residual for lv in self.ground), default=0.0)

    @property
    def max_excited_residual(self) -> float:
        return max((lv.residual for lv in self.excited), default=0.0)


@dataclass(frozen=True)
class TransitionLevelDiagram:
    """Result of `plot_transition_level_diagram`."""

    fig: Figure
    ax: Axes
    info_ax: Axes | None
    structure: TransitionLevelStructure


# ---------------- transition specification ----------------
def _coerce_transition(
    transition: OpticalTransition | None,
    *,
    J_ground: int | None,
    J_excited: int | None,
    F1_excited: float | None,
    F_excited: int | None,
    branch: OpticalTransitionType | str | int | None,
) -> OpticalTransition:
    """Accept either an `OpticalTransition` or explicit quantum numbers."""
    explicit = (J_ground, J_excited, F1_excited, F_excited, branch)

    if transition is not None:
        if any(v is not None for v in explicit):
            raise ValueError(
                "supply either transition=... or explicit quantum numbers, not both"
            )
        if not isinstance(transition, OpticalTransition):
            raise TypeError("transition must be an OpticalTransition")
        return transition

    if J_ground is None or F1_excited is None or F_excited is None:
        raise ValueError("explicit mode requires J_ground, F1_excited and F_excited")
    if branch is None and J_excited is None:
        raise ValueError("explicit mode requires either branch or J_excited")

    if branch is not None:
        if isinstance(branch, OpticalTransitionType):
            t = branch
        elif isinstance(branch, str):
            t = OpticalTransitionType[branch.upper()]
        else:
            t = OpticalTransitionType(int(branch))
        J_e = int(J_ground) + int(t.value)
        if J_excited is not None and int(J_excited) != J_e:
            raise ValueError(
                f"{t.name}({J_ground}) implies J_excited={J_e}, not {J_excited}"
            )
    else:
        dJ = int(cast(int, J_excited)) - int(J_ground)
        try:
            t = OpticalTransitionType(dJ)
        except ValueError as exc:
            raise ValueError(f"delta J = {dJ} is not one of O, P, Q, R, S") from exc

    return OpticalTransition(
        t=t,
        J_ground=int(J_ground),
        F1_excited=float(F1_excited),
        F_excited=int(F_excited),
    )


# ---------------- Hamiltonians in the coupled basis ----------------
def _qn_key(state: Any) -> tuple:
    """Hashable label for a CoupledBasisState, robust to how it was generated."""
    P = getattr(state, "P", None)
    return (
        state.electronic_state,
        int(state.J),
        float(state.F1),
        int(state.F),
        int(state.mF),
        None if P is None else int(P),
    )


def _x_matrices(
    Jmin: int, Jmax: int
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """X-state field-free, Stark and Zeeman matrices in the coupled basis."""
    Js = list(range(int(Jmin), int(Jmax) + 1))
    qn_uncoupled = list(generate_uncoupled_states_ground(Js=Js))
    qn_coupled = list(generate_coupled_states_ground(Js=Js))
    H = generate_uncoupled_hamiltonian_X(qn_uncoupled)
    S = generate_transform_matrix(qn_uncoupled, qn_coupled)

    def to_coupled(M: np.ndarray) -> np.ndarray:
        return np.asarray(S.conj().T @ M @ S)

    return qn_coupled, to_coupled(H.Hff), to_coupled(H.HSz), to_coupled(H.HZz)


def _b_matrices(
    Jmin: int, Jmax: int
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """B-state field-free, Stark and Zeeman matrices in the parity basis."""
    selector = QuantumSelector(
        J=np.arange(int(Jmin), int(Jmax) + 1),
        F1=None,
        F=None,
        mF=None,
        electronic=ElectronicState.B,
        P=[-1, +1],
        Ω=1,
    )
    qn = list(generate_coupled_states_B(selector))
    H = generate_coupled_hamiltonian_B(qn)
    H0 = H.Hrot + H.H_mhf_Tl + H.H_mhf_F + H.H_LD + H.H_cp1_Tl + H.H_c_Tl
    return qn, np.asarray(H0), np.asarray(H.HSz), np.asarray(H.HZz)


def _track_block(
    H0: np.ndarray,
    H_field: np.ndarray,
    indices: Sequence[int],
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adiabatically follow one mF block from zero field to the full field.

    `H_field` is the *total* field perturbation (E*HSz + B*HZz); it is ramped in
    from zero over `n_steps` steps, matching eigenvectors to the previous step
    with a Hungarian assignment on the overlap matrix.

    Returns (w_zero_field, V_zero_field, w_field, V_field).
    """
    ix = np.asarray(indices, dtype=int)
    H0b = H0[np.ix_(ix, ix)]
    Hfb = H_field[np.ix_(ix, ix)]

    w0, V0 = eigh(H0b)
    w_prev, V_prev = w0.copy(), V0.copy()

    for s in np.linspace(0.0, 1.0, int(n_steps) + 1)[1:]:
        w, V = eigh(H0b + s * Hfb)

        overlap = np.abs(V_prev.conj().T @ V) ** 2
        rows, cols = linear_sum_assignment(-overlap)
        order = np.empty(len(cols), dtype=int)
        order[rows] = cols
        w, V = w[order], V[:, order]

        # continuous phase convention, so the overlaps below stay meaningful
        phase = np.sum(V_prev.conj() * V, axis=0)
        nonzero = np.abs(phase) > 0
        phase[nonzero] /= np.abs(phase[nonzero])
        phase[~nonzero] = 1.0
        V = V * phase.conj()

        w_prev, V_prev = w, V

    return w0, V0, w_prev, V_prev


def _assign_parents(
    basis: Sequence[Any],
    block: Sequence[int],
    V0: np.ndarray,
    parents: Sequence[Any],
) -> dict[tuple, int]:
    """Map each nominal zero-field parent onto a column of `V0`.

    Uses a Hungarian assignment on |<parent|eigenvector>|^2 so that two parents
    can never claim the same eigenvector.
    """
    index_of = {_qn_key(qn): i for i, qn in enumerate(basis)}
    local = {g: i for i, g in enumerate(block)}

    rows = []
    for qn in parents:
        key = _qn_key(qn)
        if key not in index_of:
            raise ValueError(f"basis state not found: {qn!r}")
        rows.append(local[index_of[key]])

    overlap = np.asarray([np.abs(V0[i, :]) ** 2 for i in rows])
    r_idx, c_idx = linear_sum_assignment(-overlap)
    return {
        _qn_key(parents[int(r)]): int(c) for r, c in zip(r_idx, c_idx, strict=True)
    }


def _n_tracking_steps(
    E: float, B: float, max_step_V_cm: float, max_step_G: float
) -> int:
    """Number of ramp steps needed to track both fields on from zero.

    Both fields have to be stepped: sizing the ramp on `E` alone leaves a large
    `B` (e.g. `E=0, B=500`) crossing its whole range in the floor of 8 steps,
    which can hand a level the wrong zero-field parent with no warning.
    """
    steps_E = int(np.ceil(abs(float(E)) / max(float(max_step_V_cm), 1e-12)))
    steps_B = int(np.ceil(abs(float(B)) / max(float(max_step_G), 1e-12)))
    return max(steps_E, steps_B, 8 if B else 1, 1)


def _characterize(
    V0: np.ndarray, psi: np.ndarray, parent_columns: dict[Any, int]
) -> tuple[dict[Any, float], float]:
    """Decompose a dressed state over the selected zero-field parents."""
    raw = {
        key: float(abs(np.vdot(V0[:, col], psi)) ** 2)
        for key, col in parent_columns.items()
    }
    selected = sum(raw.values())
    character = (
        {key: value / selected for key, value in raw.items()} if selected > 0 else raw
    )
    return character, max(0.0, 1.0 - selected)


# ---------------- level structure ----------------
def _calculate_ground(
    transition: OpticalTransition,
    E: float,
    B: float,
    j_padding: int,
    max_step_V_cm: float,
    max_step_G: float,
) -> tuple[list[DressedLevel], list[tuple[float, int]]]:
    J = int(transition.J_ground)
    basis, H0, HSz, HZz = _x_matrices(max(0, J - j_padding), J + j_padding)
    H_field = E * HSz + B * HZz
    n_steps = _n_tracking_steps(E, B, max_step_V_cm, max_step_G)

    parents: list[Any] = sorted(
        generate_coupled_states_X(QuantumSelector(J=J)),
        key=lambda qn: (-float(qn.F1), -int(qn.F), int(qn.mF)),
    )
    families = sorted(
        {(float(qn.F1), int(qn.F)) for qn in parents},
        key=lambda fam: (-fam[0], -fam[1]),
    )

    levels: list[DressedLevel] = []
    for mF in sorted({int(qn.mF) for qn in parents}):
        block = [i for i, qn in enumerate(basis) if int(qn.mF) == mF]
        _, V0, w, V = _track_block(H0, H_field, block, n_steps)

        nominal = [qn for qn in parents if int(qn.mF) == mF]
        columns = _assign_parents(basis, block, V0, nominal)

        # families with |mF| > F simply have no level in this column
        plotted = {
            (float(qn.F1), int(qn.F)): columns[_qn_key(qn)] for qn in nominal
        }

        for qn in nominal:
            col = columns[_qn_key(qn)]
            character, residual = _characterize(V0, V[:, col], plotted)
            levels.append(
                DressedLevel(
                    electronic_state=ElectronicState.X,
                    F1=float(qn.F1),
                    F=int(qn.F),
                    mF=mF,
                    P=None,
                    energy_MHz=float(w[col] / 1e6),
                    character={fam: character.get(fam, 0.0) for fam in families},
                    residual=residual,
                )
            )

    return levels, families


def _calculate_excited(
    transition: OpticalTransition,
    E: float,
    B: float,
    j_padding: int,
    max_step_V_cm: float,
    max_step_G: float,
) -> tuple[list[DressedLevel], dict[int, float]]:
    J = int(transition.J_excited)
    F1 = float(transition.F1_excited)
    F = int(transition.F_excited)

    basis, H0, HSz, HZz = _b_matrices(max(1, J - j_padding), J + j_padding)
    H_field = E * HSz + B * HZz
    n_steps = _n_tracking_steps(E, B, max_step_V_cm, max_step_G)

    parents: list[Any] = []
    for P in (-1, +1):
        parents += sorted(
            generate_coupled_states_B(
                QuantumSelector(
                    J=J,
                    F1=F1,
                    F=F,
                    P=P,
                    Ω=1,
                    electronic=ElectronicState.B,
                )
            ),
            key=lambda qn: int(qn.mF),
        )

    levels: list[DressedLevel] = []
    parity_splitting: dict[int, float] = {}

    for mF in range(-F, F + 1):
        block = [i for i, qn in enumerate(basis) if int(qn.mF) == mF]
        w0, V0, w, V = _track_block(H0, H_field, block, n_steps)

        nominal = [qn for qn in parents if int(qn.mF) == mF]
        columns = _assign_parents(basis, block, V0, nominal)
        plotted = {
            int(qn.P): columns[_qn_key(qn)] for qn in nominal if qn.P is not None
        }

        parity_splitting[mF] = float(abs(w0[plotted[+1]] - w0[plotted[-1]]) / 1e6)

        for P in (-1, +1):
            col = plotted[P]
            character, residual = _characterize(V0, V[:, col], plotted)
            levels.append(
                DressedLevel(
                    electronic_state=ElectronicState.B,
                    F1=F1,
                    F=F,
                    mF=mF,
                    P=P,
                    energy_MHz=float(w[col] / 1e6),
                    character=character,
                    residual=residual,
                )
            )

    return levels, parity_splitting


def calculate_transition_level_structure(
    transition: OpticalTransition | None = None,
    *,
    E: float = 170.0,
    B: float = 0.0,
    J_ground: int | None = None,
    J_excited: int | None = None,
    F1_excited: float | None = None,
    F_excited: int | None = None,
    branch: OpticalTransitionType | str | int | None = None,
    x_j_padding: int = 2,
    b_j_padding: int = 2,
    max_tracking_step_V_cm: float = 2.0,
    max_tracking_step_G: float = 1.0,
) -> TransitionLevelStructure:
    """Field-dressed X and B levels of an optical transition, with the zero-field
    parent character of every level.

    The fields are along z (`E` in V/cm, `B` in Gauss), so mF stays good and the
    calculation runs per mF block. Both manifolds are followed adiabatically from
    zero field, which keeps the parent labelling correct well past the field
    where one-shot overlap matching starts mislabelling states.

    Specify the transition either as an `OpticalTransition`::

        calculate_transition_level_structure(transitions.P2_F1_3o2_F1, E=170)

    or through explicit quantum numbers::

        calculate_transition_level_structure(
            E=170, J_ground=2, branch="P", F1_excited=1.5, F_excited=1
        )

    Parameters
    ----------
    E
        Electric field along z, V/cm.
    B
        Magnetic field along z, Gauss. Zero by default; a nonzero value splits
        the +/-mF pairs.
    x_j_padding, b_j_padding
        How many rotational levels either side of the target J to include in the
        diagonalization. The default of 2 is converged at few-hundred V/cm;
        raise it in the kV/cm regime (see AGENTS.md on J truncation).
    max_tracking_step_V_cm, max_tracking_step_G
        Electric and magnetic field steps used for adiabatic tracking; the ramp
        uses whichever demands more steps. Smaller is safer near avoided
        crossings; a converged result is step-size independent.

    Returns
    -------
    TransitionLevelStructure
    """
    tr = _coerce_transition(
        transition,
        J_ground=J_ground,
        J_excited=J_excited,
        F1_excited=F1_excited,
        F_excited=F_excited,
        branch=branch,
    )
    E = float(E)
    B = float(B)

    ground, families = _calculate_ground(
        tr,
        E,
        B,
        int(x_j_padding),
        float(max_tracking_step_V_cm),
        float(max_tracking_step_G),
    )
    excited, parity_splitting = _calculate_excited(
        tr,
        E,
        B,
        int(b_j_padding),
        float(max_tracking_step_V_cm),
        float(max_tracking_step_G),
    )

    return TransitionLevelStructure(
        transition=tr,
        E=E,
        B=B,
        ground=tuple(ground),
        excited=tuple(excited),
        ground_parents=tuple(families),
        zero_field_parity_splitting_MHz=parity_splitting,
    )


# ---------------- plotting helpers ----------------
def _fmt_signed_mf(mF: int) -> str:
    return "0" if mF == 0 else f"{mF:+d}"


def _band_y(
    levels: Sequence[DressedLevel],
    y_low: float,
    y_high: float,
    min_separation: float,
) -> list[float]:
    """Map level energies onto a vertical band, parallel to `levels`.

    Energies are scaled linearly into `[y_low, y_high]`. Levels sharing an mF
    column that would end up closer than `min_separation` are then pushed apart,
    which is purely cosmetic -- it keeps near-degenerate levels distinguishable
    at the cost of the exact vertical spacing inside a column.
    """
    if not levels:
        return []

    energies = np.array([lv.energy_MHz for lv in levels], dtype=float)
    lo, hi = float(energies.min()), float(energies.max())
    if np.isclose(lo, hi):
        y = np.full(energies.shape, 0.5 * (y_low + y_high))
    else:
        y = y_low + (energies - lo) / (hi - lo) * (y_high - y_low)

    if min_separation > 0:
        by_column: dict[int, list[int]] = defaultdict(list)
        for i, lv in enumerate(levels):
            by_column[lv.mF].append(i)

        for idx in by_column.values():
            if len(idx) < 2:
                continue
            order = sorted(idx, key=lambda i: y[i])
            for a, b in pairwise(order):
                y[b] = max(y[b], y[a] + min_separation)

            top, bottom = y[order[-1]], y[order[0]]
            if top > y_high:
                shift = top - y_high
                if bottom - shift >= y_low:
                    for i in order:
                        y[i] -= shift
                elif top > bottom:
                    # does not fit even after shifting: compress into the band
                    for i in order:
                        y[i] = y_low + (y[i] - bottom) / (top - bottom) * (
                            y_high - y_low
                        )

    return [float(value) for value in y]


def _draw_segmented_level(
    ax: Axes,
    x: float,
    y: float,
    fractions: Sequence[float],
    colors: Sequence[Any],
    width: float,
    lw: float,
) -> None:
    """One level, drawn as coloured segments whose lengths are its character."""
    total = float(sum(fractions))
    if total <= 0:
        return
    x0 = x - width / 2
    for fraction, color in zip(fractions, colors, strict=False):
        if fraction <= 0:
            continue
        x1 = x0 + width * float(fraction) / total
        ax.plot([x0, x1], [y, y], lw=lw, color=color, solid_capstyle="butt")
        x0 = x1


def _level_segments(
    level: DressedLevel,
    parents: Sequence[Any],
    colors: Mapping[Any, Any],
    show_residual: bool,
    residual_color: Any,
) -> tuple[list[float], list[Any]]:
    """Segment lengths and colours for one level, as fractions of the full width.

    `level.character` is renormalized to sum to 1 over the plotted parents, so it
    has to be scaled back down by `1 - residual` before the residual segment is
    appended; otherwise every segment is drawn shortened by a factor `1 + residual`
    and the grey residual bar comes out too short as well.
    """
    scale = (1.0 - level.residual) if show_residual else 1.0
    fractions = [scale * level.character.get(parent, 0.0) for parent in parents]
    segment_colors = [colors[parent] for parent in parents]
    if show_residual:
        fractions.append(level.residual)
        segment_colors.append(residual_color)
    return fractions, segment_colors


def _even_spacing(n: int, y_low: float, y_high: float) -> list[float]:
    """`n` positions from the top of a band downwards."""
    if n <= 0:
        return []
    if n == 1:
        return [0.5 * (y_low + y_high)]
    return list(np.linspace(y_high, y_low, n))


def _draw_family_bracket(
    ax: Axes,
    x_line: float,
    x_label: float,
    y_values: Sequence[float],
    label: str,
    color: Any,
    fontsize: float,
    lw: float = 1.4,
) -> float:
    """Vertical bracket spanning a group of levels, with a label at its centre.

    Replaces labelling a group at the mean of its levels, which stops pointing
    at anything once the Stark effect spreads the group out.
    """
    y_lo, y_hi = float(min(y_values)), float(max(y_values))
    y_mid = 0.5 * (y_lo + y_hi)
    if y_hi - y_lo > 1e-9:
        ax.plot([x_line, x_line], [y_lo, y_hi], lw=lw, color=color,
                solid_capstyle="butt")
    ax.text(x_label, y_mid, label, ha="center", va="center", fontsize=fontsize,
            color=color)
    return y_mid


# ---------------- information panel ----------------
_INFO_ROW_UNITS: dict[str, float] = {
    "title": 1.90,
    "subtitle": 1.55,
    "rule": 1.00,
    "header": 1.75,
    "line": 1.40,
    "swatch": 1.40,
    "note": 1.30,
    "box_start": 0.30,
    "box_end": 0.55,
}
_INFO_FONT_SCALE: dict[str, float] = {
    "title": 1.12,
    "subtitle": 0.95,
    "header": 0.95,
    "line": 0.85,
    "swatch": 0.78,
    "note": 0.68,
    "sticks": 0.72,
}


def _info_row_units(kind: str, payload: Any) -> float:
    if kind == "sticks":
        return 8.6 if payload[1] else 5.4
    if kind == "gap":
        return float(payload)
    return _INFO_ROW_UNITS[kind]


def _render_info_panel(info: Axes, rows: Sequence[tuple[str, Any]], base_fs: float) -> None:
    """Lay the info panel out top-down in points, shrinking to fit.

    Every row declares its height in units of the base font size, so the panel
    can never run off the bottom the way a hand-tuned chain of `y -= 0.03` does
    once the number of mF levels grows.
    """
    info.set_xlim(0, 1)
    info.set_ylim(0, 1)
    info.axis("off")

    parent = info.get_figure()
    if parent is None:
        return
    # an Axes inside a SubFigure reports the SubFigure, which has no size of its own
    root = cast(Figure, getattr(parent, "figure", parent) or parent)
    height_pts = float(info.get_position().height * root.get_figheight() * 72.0)
    if height_pts <= 0:
        return

    needed = sum(_info_row_units(kind, payload) for kind, payload in rows)
    scale = 1.0
    if needed > 0:
        scale = min(1.0, 0.99 * height_pts / (needed * base_fs))
    fs = base_fs * scale
    unit = fs / height_pts

    y = 1.0
    box_top: float | None = None

    for kind, payload in rows:
        height = _info_row_units(kind, payload) * unit
        y_top, y_bot = y, y - height
        y_mid = 0.5 * (y_top + y_bot)
        y -= height

        if kind == "gap":
            continue

        if kind == "box_start":
            box_top = y_top
            continue

        if kind == "box_end":
            if box_top is not None:
                info.add_patch(
                    plt.Rectangle(
                        (0.03, y_bot),
                        0.94,
                        box_top - y_bot,
                        fill=False,
                        linewidth=1.0,
                        clip_on=False,
                    )
                )
                box_top = None
            continue

        if kind == "rule":
            info.plot([0.06, 0.94], [y_mid, y_mid], lw=0.8, color="0.35")
            continue

        if kind in ("title", "subtitle"):
            info.text(
                0.50,
                y_mid,
                payload,
                ha="center",
                va="center",
                fontsize=fs * _INFO_FONT_SCALE[kind],
            )
            continue

        if kind in ("header", "note"):
            info.text(
                0.06,
                y_mid,
                payload,
                ha="left",
                va="center",
                fontsize=fs * _INFO_FONT_SCALE[kind],
            )
            continue

        if kind == "line":
            info.text(
                0.09,
                y_mid,
                payload,
                ha="left",
                va="center",
                fontsize=fs * _INFO_FONT_SCALE["line"],
            )
            continue

        if kind == "swatch":
            color, label = payload
            info.plot([0.08, 0.17], [y_mid, y_mid], lw=4.0, color=color,
                      solid_capstyle="butt")
            info.text(
                0.20,
                y_mid,
                label,
                ha="left",
                va="center",
                fontsize=fs * _INFO_FONT_SCALE["swatch"],
            )
            continue

        if kind == "sticks":
            labels, rotate = payload
            n = len(labels)
            base = y_bot + 0.08 * height
            top = base + 0.30 * height
            xs = (
                np.linspace(0.17, 0.83, n)
                if n > 1
                else np.array([0.5])
            )
            info.plot([0.08, 0.92], [base, base], lw=1.1, color="black")
            for x, label in zip(xs, labels, strict=True):
                info.plot([x, x], [base, top], lw=2.3, color="black")
                info.text(
                    x,
                    top + 0.06 * height,
                    label,
                    ha="center",
                    va="bottom",
                    rotation=90 if rotate else 0,
                    fontsize=fs * _INFO_FONT_SCALE["sticks"],
                )
            continue

        raise ValueError(f"unknown info-panel row kind: {kind!r}")


def _info_mF_groups(
    excited: Sequence[DressedLevel], B: float
) -> list[tuple[str, list[DressedLevel]]]:
    """`(label, levels)` for each mF column of the excited manifold.

    At `B = 0` the +/-mF pairs are degenerate, so one row per |mF| genuinely
    describes both and is labelled that way. A nonzero B splits them (different
    separation, different parity mixing), so each mF then gets its own signed
    row -- otherwise the negative partners are silently dropped while the label
    still claims to cover both signs.
    """
    mFs = sorted({lv.mF for lv in excited})
    if not B:
        mFs = [mF for mF in mFs if mF >= 0]
        labels = [r"m_F'=0" if mF == 0 else rf"|m_F'|={mF}" for mF in mFs]
    else:
        labels = [r"m_F'=0" if mF == 0 else rf"m_F'={mF:+d}" for mF in mFs]
    return [
        (label, [lv for lv in excited if lv.mF == mF])
        for label, mF in zip(labels, mFs, strict=True)
    ]


def _build_info_rows(
    structure: TransitionLevelStructure,
    *,
    ground_colors: dict[tuple[float, int], Any],
    parity_colors: dict[int, Any],
    show_ground_residual: bool,
    show_excited_residual: bool,
) -> list[tuple[str, Any]]:
    E = structure.E
    excited = structure.excited
    mF_groups = _info_mF_groups(excited, structure.B)

    mF0 = [lv for lv in excited if lv.mF == 0]
    reference = (
        min(lv.energy_MHz for lv in mF0)
        if mF0
        else min(lv.energy_MHz for lv in excited)
    )

    offsets: list[float] = []
    for value in sorted(lv.energy_MHz - reference for lv in excited):
        if not offsets or abs(value - offsets[-1]) > 1e-4:
            offsets.append(float(value))
    stick_labels = [
        "0 MHz" if abs(v) < 5e-4 else f"{v:+.3f} MHz" for v in offsets
    ]

    rows: list[tuple[str, Any]] = [
        ("box_start", None),
        ("title", "Relative excited-state offsets"),
        ("subtitle", r"(relative to lowest $m_F'=0$ level)"),
        ("sticks", (stick_labels, len(stick_labels) > 3)),
        ("box_end", None),
        ("gap", 0.8),
        ("header", f"Opposite-parity-parent separation at {E:g} V/cm"),
    ]

    for label, pair in mF_groups:
        if len(pair) != 2:
            continue
        separation = abs(pair[0].energy_MHz - pair[1].energy_MHz)
        rows.append(("line", rf"${label}:\ {separation:.3f}\ \mathrm{{MHz}}$"))

    splittings = list(structure.zero_field_parity_splitting_MHz.values())
    rows.append(("rule", None))
    rows.append(("header", "Zero-field parity splitting"))
    if splittings and max(splittings) - min(splittings) > 5e-4:
        rows.append(
            (
                "line",
                rf"${min(splittings):.3f}$ to ${max(splittings):.3f}\ \mathrm{{MHz}}$",
            )
        )
    elif splittings:
        rows.append(("line", rf"${np.mean(splittings):.3f}\ \mathrm{{MHz}}$"))

    rows.append(("rule", None))
    rows.append(("header", f"Parity mixing at {E:g} V/cm (lower level)"))
    for label, pair in mF_groups:
        if not pair:
            continue
        lower = min(pair, key=lambda lv: lv.energy_MHz)
        minus = 100 * lower.character.get(-1, 0.0)
        plus = 100 * lower.character.get(+1, 0.0)
        rows.append(("line", rf"${label}:\ {minus:.1f}/{plus:.1f}$"))

    rows.append(("gap", 0.5))
    rows.append(("swatch", (parity_colors[-1], r"zero-field $P'=-1$ character")))
    rows.append(("swatch", (parity_colors[+1], r"zero-field $P'=+1$ character")))

    rows.append(("gap", 0.5))
    rows.append(("header", "Ground-state zero-field hyperfine character"))
    for F1, F in structure.ground_parents:
        rows.append(
            (
                "swatch",
                (
                    ground_colors[(F1, F)],
                    rf"$F_1={as_frac2(F1)},\ F={F}$",
                ),
            )
        )

    notes = []
    if show_ground_residual:
        notes.append(
            f"other ground character up to {100 * structure.max_ground_residual:.2f}%"
        )
    if show_excited_residual:
        notes.append(
            f"other excited character up to {100 * structure.max_excited_residual:.2f}%"
        )
    if notes:
        rows.append(("gap", 0.4))
        rows.append(("note", "; ".join(notes)))

    return rows


# ---------------- main entry point ----------------
def plot_transition_level_diagram(
    transition: OpticalTransition | None = None,
    *,
    E: float = 170.0,
    B: float = 0.0,
    structure: TransitionLevelStructure | None = None,
    # transition specification, alternative to `transition`
    J_ground: int | None = None,
    J_excited: int | None = None,
    F1_excited: float | None = None,
    F_excited: int | None = None,
    branch: OpticalTransitionType | str | int | None = None,
    # calculation
    x_j_padding: int = 2,
    b_j_padding: int = 2,
    max_tracking_step_V_cm: float = 2.0,
    max_tracking_step_G: float = 1.0,
    # figure
    ax: Axes | None = None,
    info_ax: Axes | None = None,
    figsize: tuple[float, float] = (17.2, 9.2),
    base_fontsize: float = 13.0,
    show_info_panel: bool = True,
    title: str | None = None,
    # appearance
    level_width: float = 0.62,
    level_lw: float = 4.2,
    min_level_separation: float = 0.012,
    residual_threshold: float = 0.01,
    ground_colors: Sequence[Any] | None = None,
    parity_colors: Sequence[Any] | None = None,
    residual_color: Any = "0.65",
) -> TransitionLevelDiagram:
    """Plot a field-dressed TlF X->B level diagram for one optical transition.

    Each level is drawn as a horizontal bar split into coloured segments whose
    lengths are the level's zero-field parent character: hyperfine `(F1, F)`
    parents in X, the two Lambda-doublet parity parents in B. Character outside
    those parents is shown as a grey segment once it exceeds
    `residual_threshold`.

    Specify the transition as an `OpticalTransition`::

        plot_transition_level_diagram(transitions.P2_F1_3o2_F1, E=170)

    through explicit quantum numbers::

        plot_transition_level_diagram(
            E=170, J_ground=2, branch="P", F1_excited=1.5, F_excited=1
        )

    or from a structure you already calculated::

        s = calculate_transition_level_structure(J_ground=2, branch="P",
                                                 F1_excited=1.5, F_excited=1)
        plot_transition_level_diagram(structure=s)

    Parameters
    ----------
    E, B
        Fields along z, in V/cm and Gauss. Ignored if `structure` is given.
    ax, info_ax
        Draw into existing axes instead of creating a figure. If `ax` is given
        and `info_ax` is not, no information panel is drawn.
    base_fontsize
        Every font size in the figure is a fixed multiple of this, so the whole
        diagram scales with one number.
    min_level_separation
        Minimum vertical gap, in axes fractions, between two levels in the same
        mF column. Cosmetic; set to 0 for exact energy positions.
    residual_threshold
        Show the grey "other character" segment once any level's residual
        exceeds this.
    ground_colors, parity_colors
        Override the segment colours. `ground_colors` is indexed in the order of
        `structure.ground_parents` (highest F1, F first).

    Returns
    -------
    TransitionLevelDiagram
        Holds the figure, the axes, and the underlying
        `TransitionLevelStructure`, so every plotted number can be read back.
    """
    if structure is not None:
        specified = (
            transition,
            J_ground,
            J_excited,
            F1_excited,
            F_excited,
            branch,
        )
        if any(v is not None for v in specified):
            raise ValueError(
                "supply either structure=... or a transition specification, not both"
            )
    else:
        structure = calculate_transition_level_structure(
            transition,
            E=E,
            B=B,
            J_ground=J_ground,
            J_excited=J_excited,
            F1_excited=F1_excited,
            F_excited=F_excited,
            branch=branch,
            x_j_padding=x_j_padding,
            b_j_padding=b_j_padding,
            max_tracking_step_V_cm=max_tracking_step_V_cm,
            max_tracking_step_G=max_tracking_step_G,
        )

    tr = structure.transition
    ground = list(structure.ground)
    excited = list(structure.excited)
    families = list(structure.ground_parents)

    # ---------------- colours ----------------
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if parity_colors is None:
        parity_map = {-1: cycle[0], +1: cycle[1]}
    else:
        parity_map = {-1: parity_colors[0], +1: parity_colors[1]}

    palette = (
        tuple(ground_colors) if ground_colors is not None else _GROUND_PARENT_COLORS
    )
    ground_map = {
        fam: palette[i % len(palette)] for i, fam in enumerate(families)
    }

    show_ground_residual = structure.max_ground_residual >= residual_threshold
    show_excited_residual = structure.max_excited_residual >= residual_threshold

    # ---------------- axes ----------------
    created_figure = ax is None
    if created_figure:
        fig = plt.figure(figsize=figsize)
        if show_info_panel:
            gs = fig.add_gridspec(1, 2, width_ratios=[3.15, 1.35], wspace=0.08)
            ax = fig.add_subplot(gs[0, 0])
            info_ax = fig.add_subplot(gs[0, 1])
        else:
            ax = fig.add_subplot(1, 1, 1)
            info_ax = None
    else:
        ax = cast(Axes, ax)
        fig = cast(Figure, ax.get_figure())
        if not show_info_panel:
            info_ax = None

    # ---------------- horizontal layout ----------------
    mf_max = max(
        max((abs(lv.mF) for lv in ground), default=0),
        max((abs(lv.mF) for lv in excited), default=0),
    )
    span = 2.0 * mf_max if mf_max > 0 else 1.0
    total_width = span / _FRAC_LEVELS

    ax.set_xlim(
        -mf_max - _FRAC_LEFT * total_width, mf_max + _FRAC_RIGHT * total_width
    )
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # mF axis names sit left of the levels; the F / F1 key occupies the right
    # gutter, so neither can collide with the outermost mF column.
    x_axis_name = -mf_max - 0.075 * total_width
    x_f_label = mf_max + 0.075 * total_width
    x_f1_line = mf_max + 0.130 * total_width
    x_f1_label = mf_max + 0.168 * total_width

    fs_title = 1.85 * base_fontsize
    fs_subtitle = 1.25 * base_fontsize
    fs_electronic = 1.60 * base_fontsize
    fs_j = 1.35 * base_fontsize
    fs_tick = 1.30 * base_fontsize
    fs_column = 1.45 * base_fontsize
    fs_value = 1.30 * base_fontsize
    fs_arrow = 1.20 * base_fontsize

    f1_excited_str = as_frac2(float(tr.F1_excited))

    if title is None:
        field_text = rf"E={structure.E:g}\ \mathrm{{V/cm}}"
        if structure.B:
            field_text += rf",\ B={structure.B:g}\ \mathrm{{G}}"
        title = rf"TlF $X\rightarrow B$ level diagram at ${field_text}$"
    if title and created_figure:
        fig.suptitle(title, fontsize=fs_title, y=0.985)
    elif title:
        # The caller owns this figure -- a suptitle would overwrite theirs. Put
        # the title in the axes instead, offset above the subtitle drawn below.
        ax.annotate(
            title,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, 1.6 * fs_subtitle),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fs_title,
        )

    ax.text(
        0.5,
        1.0,
        rf"${tr.t.name}({tr.J_ground}),\ B:\ J'={tr.J_excited},\ "
        rf"F_1'={f1_excited_str},\ F'={tr.F_excited}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fs_subtitle,
    )

    # ---------------- excited manifold ----------------
    y_excited = _band_y(excited, *_Y_EXCITED_BAND, min_level_separation)

    for lv, y in zip(excited, y_excited, strict=True):
        fractions, colors = _level_segments(
            lv, (-1, +1), parity_map, show_excited_residual, residual_color
        )
        _draw_segmented_level(
            ax, lv.mF, y, fractions, colors, level_width, level_lw
        )

    for mF in range(-int(tr.F_excited), int(tr.F_excited) + 1):
        ax.text(
            mF,
            _Y_EXCITED_TICKS,
            rf"${_fmt_signed_mf(mF)}$",
            ha="center",
            va="center",
            fontsize=fs_tick,
        )
    ax.text(
        x_axis_name,
        _Y_EXCITED_TICKS,
        r"$m_F'$",
        ha="right",
        va="center",
        fontsize=fs_column,
    )

    y_excited_mid = 0.5 * (_Y_EXCITED_BAND[0] + _Y_EXCITED_BAND[1])
    ax.text(
        0.0,
        y_excited_mid + 0.035,
        r"$B\,^3\Pi_1\;(v'=0)$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=fs_electronic,
    )
    ax.text(
        0.0,
        y_excited_mid - 0.035,
        rf"$J'={tr.J_excited}$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=fs_j,
    )

    ax.text(x_f_label, _Y_EXCITED_TICKS, r"$F'$", ha="center", va="center",
            fontsize=fs_column)
    ax.text(x_f1_label, _Y_EXCITED_TICKS, r"$F_1'$", ha="center", va="center",
            fontsize=fs_column)
    ax.text(x_f_label, y_excited_mid, rf"${tr.F_excited}$", ha="center",
            va="center", fontsize=fs_value)
    ax.text(x_f1_label, y_excited_mid, rf"${f1_excited_str}$", ha="center",
            va="center", fontsize=fs_value)

    # ---------------- transition arrow ----------------
    ax.annotate(
        "",
        xy=(0.0, _Y_ARROW[1]),
        xytext=(0.0, _Y_ARROW[0]),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="black"),
    )
    ax.text(
        0.03 * total_width,
        0.5 * (_Y_ARROW[0] + _Y_ARROW[1]),
        rf"${tr.t.name}({tr.J_ground})$"
        + "\n"
        + rf"$F_1'={f1_excited_str},\ F'={tr.F_excited}$",
        ha="left",
        va="center",
        fontsize=fs_arrow,
    )

    # ---------------- ground manifold ----------------
    y_ground = _band_y(ground, *_Y_GROUND_BAND, min_level_separation)

    for lv, y in zip(ground, y_ground, strict=True):
        fractions, colors = _level_segments(
            lv, families, ground_map, show_ground_residual, residual_color
        )
        _draw_segmented_level(
            ax, lv.mF, y, fractions, colors, level_width, level_lw
        )

    ground_mfs = sorted({lv.mF for lv in ground})
    for mF in range(min(ground_mfs), max(ground_mfs) + 1):
        ax.text(
            mF,
            _Y_GROUND_TICKS,
            rf"${_fmt_signed_mf(mF)}$",
            ha="center",
            va="center",
            fontsize=fs_tick,
        )
    ax.text(x_axis_name, _Y_GROUND_TICKS, r"$m_F$", ha="right", va="center",
            fontsize=fs_column)

    y_ground_mid = 0.5 * (_Y_GROUND_BAND[0] + _Y_GROUND_BAND[1])
    ax.text(
        0.0,
        y_ground_mid + 0.035,
        r"$X\,^1\Sigma^+\;(v=0)$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=fs_electronic,
    )
    ax.text(
        0.0,
        y_ground_mid - 0.035,
        rf"$J={tr.J_ground}$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=fs_j,
    )

    ax.text(x_f_label, _Y_GROUND_HEADERS, r"$F$", ha="center", va="center",
            fontsize=fs_column)
    ax.text(x_f1_label, _Y_GROUND_HEADERS, r"$F_1$", ha="center", va="center",
            fontsize=fs_column)

    # The F / F1 column is a key, not an alignment: in a field the hyperfine
    # families interleave in energy, so anchoring each label to its levels
    # (their mean, or a bracket over their span) puts all of them on top of each
    # other. Space them evenly in family order instead and let the colour carry
    # the identification.
    family_label_y = _even_spacing(len(families), *_Y_GROUND_BAND)
    for fam, y_label in zip(families, family_label_y, strict=True):
        ax.text(
            x_f_label,
            y_label,
            rf"${fam[1]}$",
            ha="center",
            va="center",
            fontsize=fs_value,
            color=ground_map[fam],
        )

    for F1 in sorted({fam[0] for fam in families}, reverse=True):
        ys = [
            y
            for fam, y in zip(families, family_label_y, strict=True)
            if fam[0] == F1
        ]
        _draw_family_bracket(
            ax, x_f1_line, x_f1_label, ys, rf"${as_frac2(F1)}$", "black", fs_value
        )

    # ---------------- information panel ----------------
    if info_ax is not None:
        _render_info_panel(
            info_ax,
            _build_info_rows(
                structure,
                ground_colors=ground_map,
                parity_colors=parity_map,
                show_ground_residual=show_ground_residual,
                show_excited_residual=show_excited_residual,
            ),
            base_fontsize,
        )

    return TransitionLevelDiagram(
        fig=fig, ax=ax, info_ax=info_ax, structure=structure
    )
