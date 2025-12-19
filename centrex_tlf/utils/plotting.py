from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties

from centrex_tlf.states import ElectronicState

__all__ = [
    "plot_level_diagram",
    "filter_levels_with_decay_or_coupling",
    "combine_decay_only_states",
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
