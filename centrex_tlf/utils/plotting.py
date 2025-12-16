from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from centrex_tlf.states import ElectronicState


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

    # Validate matrix shapes
    if coupling_mats is not None:
        for k, M in enumerate(coupling_mats):
            M = np.asarray(M)
            if M.shape != (n, n):
                raise ValueError(
                    f"coupling_mats[{k}] has shape {M.shape}, expected ({n},{n})"
                )

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

    # ---- couplings ----
    if coupling_mats is not None:
        for M in coupling_mats:
            A = np.abs(np.asarray(M))
            mask = A > coupling_threshold
            np.fill_diagonal(mask, False)

            involved = mask.any(axis=0) | mask.any(axis=1)
            keep |= involved & filter_mask

    # ---- decays ----
    if BR is not None:
        mask = BR > decay_threshold
        np.fill_diagonal(mask, False)

        if br_is_final_initial:
            # BR[final, initial]
            initial_involved = mask.any(axis=0)
            final_involved = mask.any(axis=1)
        else:
            # BR[initial, final]
            initial_involved = mask.any(axis=1)
            final_involved = mask.any(axis=0)

        involved = initial_involved | final_involved
        keep |= involved & filter_mask

    kept_idx = np.nonzero(keep)[0]

    states_kept = [states[i] for i in kept_idx]

    coupling_kept = None
    if coupling_mats is not None:
        coupling_kept = [
            np.asarray(M)[np.ix_(kept_idx, kept_idx)] for M in coupling_mats
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
def f_maybe(v) -> float | None:
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

    BR0 = None
    if branching_ratio is not None:
        BR0 = np.asarray(branching_ratio, dtype=float)

    # helper to compute involvement (coupled/decays)
    def compute_involvement(n: int, coupling_mats_in, BR_in):
        coupled = np.zeros(n, dtype=bool)
        if coupling_mats_in is not None:
            for M in coupling_mats_in:
                A = np.abs(np.asarray(M))
                mask = A > coupling_threshold
                np.fill_diagonal(mask, False)
                coupled |= mask.any(axis=0) | mask.any(axis=1)

        decays = np.zeros(n, dtype=bool)
        if BR_in is not None:
            mask = BR_in > decay_threshold
            np.fill_diagonal(mask, False)
            if br_is_final_initial:
                initial_involved = mask.any(axis=0)
                final_involved = mask.any(axis=1)
            else:
                initial_involved = mask.any(axis=1)
                final_involved = mask.any(axis=0)
            decays |= initial_involved | final_involved

        return coupled, decays

    # ---- optionally drop isolated before combining ----
    if drop_isolated_when_combining:
        coupled_pre, decays_pre = compute_involvement(n_pre, coupling_mats, BR0)
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

        if coupling_mats is not None:
            coupling_mats = [
                np.asarray(M)[np.ix_(keep_idx, keep_idx)] for M in coupling_mats
            ]
        if BR0 is not None:
            BR0 = np.asarray(BR0)[np.ix_(keep_idx, keep_idx)]

    # ---- identify decay-only groups eligible for collapsing ----
    n_post = len(states)
    coupled, decays = compute_involvement(n_post, coupling_mats, BR0)
    decay_only = decays & (~coupled)

    # apply optional J filter for collapsing
    if combine_for_J is not None:
        Jset = {float(J) for J in combine_for_J}
        decay_only = np.array(
            [decay_only[i] and (float(states[i].J) in Jset) for i in range(n_post)]
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
        for ii in idxs:
            to_collapse[ii] = True

    if not to_collapse.any():
        return list(states), None if coupling_mats is None else list(coupling_mats), BR0

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
        for r in range(n_post):
            nr = old_to_new[r]
            for c in range(n_post):
                v = BR0[r, c]
                if v != 0.0:
                    nc = old_to_new[c]
                    BR1[nr, nc] += float(v)

    coupling1 = None
    if coupling_mats is not None:
        coupling1 = []
        for M in coupling_mats:
            M = np.asarray(M)
            M1 = np.zeros((n1, n1), dtype=M.dtype)
            for r in range(n_post):
                nr = old_to_new[r]
                for c in range(n_post):
                    v = M[r, c]
                    if v != 0:
                        nc = old_to_new[c]
                        M1[nr, nc] += v
            coupling1.append(M1)

    return new_states, coupling1, BR1


# ---------------- main plotting function (assumes combining done externally) ----------------
def plot_level_diagram(
    states: Sequence[Any],  # CoupledBasisState, imported elsewhere
    coupling_mats: Sequence[np.ndarray] | None = None,
    branching_ratio: np.ndarray | None = None,
    *,
    ax: plt.Axes | None = None,
    coupling_threshold: float = 0.0,
    decay_threshold: float = 0.0,
    br_is_final_initial: bool = True,  # BR[final, initial] if True
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
    annotate_electronic_headers: bool = True,
    # label geometry
    mf_label_offset: float = 1.4,
    j_label_offset: float = 3.5,
    mf_label_fontsize: int = 9,
    j_label_fontsize: int = 11,
    right_labels_pad_x: float = 0.25,
    f1_extra_pad_x: float = 1.2,
    right_label_fontsize: int = 10,
    # matching tolerance
    j_match_tol: float = 1e-9,
) -> plt.Axes:
    n0 = len(states)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))

    # ---------------- validation ----------------
    if coupling_mats is not None:
        for k, M in enumerate(coupling_mats):
            M = np.asarray(M)
            if M.shape != (n0, n0):
                raise ValueError(
                    f"coupling_mats[{k}] has shape {M.shape}, expected ({n0},{n0})"
                )

    BR0 = None
    if branching_ratio is not None:
        BR0 = np.asarray(branching_ratio, dtype=float)
        if BR0.shape != (n0, n0):
            raise ValueError(
                f"`branching_ratio` has shape {BR0.shape}, expected ({n0},{n0})"
            )

    # ---------------- involvement helper (uses function args) ----------------
    def compute_involvement(n: int, coupling_mats_in, BR_in):
        coupled = np.zeros(n, dtype=bool)
        if coupling_mats_in is not None:
            for M in coupling_mats_in:
                A = np.abs(np.asarray(M))
                mask = A > coupling_threshold
                np.fill_diagonal(mask, False)
                coupled |= mask.any(axis=0) | mask.any(axis=1)

        decays = np.zeros(n, dtype=bool)
        if BR_in is not None:
            mask = BR_in > decay_threshold
            np.fill_diagonal(mask, False)
            if br_is_final_initial:
                initial_involved = mask.any(axis=0)
                final_involved = mask.any(axis=1)
            else:
                initial_involved = mask.any(axis=1)
                final_involved = mask.any(axis=0)
            decays |= initial_involved | final_involved

        return coupled, decays

    # ---------------- grouping for plotting ----------------
    n = len(states)

    by_row: dict[
        tuple[ElectronicState, float, float | None, float | None], list[int]
    ] = defaultdict(list)
    for i, st in enumerate(states):
        elec = st.electronic_state
        J = float(st.J)
        F1 = f_maybe(getattr(st, "F1", None))
        F = f_maybe(getattr(st, "F", None))
        by_row[(elec, J, F1, F)].append(i)

    elec_order = [ElectronicState.X, ElectronicState.B]

    # ---------------- vertical placement ----------------
    y_base: dict[tuple[ElectronicState, float, float | None, float | None], float] = {}
    current_y = 0.0

    for elec in elec_order:
        Js = sorted({J for (e, J, _, _) in by_row if e == elec})
        for J in Js:
            F1s = sort_none_last(
                {F1 for (e, JJ, F1, _) in by_row if e == elec and JJ == J}
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

    # ---------------- J block x layout (width ∝ expected mF span) ----------------
    all_Js = sorted({float(st.J) for st in states})

    mf_range_by_J: dict[float, np.ndarray] = {}
    ncols_by_J: dict[float, int] = {}
    for J in all_Js:
        mf_max = int(round(J + 1))
        mf_vals = np.arange(-mf_max, mf_max + 1, 1, dtype=float)
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

    # ---------------- color mapping per J ----------------
    cmap = plt.get_cmap("tab10")
    J_to_color: dict[float, tuple] = {}
    for idx, J in enumerate(all_Js):
        J_to_color[J] = cmap(idx % getattr(cmap, "N", 10))

    # ---------------- coordinates ----------------
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

    # ---------------- draw levels ----------------
    for i in range(n):
        J_i = float(states[i].J)
        color = J_to_color.get(J_i, "k")
        ax.plot(
            [x[i] - level_halfwidth, x[i] + level_halfwidth],
            [y[i], y[i]],
            lw=level_lw,
            color=color,
        )

    # ---------------- per-(elec, J) mF and J labels ----------------
    for elec in elec_order:
        Js = sorted({float(st.J) for st in states if st.electronic_state == elec})
        for J in Js:
            idx_ej = [
                i
                for i, st in enumerate(states)
                if st.electronic_state == elec and float(st.J) == J
            ]
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

            if place_above:
                y_mf = y_max + mf_label_offset
                y_J = y_max + j_label_offset
                va_mf = "bottom"
                va_J = "bottom"
            else:
                y_mf = y_min - mf_label_offset
                y_J = y_min - j_label_offset
                va_mf = "top"
                va_J = "top"

            if has_real_levels:
                for mf in sorted(mf_to_dx.keys()):
                    ax.text(
                        x0 + mf_to_dx[mf],
                        y_mf,
                        f"${as_signed_frac2(mf)}$",
                        ha="center",
                        va=va_mf,
                        fontsize=mf_label_fontsize,
                    )

            ax.text(
                x_center,
                y_J,
                f"$J={as_frac2(J)}$",
                ha="center",
                va=va_J,
                fontsize=j_label_fontsize,
            )

    # ---------------- right-side labels: only for real F/F1 ----------------
    for elec in elec_order:
        Js = sorted({J for (e, J, _, _) in by_row if e == elec})
        for J in Js:
            row_keys = [
                (e, JJ, F1, F)
                for (e, JJ, F1, F) in by_row.keys()
                if e == elec and JJ == J
            ]
            if not row_keys:
                continue

            x0 = J_to_x0[J]
            mf_to_dx = mf_grid_by_J[J]
            x_F = x0 + max(mf_to_dx.values()) + level_halfwidth + right_labels_pad_x
            x_F1 = x_F + f1_extra_pad_x

            rows_by_F1: dict[float, list[float]] = defaultdict(list)

            for e, JJ, F1, F in row_keys:
                if F1 is None or F is None:
                    continue

                y_row = y_base[(e, JJ, F1, F)]
                ax.text(
                    x_F,
                    y_row,
                    f"$F={as_frac2(F)}$",
                    ha="left",
                    va="center",
                    fontsize=right_label_fontsize,
                )
                rows_by_F1[F1].append(y_row)

            for F1, ys in rows_by_F1.items():
                ys_sorted = sorted(ys)
                y_f1 = 0.5 * (ys_sorted[0] + ys_sorted[-1])
                ax.text(
                    x_F1,
                    y_f1,
                    f"$F_1={as_frac2(F1)}$",
                    ha="left",
                    va="center",
                    fontsize=right_label_fontsize,
                )

    # ---------------- electronic headers ----------------
    if annotate_electronic_headers:
        for elec in elec_order:
            idxE = [i for i, st in enumerate(states) if st.electronic_state == elec]
            if not idxE:
                continue
            ax.text(
                np.mean(x[idxE]),
                float(np.max(y[idxE])) + 5.0,
                f"Electronic state {elec.value}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    # ---------------- couplings ----------------
    if coupling_mats:
        styles = ["-", "--", ":", "-."]
        pair_color_cache: dict[tuple[float, float], tuple] = {}
        for mi, M in enumerate(coupling_mats):
            style = styles[mi % len(styles)]
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(M[i, j]) <= coupling_threshold:
                        continue
                    Ji = float(states[i].J)
                    Jj = float(states[j].J)
                    if Ji == Jj:
                        color = J_to_color.get(Ji, "k")
                    else:
                        pair = (min(Ji, Jj), max(Ji, Jj))
                        if pair not in pair_color_cache:
                            c1 = np.array(mcolors.to_rgb(J_to_color.get(pair[0], "k")))
                            c2 = np.array(mcolors.to_rgb(J_to_color.get(pair[1], "k")))
                            blended = tuple(((c1 + c2) / 2.0).tolist())
                            pair_color_cache[pair] = blended
                        color = pair_color_cache[pair]

                    ax.plot(
                        [x[i], x[j]],
                        [y[i], y[j]],
                        linestyle=style,
                        lw=coupling_lw,
                        alpha=coupling_alpha,
                        color=color,
                    )

    # ---------------- decays ----------------
    if BR0 is not None:
        BR = BR0
        for i in range(n):  # initial
            for j in range(n):  # final
                br = float(BR[j, i]) if br_is_final_initial else float(BR[i, j])
                if br <= decay_threshold:
                    continue
                if y[i] <= y[j]:
                    continue
                ax.annotate(
                    "",
                    xy=(x[j], y[j]),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        linestyle="--",
                        lw=decay_lw * (0.5 + 2.0 * br),
                        alpha=decay_alpha,
                    ),
                )

    # ---------------- cosmetics ----------------
    ax.set_xlabel(r"$m_F$ (− → +)")
    ax.set_ylabel("schematic level layout")
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.1, y=0.25)

    return ax
