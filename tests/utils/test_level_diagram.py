"""Tests for the field-dressed X -> B transition level diagram.

The reference numbers pinned here come from an independent, hand-rolled
calculation of the same physics (its own uncoupled X Hamiltonian built from the
molecular constants directly, sympy Clebsch-Gordans, an explicit R_10 Stark
operator), not from this implementation. They are therefore a genuine
cross-check rather than a snapshot of current behaviour.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from centrex_tlf.states import ElectronicState
from centrex_tlf.transitions import (
    OpticalTransition,
    OpticalTransitionType,
)
from centrex_tlf.utils.plotting import (
    _build_info_rows,
    _info_mF_groups,
    _level_segments,
    _n_tracking_steps,
    calculate_transition_level_structure,
    plot_transition_level_diagram,
)

# X energies at 170 V/cm, in MHz relative to (F1=3/2, F=1, mF=-1)
GROUND_REFERENCE_MHz = {
    (2.5, 3, -3): 0.305264531,
    (2.5, 3, -2): 0.952347938,
    (2.5, 3, -1): 1.165615301,
    (2.5, 3, 0): 1.177035194,
    (2.5, 2, -2): 0.275086254,
    (2.5, 2, -1): 0.939456926,
    (2.5, 2, 0): 1.156920897,
    (1.5, 2, -2): 0.040455314,
    (1.5, 2, -1): 0.746045098,
    (1.5, 2, 0): 0.740871545,
    (1.5, 1, -1): 0.0,
    (1.5, 1, 0): 0.711457307,
}

# B offsets at 170 V/cm, in MHz relative to the lower-parent mF'=0 level
EXCITED_REFERENCE_MHz = {
    (-1, -1): -41.395996,
    (0, -1): 0.0,
    (1, -1): -41.395996,
    (-1, +1): 58.957653,
    (0, +1): 17.589007,
    (1, +1): 58.957653,
}

ZERO_FIELD_PARITY_SPLITTING_MHz = 17.705


@pytest.fixture(scope="module")
def p2_structure():
    return calculate_transition_level_structure(
        E=170.0, J_ground=2, branch="P", F1_excited=1.5, F_excited=1
    )


def test_ground_energies_match_reference(p2_structure):
    levels = {(lv.F1, lv.F, lv.mF): lv for lv in p2_structure.ground}
    reference = levels[(1.5, 1, -1)].energy_MHz

    for key, expected in GROUND_REFERENCE_MHz.items():
        got = levels[key].energy_MHz - reference
        assert got == pytest.approx(expected, abs=1e-6), key


def test_excited_energies_match_reference(p2_structure):
    levels = {(lv.mF, lv.P): lv for lv in p2_structure.excited}
    reference = min(
        lv.energy_MHz for lv in p2_structure.excited if lv.mF == 0 and lv.P == -1
    )

    for key, expected in EXCITED_REFERENCE_MHz.items():
        got = levels[key].energy_MHz - reference
        assert got == pytest.approx(expected, abs=1e-4), key


def test_zero_field_parity_splitting(p2_structure):
    splittings = list(p2_structure.zero_field_parity_splitting_MHz.values())
    assert splittings  # one entry per mF'
    for value in splittings:
        assert value == pytest.approx(ZERO_FIELD_PARITY_SPLITTING_MHz, abs=1e-3)


def test_excited_parity_mixing(p2_structure):
    """|mF'|=1 is strongly parity mixed at 170 V/cm; mF'=0 is not."""
    lower_1 = min(
        (lv for lv in p2_structure.excited if lv.mF == 1),
        key=lambda lv: lv.energy_MHz,
    )
    assert lower_1.character[-1] == pytest.approx(0.588, abs=2e-3)
    assert lower_1.character[+1] == pytest.approx(0.412, abs=2e-3)

    lower_0 = min(
        (lv for lv in p2_structure.excited if lv.mF == 0),
        key=lambda lv: lv.energy_MHz,
    )
    assert lower_0.character[-1] > 0.99


def test_character_is_normalized_and_residual_small(p2_structure):
    for level in p2_structure.ground + p2_structure.excited:
        assert sum(level.character.values()) == pytest.approx(1.0)
        assert 0.0 <= level.residual < 0.01


def test_structure_shape(p2_structure):
    ground = p2_structure.ground
    excited = p2_structure.excited

    # J=2 in X: (F1, F) = (5/2, 3), (5/2, 2), (3/2, 2), (3/2, 1) -> 7+5+5+3 levels
    assert len(ground) == 20
    assert p2_structure.ground_parents == ((2.5, 3), (2.5, 2), (1.5, 2), (1.5, 1))
    assert {lv.electronic_state for lv in ground} == {ElectronicState.X}
    assert all(lv.P is None for lv in ground)

    # F'=1, both Lambda-doublet parity parents
    assert len(excited) == 6
    assert {lv.electronic_state for lv in excited} == {ElectronicState.B}
    assert {lv.P for lv in excited} == {-1, +1}


def test_transition_object_and_explicit_spec_agree():
    transition = OpticalTransition(
        t=OpticalTransitionType.P, J_ground=2, F1_excited=1.5, F_excited=1
    )
    from_object = calculate_transition_level_structure(transition, E=170.0)
    from_spec = calculate_transition_level_structure(
        E=170.0, J_ground=2, J_excited=1, F1_excited=1.5, F_excited=1
    )

    a = np.array([lv.energy_MHz for lv in from_object.ground])
    b = np.array([lv.energy_MHz for lv in from_spec.ground])
    np.testing.assert_allclose(a, b)


def test_tracking_is_step_size_independent():
    coarse = calculate_transition_level_structure(
        E=170.0,
        J_ground=2,
        branch="P",
        F1_excited=1.5,
        F_excited=1,
        max_tracking_step_V_cm=5.0,
    )
    fine = calculate_transition_level_structure(
        E=170.0,
        J_ground=2,
        branch="P",
        F1_excited=1.5,
        F_excited=1,
        max_tracking_step_V_cm=0.5,
    )
    for a, b in zip(coarse.ground, fine.ground, strict=True):
        assert (a.F1, a.F, a.mF) == (b.F1, b.F, b.mF)
        assert a.energy_MHz == pytest.approx(b.energy_MHz, abs=1e-6)


def test_zero_magnetic_field_leaves_mf_pairs_degenerate():
    structure = calculate_transition_level_structure(
        E=170.0, J_ground=2, branch="P", F1_excited=1.5, F_excited=1
    )
    by_key = {(lv.F1, lv.F, lv.mF): lv.energy_MHz for lv in structure.ground}
    for (F1, F, mF), energy in by_key.items():
        if mF > 0:
            assert energy == pytest.approx(by_key[(F1, F, -mF)], abs=1e-9)


def test_magnetic_field_splits_mf_pairs():
    structure = calculate_transition_level_structure(
        E=170.0, J_ground=2, branch="P", F1_excited=1.5, F_excited=1, B=20.0
    )
    by_key = {(lv.F1, lv.F, lv.mF): lv.energy_MHz for lv in structure.ground}
    assert by_key[(1.5, 1, +1)] != pytest.approx(by_key[(1.5, 1, -1)], abs=1e-3)


@pytest.mark.parametrize(
    "spec",
    [
        dict(J_ground=2, branch="P", F1_excited=1.5, F_excited=1),
        dict(J_ground=2, branch="R", F1_excited=3.5, F_excited=3),
        dict(J_ground=0, branch="R", F1_excited=1.5, F_excited=2),
        dict(J_ground=1, J_excited=1, F1_excited=1.5, F_excited=2),
    ],
)
def test_plot_runs_for_a_range_of_transitions(spec):
    result = plot_transition_level_diagram(E=170.0, **spec)
    try:
        assert result.info_ax is not None
        assert result.structure.transition.F_excited == spec["F_excited"]
        # every level drawn, in both manifolds
        assert len(result.structure.excited) == 2 * (2 * spec["F_excited"] + 1)
    finally:
        plt.close(result.fig)


def test_plot_accepts_precomputed_structure(p2_structure):
    result = plot_transition_level_diagram(structure=p2_structure)
    try:
        assert result.structure is p2_structure
    finally:
        plt.close(result.fig)


def test_plot_into_supplied_axes(p2_structure):
    fig, axes = plt.subplots(1, 2)
    try:
        result = plot_transition_level_diagram(
            structure=p2_structure, ax=axes[0], info_ax=axes[1]
        )
        assert result.fig is fig
        assert result.ax is axes[0]
        assert result.info_ax is axes[1]
    finally:
        plt.close(fig)


def test_conflicting_specifications_raise(p2_structure):
    transition = OpticalTransition(
        t=OpticalTransitionType.P, J_ground=2, F1_excited=1.5, F_excited=1
    )
    with pytest.raises(ValueError):
        calculate_transition_level_structure(transition, J_ground=2)
    with pytest.raises(ValueError):
        calculate_transition_level_structure(J_ground=2, F1_excited=1.5)
    with pytest.raises(ValueError):
        calculate_transition_level_structure(
            J_ground=2, branch="P", J_excited=3, F1_excited=1.5, F_excited=1
        )
    with pytest.raises(ValueError):
        plot_transition_level_diagram(structure=p2_structure, J_ground=2)


def test_tracking_steps_follow_the_magnetic_field():
    """A large B must drive the ramp too, not just sit inside the 8-step floor."""
    assert _n_tracking_steps(0.0, 500.0, 2.0, 1.0) == 500
    assert _n_tracking_steps(0.0, 500.0, 2.0, 5.0) == 100
    # E still dominates when it demands more steps, and the small-B placeholder
    # keeps the previous 8-step floor.
    assert _n_tracking_steps(170.0, 500.0, 2.0, 10.0) == 85
    assert _n_tracking_steps(0.0, 1e-5, 2.0, 1.0) == 8
    assert _n_tracking_steps(0.0, 0.0, 2.0, 1.0) == 1


def test_tracking_is_step_size_independent_in_B():
    common = dict(E=0.0, J_ground=2, branch="P", F1_excited=1.5, F_excited=1, B=200.0)
    coarse = calculate_transition_level_structure(**common, max_tracking_step_G=5.0)
    fine = calculate_transition_level_structure(**common, max_tracking_step_G=0.5)
    for a, b in zip(coarse.ground, fine.ground, strict=True):
        assert (a.F1, a.F, a.mF) == (b.F1, b.F, b.mF)
        assert a.energy_MHz == pytest.approx(b.energy_MHz, abs=1e-6)


def test_info_mF_groups_are_signed_only_when_B_splits_them():
    zero_field = calculate_transition_level_structure(
        E=170.0, J_ground=2, branch="P", F1_excited=1.5, F_excited=1
    )
    groups = _info_mF_groups(zero_field.excited, zero_field.B)
    assert [label for label, _ in groups] == [r"m_F'=0", r"|m_F'|=1"]

    split = calculate_transition_level_structure(
        E=170.0, J_ground=2, branch="P", F1_excited=1.5, F_excited=1, B=20.0
    )
    groups = _info_mF_groups(split.excited, split.B)
    # every mF gets its own signed row; none is silently dropped
    assert [label for label, _ in groups] == [r"m_F'=-1", r"m_F'=0", r"m_F'=+1"]
    assert sum(len(levels) for _, levels in groups) == len(split.excited)


def test_info_rows_cover_both_mF_signs_at_nonzero_B():
    structure = calculate_transition_level_structure(
        E=170.0, J_ground=2, branch="P", F1_excited=1.5, F_excited=1, B=20.0
    )
    rows = _build_info_rows(
        structure,
        ground_colors={fam: "C0" for fam in structure.ground_parents},
        parity_colors={-1: "C0", +1: "C1"},
        show_ground_residual=False,
        show_excited_residual=False,
    )
    lines = [text for kind, text in rows if kind == "line"]
    for sign in ("m_F'=-1", "m_F'=+1"):
        assert any(sign in line for line in lines), sign


def test_level_segments_scale_character_by_the_residual():
    """The residual segment is appended on top of a character that sums to 1."""
    from centrex_tlf.utils.plotting import DressedLevel

    level = DressedLevel(
        electronic_state=ElectronicState.B,
        F1=1.5,
        F=1,
        mF=0,
        P=-1,
        energy_MHz=0.0,
        character={-1: 0.75, +1: 0.25},
        residual=0.2,
    )
    colors = {-1: "C0", +1: "C1"}

    fractions, _ = _level_segments(level, (-1, +1), colors, True, "0.7")
    assert sum(fractions) == pytest.approx(1.0)
    assert fractions == pytest.approx([0.6, 0.2, 0.2])

    # without a residual segment the character is drawn at full width
    fractions, _ = _level_segments(level, (-1, +1), colors, False, "0.7")
    assert fractions == pytest.approx([0.75, 0.25])


def test_dressed_levels_are_hashable(p2_structure):
    assert len(set(p2_structure.excited)) == len(p2_structure.excited)
    assert len(set(p2_structure.ground)) == len(p2_structure.ground)
    hash(p2_structure)


def test_plot_into_supplied_axes_keeps_the_figure_suptitle(p2_structure):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("caller's own title")
    try:
        plot_transition_level_diagram(
            structure=p2_structure, ax=axes[0], info_ax=axes[1]
        )
        assert fig.get_suptitle() == "caller's own title"
    finally:
        plt.close(fig)
