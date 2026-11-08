"""Tests for the transition field grid (`centrex_tlf.transitions.field_grid`).

The grid is small on purpose (ground J <= 2, 0 to 50 V/cm). Exact references
come from independent paths through the package: dense diagonalization of the
whole manifold, `calculate_transition_level_structure`, and
`generate_ED_ME_mixed_state` on explicitly built dressed states.
"""

import json

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from centrex_tlf import transitions
from centrex_tlf.constants import X_B_REFERENCE_FREQUENCY_HZ
from centrex_tlf.couplings import polarization_X, polarization_Z
from centrex_tlf.hamiltonian import (
    field_matrices_B,
    field_matrices_X,
    generate_ED_ME_mixed_state,
    track_levels,
)
from centrex_tlf.states import CoupledState
from centrex_tlf.transitions import (
    TransitionFieldGrid,
    build_optical_field_grid,
    spherical_components,
)
from centrex_tlf.transitions.field_grid import _UNIT_POLARIZATION
from centrex_tlf.utils.plotting import calculate_transition_level_structure

E_MAX = 50.0
FREQ_TOL_HZ = 1e5


@pytest.fixture(scope="module")
def grid():
    return build_optical_field_grid(2, E_MAX)


def _between_stored(grid, n=4):
    """Fields halfway between stored points, picking the widest intervals."""
    E = grid.E_fields
    widths = np.diff(E)
    order = np.argsort(widths)[::-1][:n]
    return [float(E[i] + 0.37 * widths[i]) for i in sorted(order)]


def _parity(p):
    return (-1) ** p.J if p.P is None else p.P


def test_manifolds_and_selection_rule(grid):
    assert grid.parameters["lower"] == ["X", 0, 4]
    assert grid.parameters["upper"] == ["B", 1, 5]
    assert max(p.J for p in grid.lower_parents) <= 2
    for k in range(grid.n_components):
        lo = grid.lower_parents[grid.component_lower[k]]
        up = grid.upper_parents[grid.component_upper[k]]
        # E along z: Delta m = q is exact
        assert up.mF - lo.mF == grid.component_q[k]
        assert abs(up.J - lo.J) <= 2


@pytest.mark.parametrize("which", ["lower", "upper"])
def test_energies_between_stored_points_match_full_diagonalization(grid, which):
    matrices = field_matrices_X(0, 4) if which == "lower" else field_matrices_B(1, 5)
    parents = grid.lower_parents if which == "lower" else grid.upper_parents
    for E in _between_stored(grid):
        exact = track_levels(matrices, [E])
        # per-block tracking reproduces the whole spectrum of the dense matrix
        dense = eigvalsh(matrices.H0 + E * matrices.HSz)
        np.testing.assert_allclose(np.sort(exact.energies_Hz[0]), dense, rtol=0, atol=1.0)
        interp = grid.level_energies(E)[0 if which == "lower" else 1]
        want = np.array([exact.energies_Hz[0, exact.index(p)] for p in parents])
        assert np.max(np.abs(interp - want)) < FREQ_TOL_HZ / 2, E


def test_component_frequencies_between_stored_points(grid):
    lower = field_matrices_X(0, 4)
    upper = field_matrices_B(1, 5)
    for E in _between_stored(grid, n=2):
        tl = track_levels(lower, [E])
        tu = track_levels(upper, [E])
        el = np.array([tl.energies_Hz[0, tl.index(p)] for p in grid.lower_parents])
        eu = np.array([tu.energies_Hz[0, tu.index(p)] for p in grid.upper_parents])
        exact = grid.frequency_origin_Hz + eu[grid.component_upper] - el[grid.component_lower]
        got = grid.at(E)["frequency_Hz"].to_numpy()
        assert np.max(np.abs(got - exact)) < FREQ_TOL_HZ


def test_labels_and_energies_match_level_structure(grid):
    E = 40.0
    tr = transitions.P2_F1_3o2_F1
    structure = calculate_transition_level_structure(tr, E=E)
    e_lower, e_upper = grid.level_energies(E)
    lower_index = {p.key: i for i, p in enumerate(grid.lower_parents)}
    upper_index = {p.key: i for i, p in enumerate(grid.upper_parents)}

    for lv in structure.ground:
        key = (2, lv.F1, lv.F, lv.mF)
        assert e_lower[lower_index[key]] / 1e6 == pytest.approx(lv.energy_MHz, abs=0.05)
    for lv in structure.excited:
        key = (1, lv.F1, lv.F, lv.mF, lv.P)
        # the structure uses B J=1..3 and the grid J=1..5; both are converged
        assert e_upper[upper_index[key]] / 1e6 == pytest.approx(lv.energy_MHz, abs=0.05)

    comps = grid.components(tr, E)
    assert comps and all(c.label == tr for c in comps)
    assert {(c.upper.J, c.upper.F1, c.upper.F) for c in comps} == {(1, 1.5, 1)}
    assert all(c.lower.J == 2 for c in comps)
    assert {c.upper.P for c in comps} == {-1, +1}


def _dressed_state(tracked, level):
    vec = tracked.vector(level, 0)
    basis = tracked.matrices.basis
    return CoupledState([(complex(a), basis[i]) for i, a in enumerate(vec) if abs(a) > 1e-14])


def test_dressed_coupling_matches_mixed_state_matrix_element(grid):
    k_field = int(np.argmin(np.abs(grid.E_fields - 25.0)))
    E = float(grid.E_fields[k_field])  # a stored field: no interpolation involved
    tl = track_levels(field_matrices_X(0, 4), [E])
    tu = track_levels(field_matrices_B(1, 5), [E])

    strongest = int(np.argmax(grid.strengths[k_field]))
    mixing_only = [
        k
        for k in range(grid.n_components)
        if _parity(grid.lower_parents[grid.component_lower[k]])
        == _parity(grid.upper_parents[grid.component_upper[k]])
    ]
    mixing = max(mixing_only, key=lambda k: grid.strengths[k_field, k])
    for k in (strongest, mixing):
        lo = grid.lower_parents[grid.component_lower[k]]
        up = grid.upper_parents[grid.component_upper[k]]
        q = int(grid.component_q[k])
        ground = _dressed_state(tl, tl.index(lo))
        excited = _dressed_state(tu, tu.index(up))
        me = generate_ED_ME_mixed_state(
            excited, ground, pol_vec=_UNIT_POLARIZATION[q], normalize_pol=False
        )
        assert abs(me) ** 2 == pytest.approx(grid.strengths[k_field, k], rel=1e-6, abs=1e-12)
    assert grid.strengths[k_field, mixing] > 1e-3 * grid.strengths[k_field, strongest]


def test_forbidden_couplings_vanish_at_zero_field_and_appear_with_mixing(grid):
    assert grid.E_fields[0] == 0.0 and grid.E_fields[-1] == E_MAX
    s0, s50 = grid.strengths[0], grid.strengths[-1]
    smax = grid.strengths.max()
    lower = [grid.lower_parents[i] for i in grid.component_lower]
    upper = [grid.upper_parents[i] for i in grid.component_upper]
    forbidden = np.array(
        [
            _parity(lo) == _parity(up) or abs(lo.F - up.F) > 1
            for lo, up in zip(lower, upper, strict=True)
        ]
    )
    assert forbidden.any()
    assert np.max(s0[forbidden]) < 1e-12 * smax

    # Known mixing-only pair (Delta F = 2): X J=2 F=3 mF=-1 -> B J'=1 F1'=3/2 F'=1 mF'=0
    (k,) = [
        k
        for k in range(grid.n_components)
        if lower[k].key == (2, 2.5, 3, -1) and upper[k].key == (1, 1.5, 1, 0, -1)
    ]
    assert grid.component_q[k] == 1
    assert s0[k] < 1e-12 * smax
    assert s50[k] > 1e-4 * smax
    assert grid.labels[k] == transitions.P2_F1_3o2_F1


def test_reference_line_at_zero_field(grid):
    comps = [
        c
        for c in grid.components(transitions.R2_F1_7o2_F3, 0.0)
        if transitions.R2_F1_7o2_F3.P_excited == c.upper.P
    ]
    f = np.array([c.frequency_Hz for c in comps])
    w = np.array([c.strength for c in comps])
    centroid = np.sum(w * f) / np.sum(w)
    assert centroid == pytest.approx(X_B_REFERENCE_FREQUENCY_HZ, abs=1.0)
    assert centroid / 1e12 == pytest.approx(1103.435082708, abs=1e-9)
    # the hyperfine components of the line sit within about a MHz of it
    assert np.max(np.abs(f - centroid)) < 2e6


def test_strength_weighting(grid):
    E = 30.0
    base = grid.at(E)["strength"].to_numpy()
    q = grid.component_q
    # pi light only drives q = 0, with unit weight
    s_z = grid.strength(polarization_Z, E)
    np.testing.assert_allclose(s_z, np.where(q == 0, base, 0.0))
    # x light: |eps_+-1|^2 = 1/2
    s_x = grid.strength(polarization_X, E)
    np.testing.assert_allclose(s_x, np.where(q != 0, base / 2, 0.0))
    assert spherical_components(polarization_X)[0] == 0
    # population mapping by key; unnamed levels get zero weight
    s_j2 = grid.strength(polarization_Z, E, ground_population={(2, 2.5, 3, 0): 0.5})
    lower = [grid.lower_parents[i].key for i in grid.component_lower]
    mask = np.array([key == (2, 2.5, 3, 0) for key in lower])
    np.testing.assert_allclose(s_j2, np.where(mask, 0.5 * s_z, 0.0))


def test_flags_are_well_formed(grid):
    for flag in grid.flags:
        assert flag.reason in {"purity", "strength", "min_spacing"}
        assert 0.0 <= flag.E_low < flag.E_high <= E_MAX


def test_save_load_round_trip_and_refusals(grid, tmp_path):
    path = grid.save(tmp_path / "grid.npz")
    loaded = TransitionFieldGrid.load(path, expected={"J_ground_max": 2})
    assert loaded.labels == grid.labels
    assert loaded.lower_parents == grid.lower_parents
    assert loaded.upper_parents == grid.upper_parents
    assert loaded.flags == grid.flags
    for E in (0.0, 17.3, E_MAX):
        a, b = grid.at(E), loaded.at(E)
        assert a.equals(b)
    c = loaded.components(transitions.R2_F1_7o2_F3, 10.0)
    assert c and isinstance(c[0].label, transitions.OpticalTransition)

    with pytest.raises(ValueError, match="J_ground_max"):
        TransitionFieldGrid.load(path, expected={"J_ground_max": 3})

    # a file written by another package version is stale
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays["meta"]))
    meta["package_version"] = "0.0.0"
    arrays["meta"] = np.asarray(json.dumps(meta))
    stale = tmp_path / "stale.npz"
    np.savez_compressed(stale, **arrays)
    with pytest.raises(ValueError, match="rebuild"):
        TransitionFieldGrid.load(stale)


def test_out_of_range_field_raises(grid):
    with pytest.raises(ValueError):
        grid.at(E_MAX + 1.0)


def test_core_is_generic_same_manifold_x_to_x():
    """X -> X with the q=0 Stark operator: a microwave-like use of the core."""
    X = field_matrices_X(0, 2)
    D0 = X.HSz / np.max(np.abs(X.HSz))
    g = TransitionFieldGrid.build(
        X,
        X,
        {0: D0},
        E_max_V_cm=20.0,
        frequency_origin_Hz=0.0,
        pair_filter=lambda lo, up, q: up.J == lo.J + 1,
        frequency_tol_Hz=1e3,
        kind="microwave-test",
    )
    assert g.parameters["same_manifold"] is True
    assert set(g.component_q.tolist()) == {0}
    lower = [g.lower_parents[i] for i in g.component_lower]
    upper = [g.upper_parents[i] for i in g.component_upper]
    assert all(up.J == lo.J + 1 and up.mF == lo.mF for lo, up in zip(lower, upper, strict=False))
    assert {lo.J for lo in lower} == {0, 1}

    E = 7.7
    tracked = track_levels(X, [E])
    exact = np.array(
        [
            tracked.energies_Hz[0, tracked.index(up)] - tracked.energies_Hz[0, tracked.index(lo)]
            for lo, up in zip(lower, upper, strict=False)
        ]
    )
    got = g.at(E)["frequency_Hz"].to_numpy()
    assert np.max(np.abs(got - exact)) < 1e3
    # J=0 -> J=1 near 2 B_rot ~ 13.3 GHz
    j01 = got[[lo.J == 0 for lo in lower]]
    assert np.all((j01 > 13.2e9) & (j01 < 13.5e9))
