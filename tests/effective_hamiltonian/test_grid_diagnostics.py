import warnings

import numpy as np

from centrex_tlf.effective_hamiltonian.models import InterpolatedEffectivePatch
from centrex_tlf.effective_hamiltonian.operator_bundle import OperatorBundle
from centrex_tlf.effective_hamiltonian.preparation import (
    _operator_grid_variation_diagnostics,
    _warn_large_operator_grid_variation,
)


def _bundle(scale: float) -> OperatorBundle:
    h_internal = np.diag([0.0, scale]).astype(np.complex128)
    h_opt = np.array([[0.0, scale], [scale, 0.0]], dtype=np.complex128)
    h_det = np.diag([0.0, 1.0]).astype(np.complex128)
    c_array = np.zeros((0, 2, 2), dtype=np.complex128)
    zero = np.zeros((2, 2), dtype=np.complex128)
    return OperatorBundle(
        electric_field=np.array([0.0, 0.0, scale], dtype=np.float64),
        magnetic_field=np.array([0.0, 0.0, 1e-5], dtype=np.float64),
        omega_reference=0.0,
        h_internal=h_internal,
        h_opt=h_opt,
        h_det=h_det,
        c_array=c_array,
        excited_indices=np.array([1], dtype=np.int64),
        loss_operator=zero,
        h_full_internal=h_internal,
        h_lab_internal=h_internal,
        dissipator_superop=np.zeros((4, 4), dtype=np.complex128),
        jump_rate_operator_override=zero,
    )


def _patch(scale: float) -> InterpolatedEffectivePatch:
    return InterpolatedEffectivePatch(
        electric_field=np.array([0.0, 0.0, scale], dtype=np.float64),
        aligned_basis_vectors=np.eye(2, dtype=np.complex128),
        bundle=_bundle(scale),
    )


def test_operator_grid_variation_diagnostics_reports_adjacent_jumps():
    diagnostics = _operator_grid_variation_diagnostics(
        [0.0, 1.0],
        [_patch(1.0), _patch(10.0)],
    )

    assert diagnostics["metric"] == "adjacent_relative_frobenius_norm"
    assert diagnostics["max_operator"] in {"h_internal", "h_opt"}
    assert diagnostics["max_interval"] == (0.0, 1.0)
    assert diagnostics["max_relative_variation"] > 0.8


def test_operator_grid_variation_warning_threshold():
    diagnostics = _operator_grid_variation_diagnostics(
        [0.0, 1.0],
        [_patch(1.0), _patch(10.0)],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_large_operator_grid_variation(diagnostics, threshold=0.1)

    assert len(caught) == 1
    assert "large adjacent variation" in str(caught[0].message)


def test_operator_grid_variation_warning_can_be_disabled():
    diagnostics = _operator_grid_variation_diagnostics(
        [0.0, 1.0],
        [_patch(1.0), _patch(10.0)],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_large_operator_grid_variation(diagnostics, threshold=None)

    assert caught == []
