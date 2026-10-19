import numpy as np
import pytest

from centrex_tlf.lindblad.state_layout import PackedHermitianLayout


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("complex_input", [False, True])
def test_pack_batch_matches_scalar_and_owns_snapshot(n, complex_input):
    layout = PackedHermitianLayout(n)
    rng = np.random.default_rng(7)
    matrices = rng.normal(size=(3, 8, n, n))
    if complex_input:
        matrices = matrices + 1j * rng.normal(size=matrices.shape)
    matrices = matrices[:, ::2]
    expected = np.array([[layout.pack(matrix) for matrix in group] for group in matrices])
    packed = layout.pack_batch(matrices)
    np.testing.assert_array_equal(packed, expected)
    np.testing.assert_array_equal(layout.pack_batch(matrices[0, 0]), expected[0, 0])
    assert packed.flags.c_contiguous and packed.flags.owndata
    assert packed.dtype == np.float64
    matrices[...] = 0
    np.testing.assert_array_equal(packed, expected)
    for shape in [(0, n, n), (3, 0, n, n)]:
        assert layout.pack_batch(np.empty(shape)).shape == (*shape[:-2], n*n)
    for invalid in [0., np.zeros(n), np.zeros((3, n, n+1))]:
        with pytest.raises(ValueError, match="final matrix dimensions"):
            layout.pack_batch(invalid)


def test_pack_batch_chunk_boundary_preserves_complex_signs():
    layout = PackedHermitianLayout(3)
    rng = np.random.default_rng(8)
    matrices = rng.normal(size=(2051, 3, 3)) + 1j * rng.normal(size=(2051, 3, 3))
    matrices = matrices + matrices.conj().swapaxes(-1, -2)
    packed = layout.pack_batch(matrices)
    for index in [0, 2047, 2048, 2050]:
        np.testing.assert_array_equal(packed[index], layout.pack(matrices[index]))
    np.testing.assert_array_equal(layout.unpack_batch(packed), matrices)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_unpack_batch_matches_scalar_for_strided_and_empty_arrays(n):
    layout = PackedHermitianLayout(n)
    packed = np.random.default_rng(1).normal(size=(3, 8, n * n))[:, ::2]
    expected = np.array([[layout.unpack(row) for row in group] for group in packed])
    np.testing.assert_array_equal(layout.unpack_batch(packed), expected)
    np.testing.assert_array_equal(layout.unpack_batch(packed[0, 0]), expected[0, 0])
    for shape in [(0, n * n), (3, 0, n * n)]:
        answer = layout.unpack_batch(np.empty(shape))
        assert answer.shape == (*shape[:-1], n, n)
        assert answer.dtype == np.complex128
    for invalid in [0.0, np.zeros((3, n * n + 1))]:
        with pytest.raises(ValueError, match="final packed dimension"):
            layout.unpack_batch(invalid)
