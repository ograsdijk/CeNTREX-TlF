from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["PackedHermitianLayout"]


@dataclass(frozen=True)
class PackedHermitianLayout:
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("layout dimension must be positive")

    @property
    def packed_len(self) -> int:
        return self.n * self.n

    def diagonal_index(self, i: int) -> int:
        self._check_index(i)
        return i

    def upper_real_index(self, i: int, j: int) -> int:
        self._check_pair(i, j)
        return self.n + 2 * self._upper_offset(i, j)

    def upper_imag_index(self, i: int, j: int) -> int:
        self._check_pair(i, j)
        return self.n + 2 * self._upper_offset(i, j) + 1

    def pack(self, matrix: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
        arr = np.asarray(matrix, dtype=np.complex128)
        if arr.shape != (self.n, self.n):
            raise ValueError(f"expected matrix shape {(self.n, self.n)}, got {arr.shape}")
        packed = np.empty(self.packed_len, dtype=np.float64)
        packed[: self.n] = np.real(np.diag(arr))
        cursor = self.n
        for i in range(self.n):
            for j in range(i + 1, self.n):
                packed[cursor] = float(np.real(arr[i, j]))
                packed[cursor + 1] = float(np.imag(arr[i, j]))
                cursor += 2
        return packed

    def unpack(self, packed: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        arr = np.asarray(packed, dtype=np.float64)
        if arr.shape != (self.packed_len,):
            raise ValueError(f"expected packed vector length {self.packed_len}, got {arr.shape}")
        matrix = np.zeros((self.n, self.n), dtype=np.complex128)
        for i in range(self.n):
            matrix[i, i] = complex(arr[i], 0.0)
        cursor = self.n
        for i in range(self.n):
            for j in range(i + 1, self.n):
                value = complex(arr[cursor], arr[cursor + 1])
                matrix[i, j] = value
                matrix[j, i] = np.conjugate(value)
                cursor += 2
        return matrix

    def pack_batch(self, matrices: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
        """Pack ``(..., n, n)`` matrices into an owned ``(..., n*n)`` array.

        Reuse this snapshot as initial input to batch/scan solvers until the
        initial states change. Like :meth:`pack`, it reads the real diagonal
        and complex upper triangle; it does not validate Hermiticity.
        """
        arr = np.asarray(matrices)
        if arr.ndim < 2 or arr.shape[-2:] != (self.n, self.n):
            raise ValueError(f"expected final matrix dimensions {(self.n, self.n)}, got {arr.shape}")
        result = np.empty((*arr.shape[:-2], self.packed_len), dtype=np.float64)
        flat = result.reshape(-1, self.packed_len)
        matrices_flat = arr.reshape(-1, self.n, self.n)
        ui, uj = np.triu_indices(self.n, 1)
        chunk = max(1, min(2048, 128_000_000 // (16 * self.n * self.n)))
        for start in range(0, len(flat), chunk):
            block = matrices_flat[start : start + chunk]
            packed = flat[start : start + chunk]
            packed[:, : self.n] = np.diagonal(block, axis1=-2, axis2=-1).real
            upper = block[:, ui, uj]
            packed[:, self.n :: 2] = upper.real
            packed[:, self.n + 1 :: 2] = upper.imag
        return result

    def unpack_batch(self, packed: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """Unpack arrays shaped ``(..., n*n)`` in bounded vectorized batches."""
        arr = np.asarray(packed, dtype=np.float64)
        if arr.ndim < 1 or arr.shape[-1] != self.packed_len:
            raise ValueError(f"expected final packed dimension {self.packed_len}, got {arr.shape}")
        n = self.n
        result = np.empty((*arr.shape[:-1], n, n), dtype=np.complex128)
        di = np.arange(n)
        ui, uj = np.triu_indices(n, 1)
        # Bound temporary complex upper triangles for large output batches.
        chunk = max(1, min(2048, 128_000_000 // (16 * n * n)))
        flat = result.reshape(-1, n, n)
        packed_flat = arr.reshape(-1, n * n)
        for start in range(0, len(flat), chunk):
            block = packed_flat[start : start + chunk]
            matrices = flat[start : start + chunk]
            matrices[:, di, di] = block[:, :n]
            upper = block[:, n::2] + 1j * block[:, n + 1 :: 2]
            matrices[:, ui, uj] = upper
            matrices[:, uj, ui] = upper.conj()
        return result

    def _upper_offset(self, i: int, j: int) -> int:
        offset = 0
        for row in range(i):
            offset += self.n - row - 1
        return offset + (j - i - 1)

    def _check_index(self, i: int) -> None:
        if not (0 <= i < self.n):
            raise IndexError(f"index {i} out of bounds for size {self.n}")

    def _check_pair(self, i: int, j: int) -> None:
        self._check_index(i)
        self._check_index(j)
        if not i < j:
            raise ValueError(f"expected upper-triangle pair with i < j, got {(i, j)}")
