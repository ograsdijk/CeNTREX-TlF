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
