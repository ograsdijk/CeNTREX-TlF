"""Bounded CUDA reconstruction from a reusable real Lindblad eigenbasis.

Imported only when requested. Factorization and initial-state solves remain on
the CPU; no approximation or eigensystem truncation is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FullProjection:
    n: int
    w: np.ndarray
    basis: np.ndarray
    real: np.ndarray
    positive: np.ndarray
    keep: np.ndarray
    sinks: np.ndarray
    feed_basis: np.ndarray
    coefficients: np.ndarray
    initial_sinks: np.ndarray
    resident: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_propagator(cls, propagator: Any, initial: Any) -> FullProjection:
        coefficients, sinks = propagator._coefficients(initial)
        if not hasattr(propagator, "_cuda_full_cache"):
            propagator._cuda_full_cache = {}
        return cls(
            propagator.layout.n,
            propagator.w,
            propagator.basis,
            propagator.real,
            propagator.positive,
            propagator.keep,
            propagator.sinks,
            propagator.feed_basis,
            coefficients,
            sinks,
            propagator._cuda_full_cache,
        )

    def evaluate(self, times: np.ndarray, *, complex_output: bool = False) -> np.ndarray:
        from .dense import _torch_device

        torch = _torch_device("cuda")
        device = torch.cuda.current_device()
        if device not in self.resident:
            self.resident[device] = {
                name: torch.as_tensor(getattr(self, name), device="cuda")
                for name in ["w", "basis", "real", "positive", "keep", "sinks", "feed_basis"]
            }
        g = self.resident[device]
        count, nt = self.coefficients.shape[1], len(times)
        shape = (count, nt, self.n, self.n) if complex_output else (count, nt, self.n**2)
        answer = np.empty(shape, dtype=complex if complex_output else float)
        if not count or not nt:
            return answer
        flat = answer.reshape(count * nt, -1)
        # Bound intermediates even for large manifolds and many initial states.
        per_case = 48 * len(self.w) + (32 if complex_output else 16) * self.n**2
        chunk = max(1, min(2048, 128_000_000 // max(1, per_case)))
        with torch.inference_mode():
            C = torch.as_tensor(self.coefficients, device="cuda")
            s0 = torch.as_tensor(self.initial_sinks, device="cuda")
            r, p = g["real"], g["positive"]
            ui, uj = np.triu_indices(self.n, 1)
            ui, uj = torch.as_tensor(ui, device="cuda"), torch.as_tensor(uj, device="cuda")
            di = torch.arange(self.n, device="cuda")
            for start in range(0, count * nt, chunk):
                cases = np.arange(start, min(start + chunk, count * nt))
                initials = torch.as_tensor(cases // nt, device="cuda")
                unique, inverse = np.unique(times[cases % nt], return_inverse=True)
                ts = torch.as_tensor(unique, device="cuda")
                inverse = torch.as_tensor(inverse, device="cuda")
                width = len(cases)
                modes = torch.empty((len(self.w), width), dtype=torch.float64, device="cuda")
                integrals = torch.empty_like(modes) if len(self.sinks) else None
                for indices, paired in [(r, False), (p, True)]:
                    w = g["w"][indices, None]
                    z = w * ts[None]
                    expz = torch.exp(z)[:, inverse]
                    amplitudes = C[indices[:, None], initials[None]]
                    if paired:
                        amplitudes = amplitudes - 1j * C[(indices + 1)[:, None], initials[None]]
                    value = amplitudes * expz
                    modes[indices] = value.real
                    if paired:
                        modes[indices + 1] = -value.imag
                    if integrals is not None:
                        safe = torch.where(w != 0, w, 1)
                        phi = torch.where(
                            abs(z) < 1e-7, ts[None] * (1 + z / 2 + z * z / 6), torch.expm1(z) / safe
                        )[:, inverse]
                        value = amplitudes * phi
                        integrals[indices] = value.real
                        if paired:
                            integrals[indices + 1] = -value.imag
                block = torch.zeros((self.n**2, width), dtype=torch.float64, device="cuda")
                if len(self.w):
                    block[g["keep"]] = g["basis"] @ modes
                if len(self.sinks):
                    block[g["sinks"]] = g["feed_basis"] @ integrals + s0[:, initials]
                if complex_output:
                    matrix = torch.zeros(
                        (self.n, self.n, width), dtype=torch.complex128, device="cuda"
                    )
                    matrix[di, di] = block[: self.n].to(torch.complex128)
                    upper = block[self.n :: 2] + 1j * block[self.n + 1 :: 2]
                    matrix[ui, uj], matrix[uj, ui] = upper, upper.conj()
                    block = matrix.reshape(self.n**2, width)
                flat[start : start + width] = block.T.contiguous().cpu().numpy()
        return answer
