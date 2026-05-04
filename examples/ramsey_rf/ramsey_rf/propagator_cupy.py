"""V7: dense midpoint propagation using CuPy on a CUDA GPU.

This is an optional benchmark-only path. Importing this module requires a CuPy
CUDA wheel such as `cupy-cuda13x`; on Windows with `uv --with`, include the
CUDA component wheels with `uv run --with 'cupy-cuda13x[ctk]' ...`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import site
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt


_DLL_HANDLES: list[object] = []
_ORIGINAL_GETSITEPACKAGES = site.getsitepackages


def _add_nvidia_cuda_dll_dirs() -> None:
    """Make NVIDIA CUDA component-wheel DLLs discoverable on Windows."""
    if os.name != "nt":
        return
    overlay_site_packages = [
        str(path)
        for path in map(Path, sys.path)
        if path.name == "site-packages" and (path / "nvidia").exists()
    ]
    if overlay_site_packages:

        def getsitepackages_with_uv_overlay() -> list[str]:
            paths = list(_ORIGINAL_GETSITEPACKAGES())
            for path in overlay_site_packages:
                if path not in paths:
                    paths.append(path)
            return paths

        site.getsitepackages = getsitepackages_with_uv_overlay

    for root in map(Path, sys.path):
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        seen: set[Path] = set()
        candidates = list(nvidia_root.glob("cu*/bin/x86_64"))
        candidates.extend(nvidia_root.glob("*/bin"))
        for bin_dir in candidates:
            if bin_dir.is_dir():
                resolved = bin_dir.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    _DLL_HANDLES.append(os.add_dll_directory(str(resolved)))


_add_nvidia_cuda_dll_dirs()

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - optional benchmark dependency
    raise ImportError(
        "V7 requires CuPy. Run with `uv run --with 'cupy-cuda13x[ctk]' ...` "
        "or install a CUDA-compatible CuPy wheel."
    ) from exc

from .propagator import HMidFn, PropagationResult


@dataclass
class CuPyDeviceInfo:
    """Small, serializable snapshot of the active CuPy CUDA device."""

    device_id: int
    name: str
    runtime_version: int


@dataclass
class CuPyPropagationResult(PropagationResult):
    device_info: CuPyDeviceInfo


def get_cupy_device_info() -> CuPyDeviceInfo:
    """Return active CUDA device info and force runtime initialization."""
    runtime = cp.cuda.runtime
    device_id = int(runtime.getDevice())
    props = runtime.getDeviceProperties(device_id)
    name_raw = props["name"]
    name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
    return CuPyDeviceInfo(
        device_id=device_id,
        name=name,
        runtime_version=int(runtime.runtimeGetVersion()),
    )


def warm_up_cupy_eigh(n: int = 32) -> None:
    """Compile/initialize CUDA libraries before timing a benchmark."""
    H = cp.eye(n, dtype=cp.complex128)
    _D, _V = cp.linalg.eigh(H)
    cp.cuda.Stream.null.synchronize()


def _step_eigh_cupy(
    Psi_gpu: "cp.ndarray",
    t_k: float,
    t_kp1: float,
    H_mid_fn: HMidFn,
) -> "cp.ndarray":
    t_mid = 0.5 * (t_k + t_kp1)
    dt = t_kp1 - t_k
    H_mid = cp.asarray(H_mid_fn(t_mid), dtype=cp.complex128)
    D, V = cp.linalg.eigh(H_mid)
    tmp = V.conj().T @ Psi_gpu
    tmp *= cp.exp((-1j) * D * dt)[:, None]
    return V @ tmp


def propagate_midpoint_cupy(
    Psi0: npt.NDArray[np.complex128],
    t_grid: npt.NDArray[np.float64],
    H_at_t: HMidFn,
    *,
    warm_up: bool = True,
    store_norm: bool = True,
    store_snapshots: bool = False,
) -> CuPyPropagationResult:
    """Propagate with dense `cupy.linalg.eigh` and GPU-resident state vectors.

    Hamiltonian assembly remains on the CPU through `H_at_t`; each midpoint
    Hamiltonian is copied to the GPU for diagonalization. `Psi_final` is copied
    back to NumPy only once at the end.
    """
    Psi_np = np.asarray(Psi0, dtype=np.complex128)
    if Psi_np.ndim != 2:
        raise ValueError(f"Psi0 must be 2-D (N, K); got shape {Psi_np.shape}")
    n_times = t_grid.shape[0]
    if n_times < 2:
        raise ValueError("t_grid must have at least 2 points")

    device_info = get_cupy_device_info()
    if warm_up:
        warm_up_cupy_eigh()
    Psi_gpu = cp.asarray(Psi_np)

    norm_trace: Optional[npt.NDArray[np.float64]] = None
    snapshots: Optional[npt.NDArray[np.complex128]] = None
    if store_norm:
        norm_trace = np.empty((n_times, Psi_np.shape[1]), dtype=np.float64)
        norm_trace[0, :] = np.sum(np.abs(Psi_np) ** 2, axis=0)
    if store_snapshots:
        snapshots = np.empty((n_times, Psi_np.shape[0], Psi_np.shape[1]), dtype=np.complex128)
        snapshots[0] = Psi_np

    for k in range(n_times - 1):
        Psi_gpu = _step_eigh_cupy(Psi_gpu, float(t_grid[k]), float(t_grid[k + 1]), H_at_t)
        if norm_trace is not None:
            norm_trace[k + 1, :] = cp.asnumpy(cp.sum(cp.abs(Psi_gpu) ** 2, axis=0))
        if snapshots is not None:
            snapshots[k + 1] = cp.asnumpy(Psi_gpu)

    cp.cuda.Stream.null.synchronize()
    Psi_final = cp.asnumpy(Psi_gpu)
    return CuPyPropagationResult(
        Psi_final=Psi_final,
        norm_trace=norm_trace,
        snapshots=snapshots,
        t_grid=np.asarray(t_grid, dtype=np.float64),
        device_info=device_info,
    )
