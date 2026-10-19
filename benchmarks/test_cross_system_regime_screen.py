import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks import run_cross_system_regime_screen as screen


@dataclass(frozen=True)
class TinyRegime:
    name: str
    detuning_rad_s: float = 0.0
    optical_rabi_rad_s: float = 1.0
    microwave_rabi_rad_s: float = 0.0


def test_synthetic_accuracy_all_products_and_two_regime_checkpoints(monkeypatch):
    rho = np.diag(np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.complex128))
    photons = np.asarray([0.25])
    reference = {
        "full": rho,
        "photon_integral": photons,
        "dopri5_full": rho,
        "dopri5_photon_integral": photons,
        "tsit5_full": rho,
        "tsit5_photon_integral": photons,
    }
    fixture = SimpleNamespace(
        spec=SimpleNamespace(name="synthetic", t_span=(0.0, 1.0), rtol=1e-7,
                             atol=1e-9, dt=0.1),
        prepared=SimpleNamespace(parameters=SimpleNamespace(base_parameters={"rabi": 0.0,
                                                                                "detuning": 0.0})),
        rc_case=None, rho0=rho, selected_indices_1=(0,), selected_indices_4=(0, 1, 2, 3),
        group_indices=(1, 2), metadata={},
    )
    monkeypatch.setattr(screen, "_resource_sample", lambda: {
        "cpu_percent_samples": [0.0, 0.0, 0.0], "cpu_median_percent": 0.0,
        "available_memory_bytes": 1_000_000, "competing_python_processes": [],
    })
    monkeypatch.setattr(screen, "_reference", lambda _fixture, _regime: (reference, {}))

    def fake_solve(_fixture, _regime, _mode, product, **_kwargs):
        values = {
            "full": rho,
            "photon_integral": photons,
            "populations": np.real(np.diag(rho)),
            "selected_1": np.asarray([0.4]),
            "selected_4": np.asarray([0.4, 0.3, 0.2, 0.1]),
            "group_sum": np.asarray([0.5]),
            "full_and_integral": {"full": rho, "photon_integral": photons},
        }
        return values[product], 0.001, {"n_steps": 1}

    monkeypatch.setattr(screen, "_static_product", fake_solve)
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
        output = Path(temporary)
        for name in ("regime_a", "regime_b"):
            result = screen._run_regime(fixture, TinyRegime(name), output,
                                        fixture_build_seconds=0.0)
            assert result.is_file()
            regime_dir = output / name
            assert len(list((regime_dir / "cells").glob("*.npz"))) == len(screen.KERNELS) * len(screen.PRODUCTS)
            assert (regime_dir / "progress_after_077.json").is_file()
            with np.load(regime_dir / f"{name}.npz") as data:
                assert "reference_dopri5_full" in data
                assert "reference_tsit5_full" in data
                assert "structured__full_and_integral__accepted__rho" in data
                assert "structured__full__accepted" in data
