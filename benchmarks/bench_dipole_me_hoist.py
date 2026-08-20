"""Count redundant Omega-basis transforms in the dipole matrix-element path.

Decision gate for the "hoist loop-invariant state transforms" plan. The claim
under test is that `generate_ED_ME_mixed_state` re-transforms the same mixed
states to the Omega basis on every call because its callers invoke it from
nested loops in which one of the two arguments is loop-invariant.

The bare-basis matrix elements themselves are NOT the issue -- `ED_ME_coupled`,
`ED_ME_uncoupled`, `_ED_ME_uncoupled_omega` and `angular_part` all already carry
`@lru_cache(maxsize=1e6)`. What is uncached is
`State.transform_to_omega_basis`, which rebuilds a state by repeated `+=`.

Decision rule, fixed before running:
  * distinct/total ratio near 1  -> transforms are already effectively unique,
    the hoist buys nothing, record and stop.
  * distinct/total ratio ~1/n    -> the hoist is real, proceed.

Run:  uv run python benchmarks/bench_dipole_me_hoist.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.hamiltonian import matrix_elements_electric_dipole as edm
from centrex_tlf.states import states as states_mod

RESULTS_DIR = Path(__file__).with_name("dipole_me_hoist_results")


class TransformCounter:
    """Instrument the two functions the plan targets.

    Counts total calls against distinct argument objects. Keying on `id()` is
    sound here because every state involved stays alive for the whole build --
    the lists that own them are still in scope -- so ids cannot be recycled
    underneath the counter. The live states are retained explicitly anyway to
    make that guarantee unconditional.
    """

    def __init__(self) -> None:
        self.me_calls = 0
        self.transform_calls = 0
        self.transform_ids: Counter[int] = Counter()
        self._keepalive: list[Any] = []

    def __enter__(self) -> "TransformCounter":
        self._orig_me = edm.generate_ED_ME_mixed_state
        self._orig_transform = states_mod.State.transform_to_omega_basis

        counter = self

        def counting_me(*args: Any, **kwargs: Any) -> complex:
            counter.me_calls += 1
            return counter._orig_me(*args, **kwargs)

        def counting_transform(self_state):  # type: ignore[no-untyped-def]
            counter.transform_calls += 1
            counter.transform_ids[id(self_state)] += 1
            counter._keepalive.append(self_state)
            return counter._orig_transform(self_state)

        edm.generate_ED_ME_mixed_state = counting_me  # type: ignore[assignment]
        states_mod.State.transform_to_omega_basis = counting_transform  # type: ignore[assignment]
        # The reduced-Hamiltonian module imported the symbol directly, so patch
        # its binding too or site A goes uncounted.
        import centrex_tlf.hamiltonian.reduced_hamiltonian as rh

        self._rh = rh
        self._orig_rh_me = rh.generate_ED_ME_mixed_state
        rh.generate_ED_ME_mixed_state = counting_me  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        edm.generate_ED_ME_mixed_state = self._orig_me  # type: ignore[assignment]
        states_mod.State.transform_to_omega_basis = self._orig_transform  # type: ignore[assignment]
        self._rh.generate_ED_ME_mixed_state = self._orig_rh_me  # type: ignore[assignment]
        self._keepalive.clear()

    @property
    def distinct_transforms(self) -> int:
        return len(self.transform_ids)

    @property
    def max_repeat(self) -> int:
        return max(self.transform_ids.values()) if self.transform_ids else 0


def build(transition, pol_vec, **kwargs) -> Any:
    pol = couplings.Polarization(np.asarray(pol_vec, dtype=np.complex128), name="P")
    selectors = couplings.generate_transition_selectors(
        transitions=[transition], polarizations=[[pol]]
    )
    return lindblad.generate_OBE_system_transitions(
        [transition],
        selectors,
        E=np.array([0.0, 0.0, 171.6]),
        B=np.array([0.0, 0.0, 1e-3]),
        **kwargs,
    )


CASES: list[dict[str, Any]] = [
    {
        "name": "A: R0_F1_3o2_F2 (65-state)",
        "transition": transitions.R0_F1_3o2_F2,
        "pol": [0.0, 0.0, 1.0],
        "kwargs": {},
    },
    {
        "name": "B: R2_F1_7o2_F3 (r2-style)",
        "transition": transitions.R2_F1_7o2_F3,
        "pol": [1.0, 0.0, 0.0],
        "kwargs": {"retain_opposite_parity_levels": True},
    },
    {
        "name": "C: Q1_F1_3o2_F2 compact (20-state)",
        "transition": transitions.Q1_F1_3o2_F2,
        "pol": [0.0, 0.0, 1.0],
        "kwargs": {"qn_compact": True},
    },
]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []

    for case in CASES:
        # Warm the lru_caches so the counted run measures the steady state a
        # repeated build (e.g. a field grid) actually sees.
        build(case["transition"], case["pol"], **case["kwargs"])

        t0 = time.perf_counter()
        with TransformCounter() as counter:
            system = build(case["transition"], case["pol"], **case["kwargs"])
        elapsed = time.perf_counter() - t0

        distinct = counter.distinct_transforms
        total = counter.transform_calls
        row = {
            "case": case["name"],
            "n_states": len(system.QN),
            "build_s": round(elapsed, 3),
            "me_calls": counter.me_calls,
            "transform_calls": total,
            "distinct_transforms": distinct,
            "redundancy": round(total / distinct, 2) if distinct else float("nan"),
            "max_repeat_one_state": counter.max_repeat,
        }
        rows.append(row)
        print(
            f"{row['case']:<36} n={row['n_states']:>4}  build={row['build_s']:>6.3f}s  "
            f"ME calls={row['me_calls']:>6}  transforms={total:>6}  "
            f"distinct={distinct:>5}  redundancy={row['redundancy']}x  "
            f"worst state repeated {row['max_repeat_one_state']}x"
        )

    (RESULTS_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
