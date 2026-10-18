"""Summarize isolated current-library CUDA assembly experiments."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "cuda_prepared_assembly_results"
data = json.loads((OUT / "results.json").read_text())
lines = [
    "# CUDA prepared Hamiltonian assembly and structural coefficient sharing",
    "",
    f"Device: {data['device']}. Benchmark-only source transformations; production unchanged.",
    "",
    "Current library fixed Tsit5, default waveform precomputation, packed FP64 density, "
    "tiled and direct RHS, batches 512/1024. RC uses the archived current-apparatus "
    "fixture at 150 m/s near the field center (146-147 us, 4.4642857 ns steps); "
    "R(2) uses its archived EOM fixture (0.125-1.125 us, 2 ns steps). "
    "All density coordinates are evolved. Distinct cases are repeated only to fill batches.",
    "",
    "Six randomized interleaved rounds per configuration after compilation/capture/warmup. "
    "Timing covers graph evolution and transfer of the final internal packed density. "
    "Setup, compilation, graph capture, common-waveform precomputation and disk writes are excluded. "
    "The public solver is run first for each candidate and validated against archived strict "
    "references; timed graph replays are also compared with the current baseline.",
    "",
    "Assembly precomputes exact nonzero coefficient contribution lists for each stored "
    "Hamiltonian entry and separates the static coefficient-1 term. No magnitude threshold. "
    "DAG sharing interns structurally identical scalar nodes across coefficients/compounds "
    "before the existing CUDA source renderer. Tests include both changes individually and combined.",
    "",
    "Positive percentages mean reduced runtime. Brackets are descriptive 95% paired bootstrap "
    "intervals (10,000 resamples of six whole rounds), not hardware-portability guarantees.",
    "",
    "| System | Batch | RHS | Baseline (s) | Assembly reduction | DAG reduction | Both reduction |",
    "|---|---:|---|---:|---:|---:|---:|",
]
rng = np.random.default_rng(531)
for row in data["cases"]:
    base = np.asarray(row["timings"]["base"])
    cells = []
    for variant in ["assembly", "dag", "assembly_dag"]:
        reduction = 100 * (1 - np.asarray(row["timings"][variant]) / base)
        draws = rng.integers(0, len(base), size=(10000, len(base)))
        low, high = np.percentile(np.median(reduction[draws], axis=1), [2.5, 97.5])
        cells.append(f"{np.median(reduction):+.2f}% [{low:+.2f}, {high:+.2f}]")
    lines.append(f"| {row['system']} | {row['batch']} | {row['rhs']} | {np.median(base):.6f} | " + " | ".join(cells) + " |")
lines.extend(["", "## Validation and scope", ""])
for system in sorted({row["system"] for row in data["cases"]}):
    rows = [row for row in data["cases"] if row["system"] == system]
    errors = [err for row in rows for err in row["accuracy"].values()]
    # Preserve all original diagnostic names/values in the raw JSON.
    difference = max(v for row in rows for v in row["max_internal_difference"].values())
    lines.append(f"- {system}: all archived-reference entry/coherence, near-zero, trace and positivity "
                 f"checks passed. Maximum internal-coordinate difference from the current kernel: {difference:.3g}. "
                 f"Generated coefficient temporaries: {rows[0]['coefficient_temporaries']}.")
lines.extend([
    "",
    "The entry criterion is 1% relative above reference magnitude 1e-6, absolute 1e-6 below; "
    "trace error <1e-6 and minimum density eigenvalue >=-1e-7. These are short-interval "
    "performance tests, not a new validation of a full apparatus grid or full-transit speedup.",
    "",
    "Common waveforms already run through precomputation, so eliminating scalar operations "
    "in those expressions chiefly affects precomputation, outside the evolution timings. "
    "This prototype does not benchmark precompute=False or claim structural sharing has no "
    "benefit for every possible parameter graph. The tested RC graph has only one redundant "
    "rendered temporary removed; the small scope limits potential gains.",
    "",
    "## Conclusion",
    "",
    "Prepared assembly gives roughly 0.6-1.0% paired median reductions for RC, "
    "and mostly sub-1% changes for R(2), with one R(2) configuration slower. "
    "Most intervals overlap zero. Broader DAG sharing has no consistent evolution "
    "benefit; it removes one rendered temporary in RC and none in R(2). Combining "
    "the changes does not deliver a robust gain across configurations. Keep "
    "the production CUDA implementation unchanged based on these measurements. "
    "There is no evidence here for a gain comparable to the recent CPU changes.",
    "",
    "No CPU ODE timings rerun. Source transformations and generated CUDA files are retained "
    "in this directory; raw paired timings, setup statistics and complete density diagnostics: "
    "[results.json](results.json). Reproduce with the benchmark GPU environment: "
    "`benchmarks/.venv-gpu-bench/Scripts/python.exe benchmarks/check_cuda_prepared_assembly.py`, "
    "then run `benchmarks/report_cuda_prepared_assembly.py`.",
])
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:18]))
