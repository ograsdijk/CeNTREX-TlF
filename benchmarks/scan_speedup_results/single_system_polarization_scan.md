# Single-System 2D Polarization+Detuning Scan (r2 in a static E field)

Produced by `benchmarks/bench_single_system_polarization_scan.py`. Prototype replacing the per-fz rebuild loop of `examples/lindblad/r2_peak_ratio_vs_z_polarization.ipynb` with ONE OBE system whose X and Z polarization amplitudes are runtime symbols.

## Design

- One system built with `generate_transition_selectors(..., polarizations=[[pol_X, pol_Z]])` (pol_X=[1,0,0], pol_Z=[0,0,1]) and `generate_OBE_system_transitions([R2_F1_7o2_F3], selectors, qn_compact=True, E=[0,0,171.6], B=[0,0,1e-5], retain_opposite_parity_levels=True, normalize_pol=True)` -- all other settings identical to `diagnose_step_size.build_system` / the notebook.
- The selector gets TWO polarization amplitude symbols (`PX0`, `PZ0`); the symbolic Hamiltonian carries `(PX0*Omega/main + PZ0*Omega/main)/2` terms with separate X and Z coupling matrices, `main` = main coupling for the FIRST polarization (X).
- Base runtime parameters `rabi`, `detuning`, `px`, `pz`; bindings: coupling symbol -> rabi, delta -> detuning, PX0 -> px, PZ0 -> pz.
- One prepared Rust plan; one `parameter_scan` over a 360x3 (detuning, px, pz) table (10 fz values x 36 detunings, -5..30 MHz in 1 MHz steps; fz and detuning are NOT a Cartesian product in (px, pz, detuning) space, hence `parameter_scan` with an explicit table rather than `grid_scan`). Output `photon_integral` with weights Gamma on the 14 B-manifold levels, `output_when='final'`, dopri5 / expanded_sparse, reltol=1e-7, abstol=1e-9, default threads.

## Normalization physics (verified numerically)

The per-fz reference (notebook) bakes the mixed polarization `eps = sqrt(1-fz) X + sqrt(fz) Z` into a single coupling matrix and computes its rabi from ITS OWN main coupling. Because `power_to_rabi_rectangular_beam` is linear in its coupling argument, `Omega_ref/main_ref = E_field*D/hbar` is fz-independent; moreover the Z component does not couple the mF=0 -> mF'=1 main states, so `main_ref(fz) = sqrt(1-fz)*main_X` EXACTLY. Binding a constant `rabi = power_to_rabi_rectangular_beam(P, |main_X|, wx, wy)` and setting `px = sqrt(1-fz)`, `pz = sqrt(fz)` therefore reproduces the reference field exactly (coupling matrices are linear in the polarization vector). Measured:

- main_X = -0.233755+0.000000j, rabi = 2pi x 0.3164 MHz (constant across fz)
- max over fz of |main_ref(fz) - sqrt(1-fz)*main_X| = 5.551e-17 (machine precision)
- max over fz of |Omega_ref/main_ref - Omega_X/main_X| = 3.725e-09 rad/s (constant-rabi claim confirmed; no per-fz rescaling needed)
- filled numeric H at fz=0.05, detuning 5 MHz: state ordering identical: True; entrywise max |H_single - H_ref| = 6.985e-10 rad/s = 4.70e-16 of the largest coupling (1.488e+06 rad/s).

## Validation (photon-integral detuning curves)

Per-fz reference systems built exactly as the notebook (single mixed polarization vector, own rabi) and scanned over the same detunings at the same tolerances. Gates: fz in {0, 0.01, 0.2}; the other fz values come for free from the benchmark loop and are reported too.

| fz | gate | max abs diff | max rel diff | peaks single (MHz) | peaks ref (MHz) | opp/normal single | opp/normal ref | elements thresholded in ref only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | x | 0.00e+00 | 0.00e+00 | 0, 25 | 0, 25 | 0.887128 | 0.887128 | 0 |
| 0.0001 |  | 2.10e-06 | 5.11e-06 | 0, 25 | 0, 25 | 0.886721 | 0.886721 | 4 |
| 0.0003 |  | 6.30e-06 | 1.53e-05 | 0, 25 | 0, 25 | 0.885916 | 0.885917 | 4 |
| 0.001 |  | 1.67e-09 | 2.81e-09 | 0, 25 | 0, 25 | 0.883205 | 0.883205 | 2 |
| 0.003 |  | 4.80e-09 | 8.37e-09 | 0, 25 | 0, 25 | 0.876284 | 0.876284 | 2 |
| 0.01 | x | 1.75e-14 | 2.18e-14 | 0, 25 | 0, 25 | 0.859264 | 0.859264 | 0 |
| 0.02 |  | 1.58e-14 | 1.37e-14 | 0, 25 | 0, 25 | 0.846417 | 0.846417 | 0 |
| 0.05 |  | 2.24e-14 | 1.81e-14 | 0, 25 | 0, 25 | 0.833977 | 0.833977 | 0 |
| 0.1 |  | 1.52e-14 | 2.41e-14 | 0, 25 | 0, 25 | 0.824431 | 0.824431 | 0 |
| 0.2 | x | 1.64e-14 | 1.60e-14 | 0, 25 | 0, 25 | 0.798855 | 0.798855 | 0 |

Gate summary: max abs diff 1.75e-14, max rel diff 2.18e-14; over all 10 fz: 6.30e-06 / 1.53e-05. Peak argmax positions match for every fz (asserted for the gates).

Interpretation of the difference pattern: wherever the last column is 0, both formulations lower to numerically identical Hamiltonian plans and the Rust solver takes bit-identical steps -- curves agree to ~1e-14 (fz=0 is exactly 0). The only visible differences (up to ~1e-5 relative, at the smallest nonzero fz) are NOT solver error: the reference build zeroes mixed-coupling-matrix elements below `relative_coupling=1e-3` of the mixed matrix's largest element, which at small fz removes sqrt(fz)-scaled Z couplings that the single-system build (which thresholds its X and Z fields independently) retains. The single-system curve is therefore the slightly MORE faithful one at small fz; the discrepancy vanishes once sqrt(fz) lifts those elements above the cutoff (fz >= 0.01 here).

## Timing

| approach | build | prepare | scan | total |
| --- | --- | --- | --- | --- |
| single system (1 build + 1 prepare + 1 parameter_scan of 360) | 6.94 s | 0.59 s | 67.01 s | 74.54 s |
| per-fz rebuild (10 builds + prepares + grid_scans of 36) | 81.37 s | 6.21 s | 77.68 s | 165.26 s |

**Total speedup: 2.22x** (165.3 s -> 74.5 s). Build+prepare overhead alone drops 11.6x (87.6 s -> 7.5 s); solve time is unchanged by construction (identical Hamiltonians, same trajectory count), so the scan-phase difference reflects Rust batch scheduling of one large batch vs 10 smaller ones.

## Framework notes / gaps

- No library changes were needed. The existing machinery -- multiple polarization components per `TransitionSelector`, per-component amplitude symbols in `generate_symbolic_hamiltonian`, `LindbladParameters.bind` of a polarization symbol to a base `Parameter`, and `parameter_scan` over base-parameter slots -- composes as designed.
- Subtlety worth documenting: `main_coupling` (and hence the notebook's rabi) depends on the polarization mix, but the ratio `Omega/main_coupling` entering H does not. Anyone porting a per-mix scan to symbol-bound amplitudes must bind rabi computed from the SHARED system's main coupling and put the mix entirely into the amplitude symbols; binding per-fz rabis AND amplitudes would double-count sqrt(1-fz).
- Thresholding caveat (measured, see the validation table's last column): per-field coupling matrices are thresholded independently (relative_coupling=1e-3 of each field's own max), while the reference thresholds the mixed matrix, dropping sqrt(fz)-scaled Z elements at small fz. Curve-level impact stayed below ~2e-5 relative and the single-system result is the more complete one.
- The detuning grid here is -5..30 MHz in 1 MHz steps (36 points, 360 trajectories total); per-trajectory solve cost is flat (~1.1 s, see `step_size_diagnostics_report.md`), so timings scale linearly to the notebook's denser grids.
