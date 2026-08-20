# AGENTS.md — working in `centrex-tlf`

Orientation for coding agents. Read this before exploring the package; it should be enough to set
up most simulations without reading the source. For depth see `README.md` (states, Hamiltonians,
couplings) and `README_OBE_SOLVER.md` (solver selection, output modes, runtime parameters).

## What this repo is

`centrex-tlf` generates the thallium fluoride (TlF) states, Hamiltonians, transitions, couplings
and Lindblad / optical-Bloch equations used by **CeNTREX** — a cryogenic molecular-beam experiment
searching for hadronic CP violation via the Schiff moment of the <sup>205</sup>Tl nucleus.

The physical model is a two-electronic-state molecule: the ground **X** <sup>1</sup>Σ<sup>+</sup>
state (Ω=0) and the electronically excited **B** <sup>3</sup>Π<sub>1</sub> state (Ω=±1, with
Λ-doubled parity pairs). Both carry hyperfine structure from two spin-1/2 nuclei
(<sup>205</sup>Tl and <sup>19</sup>F), coupled as `F1 = J + I_F` and `F = F1 + I_Tl`.

Published to PyPI as `centrex-tlf`. Heavy numerics run through a compiled Rust extension
(`centrex_tlf_rust`, sources in `rust/`). An optional Julia backend lives in a separate package,
`centrex-tlf-julia-extension`.

## Environment

Windows. Use the project virtualenv interpreter directly — the package is installed editable:

```
.venv\Scripts\python.exe          # PowerShell
./.venv/Scripts/python.exe        # Bash tool
```

Set `PYTHONIOENCODING=utf-8` when printing state labels; they contain `Ω`, `σ`, `₁` and the
default Windows console codec (cp1252) raises `UnicodeEncodeError`.

## Module map

| Module | What it is for |
|---|---|
| `states` | `CoupledBasisState`, `UncoupledBasisState`, `State`/`CoupledState` superpositions, `QuantumSelector` for picking subsets, `generate_coupled_states_X/B`, `find_exact_states`, state compaction |
| `hamiltonian` | X and B Hamiltonians (rotational, hyperfine, nuclear spin-rotation, Λ-doubling, Stark, Zeeman), basis transformations, dipole matrix elements, and the reduced/diagonalized builders that most code actually calls |
| `transitions` | `OpticalTransition` (types `O,P,Q,R,S` = ΔJ of −2…+2), `MicrowaveTransition`, and ~48 predefined constants such as `P2_F1_3o2_F1`, `R2_F1_7o2_F3`, `Q1_F1_3o2_F2` |
| `couplings` | `Polarization`, coupling fields, branching ratios, collapse (decay) operators, `TransitionSelector`, `generate_transition_selectors` |
| `lindblad` | OBE assembly (`generate_OBE_system_transitions`), runtime parameters (`LindbladParameters`), prepared problems, solvers, batch/grid scans, terminal events |
| `effective_hamiltonian` | Lower-dimensional effective models — a **separate** path with its own API, not a drop-in replacement for the full OBE |
| `utils` | Rabi/intensity/power conversions, Doppler helpers, multipass beam profiles, phase modulation, initial populations, level-diagram plotting |
| `constants` | `XConstants`, `BConstants`, `TlFNuclearSpins`, `ED_XtB`, and `Γ` (the B-state decay rate, `1/B_LIFETIME` ≈ 1.0101e7 s⁻¹, from the measured τ = 99(9) ns lifetime; Γ/(2π) ≈ 1.608 MHz) |

## Conventions

- **Electric field `E`: V/cm.** **Magnetic field `B`: Gauss.** Both are 3-vectors, `[Ex, Ey, Ez]`.
- **All energies, detunings and Rabi rates are angular frequencies in rad/s.** To scan a detuning
  in MHz: `2 * np.pi * 1e6 * detuning_MHz`. To read one out: `value / (2 * np.pi * 1e6)`.
- Use `B=[0, 0, 1e-5]` rather than exactly zero — X states become degenerate below roughly that,
  and degenerate eigenvectors break state identification.
- Primed quantum numbers are the **excited** state. `P(2) F1'=3/2 F'=1` names the *ground* J in
  parentheses: J=2 → J′=1 (P branch is ΔJ=−1).
- X-state parity is `(-1)**J`; the optical `P_excited = -P_ground`.
- Quantum numbers are floats where half-integral (`F1=3/2` is `1.5`).
- `couplings.collapse_matrices` takes `decay_rate=` (the population decay rate Γ = 1/τ, s⁻¹).
  The old `gamma=` keyword still works but emits a `DeprecationWarning`; passing both raises.

## The canonical OBE recipe

This snippet is validated against the current API. It builds a single optical transition with the
excited-state Λ-doublet partner retained, in a DC field, and scans laser detuning.

```python
import numpy as np
from centrex_tlf import couplings, hamiltonian, lindblad, states, transitions
from centrex_tlf.couplings.polarization import Polarization
from centrex_tlf.utils import population
from centrex_tlf.utils.rabi import power_to_rabi_rectangular_beam

GAMMA = getattr(hamiltonian, "Γ")
transition = transitions.P2_F1_3o2_F1
E_FIELD, B_FIELD = np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 1e-5])
T_END = 0.02 / 184.0                       # 2 cm interaction length at 184 m/s

# 1. Reference ("main") states: fix the detuning origin and the power->Rabi normalization.
ground_main = 1 * next(iter(states.generate_coupled_states_X(states.QuantumSelector(
    electronic=states.ElectronicState.X, J=2, F1=5 / 2, F=2, mF=0, P=transition.P_ground))))
excited_main = 1 * next(iter(states.generate_coupled_states_B(states.QuantumSelector(
    electronic=states.ElectronicState.B, J=transition.J_excited, F1=transition.F1_excited,
    F=transition.F_excited, mF=1, P=transition.P_excited))))

# 2. Polarization -> transition selector.
pol = Polarization(np.array([1.0, 0.0, 0.0], dtype=np.complex128), name="X")
selectors = couplings.generate_transition_selectors(
    transitions=[transition], polarizations=[[pol]],
    ground_mains=[ground_main], excited_mains=[excited_main])

# 3. Build the OBE system.
system = lindblad.generate_OBE_system_transitions(
    [transition], selectors, qn_compact=True, E=E_FIELD, B=B_FIELD,
    retain_opposite_parity_levels=True, normalize_pol=True, Jmax_X=4, Jmax_B=4)

# 4. Bind runtime parameters. This loop is the least obvious part: every free symbol in
#    H_symbolic must be bound or registered, or preparation fails.
params = lindblad.LindbladParameters()
rabi = params.real("rabi", 0.0)
detuning = params.real("detuning", 0.0)
for symbol in system.H_symbolic.free_symbols:
    if symbol in system.coupling_symbols:
        params.bind(symbol, rabi, finalize=False)          # Rabi rate of the main coupling
    elif str(symbol) == str(selectors[0].δ):
        params.bind(symbol, detuning, finalize=False)      # laser detuning
    else:
        params.real(str(symbol), 0.0)
for group in system.polarization_symbols:                  # amplitude of each field component
    for symbol in (group if isinstance(group, (list, tuple)) else [group]):
        params.bind(symbol, 1.0, finalize=False)
params._finalize()

prepared = lindblad.prepare_lindblad_problem(
    system, params, backend="rust", hamiltonian_representation="decomposed")

# 5. Initial density matrix, by quantum-number label.
selectors_rho0 = [states.QuantumSelector(J=2, F1=5 / 2, F=2, mF=0),
                  states.QuantumSelector(J=2, F1=5 / 2, F=3, mF=0)]
groups = [np.asarray(s.get_indices(system.QN), dtype=int).ravel() for s in selectors_rho0]
weights = np.concatenate([np.full(g.size, w / g.size) for g, w in zip(groups, [2 / 3, 1 / 3])])
rho0 = population.generate_uniform_population_state_indices(
    np.concatenate(groups), len(system.QN), weights=weights)

# 6. Scan. Power -> Rabi uses this build's own main_coupling.
rabi_value = power_to_rabi_rectangular_beam(
    60e-3, abs(system.couplings[0].main_coupling), 0.02, 0.02)
excited = [i for i, s in enumerate(system.QN)
           if s.largest.electronic_state == states.ElectronicState.B]

result = lindblad.grid_scan(
    prepared, rho0, (0.0, T_END),
    scan={"detuning": 2 * np.pi * 1e6 * np.arange(-30.0, 160.0, 0.5),
          "rabi": np.array([rabi_value])},
    solver="dopri5", execution_mode="expanded_sparse",
    output="photon_integral", integral_weights=[(i, float(GAMMA)) for i in excited],
    output_when="saveat", saveat=np.linspace(0.0, T_END, 801),
    dt=2e-9, reltol=1e-7, abstol=1e-9, parallel=True)
photons = result.values.reshape(-1, 801)[:, -1].real   # integrated photons per molecule
```

Everything used above is re-exported from `centrex_tlf.lindblad` directly; the deeper import paths
(`lindblad.plan_static`, `lindblad.parameters`) that older notebooks use still work but are not
needed.

## Gotchas

Each of these produces plausible-looking but wrong output rather than an error.

- **The X Hamiltonian changed; cached or pickled X Hamiltonians are stale.** X now carries a
  quartic centrifugal-distortion term `-D_rot·[J(J+1)]²` (`D0_X = -Y02_X` ≈ 5.84 kHz), and
  `B_rot` moved by ~24 kHz to the Dunham-derived `B0_X = Y01 + Y11/2 + Y21/4` ≈ 6.667355 GHz.
  Anything rebuilt from a `.pkl` or an `@lru_cache` populated before this change will disagree
  with a fresh build by tens of kHz in J=2 — enough to move a line position but not enough to
  look broken. Regenerate cached Hamiltonians rather than reusing them. Both constants stay
  derived from the Dunham coefficients `Y01_X`/`Y11_X`/`Y21_X`/`Y02_X`; do not hard-code them
  separately, and note `rust/src/constants.rs` mirrors the same derivation.
- **Read level positions from `H_symbolic`, not `H_int`.** The rotating-frame line positions are
  `complex(sympy.N(system.H_symbolic[i, i].subs({selector.δ: 0}))).real / (2*np.pi*1e6)`. The
  `H_int` B block sits in a different energy origin and gives numbers off by orders of magnitude.
- **Registered parameters default to 0.0.** Calling `solve_lindblad` without setting `rabi`
  silently integrates a completely undriven trajectory and returns clean, believable output. Set
  drive values explicitly through the scan or via parameter overrides.
- **Select states by `QuantumSelector` label, never by energy ordering.** Level orderings cross as
  the field changes (in X J=2 the F=3, mF=±1 pair drops below F=2, mF=0 between 200 and
  250 V/cm), so `min(candidates, key=lambda i: H_int[i, i])` silently picks a different state at
  different fields.
- **State labels stop being pure long before state *identification* breaks.** Stark mixing
  drives the overlap of every X state with its bare label below 0.5 above roughly
  400 V/cm (the four |mF|=1 levels in J=2 mix among themselves), and every B state below
  0.5 by 100 V/cm when both Λ-doublet partners are retained — yet the assignment
  `find_exact_states` returns is still exact in both cases, verified against adiabatic
  continuation from zero field. So `find_exact_states_indices` no longer warns on low
  purity by default (`overlap_threshold=None`, pass `0.5` to restore it). It warns instead
  on a small or negative **margin** between the best and second-best overlap
  (`margin_threshold=0.02`), the condition under which the label can actually land on the
  wrong eigenvector.
- **The margin warning is sufficient, not necessary.** It is calibrated to stay silent
  wherever single-shot matching is genuinely correct (X to 10 kV/cm, B to 200 V/cm), which
  forces the threshold low: at `0.12` an ordinary B analysis at 200 V/cm flags 52 of 192
  states with nothing wrong. The price is incomplete coverage — at 30 kV/cm in X it catches
  4 of 8 mis-assignments, at 1 kV/cm in B 10 of 44. Margins of mis-assigned states run as
  high as +0.11 (X, 50 kV/cm) and +0.22 (B, 500 V/cm), so no threshold separates right from
  wrong cleanly. **Silence is not a guarantee; use the field-based rule below.**
- **Above ~10 kV/cm in X, or ~500 V/cm in B with both parities retained, match states
  adiabatically — one-shot matching is wrong, not merely uncertain.** Converged X basis
  (J=0–6, 196 levels): one-shot disagrees with tracking on 8 states at 20 and 30 kV/cm and
  66 at 50 kV/cm. B (J=1–4, both parities, 192 levels): 12 states wrong at 500 V/cm, 44 at
  1 kV/cm. In every case one-shot scores the *higher* total overlap (X 30 kV/cm: 91.83 vs
  90.22 tracked), i.e. its objective prefers the physically wrong answer, so no
  bare-overlap diagnostic can rescue it. Step the field up from ~0 and match each set of
  eigenvectors to the previous set; the result is step-size independent (X: 50 vs 20 V/cm
  steps to 50 kV/cm; B: 1.0 vs 0.2 V/cm steps to 1 kV/cm).
  The OBE builders are exposed to this too. `generate_reduced_B_hamiltonian` diagonalizes
  the *full* manifold — `J = 1 … J′+2`, `P=[-1, 1]`, `Ω=1`, every F₁/F/mF (120 states for
  J′=1) — and matches against all of it; only `B_states_approx`, the handful of states the
  transition names, gets a margin evaluated. Whether the builder warns therefore depends on
  which manifold the transition targets: at 171.6 V/cm the J′=1 F₁=3/2 F=1 states sit at
  margin +0.125, but J′=2 F₁=3/2 F=2 sits at +0.049 and does warn at a 0.12 threshold.
  This is the concrete reason the default is 0.02 and not 0.12.
- **At tens of kV/cm the J truncation must be raised — J≤3 does not have the physics.**
  Going from `Jmax=3` to `Jmax=4` at 30 kV/cm shifts X state purities by 0.055 and
  assignment margins by 0.092, i.e. more than the whole ambiguity threshold; it also
  understates how badly one-shot matching breaks (24 wrong states at 50 kV/cm instead of
  66). Converged by `Jmax=5`–`6` (Δmargin 5→6 is 0.0005, 6→7 is 0.0001). At 1 and 5 kV/cm
  `Jmax=3` is already converged to 4 decimals, so this is specific to the tens-of-kV/cm
  regime.
- **`excited_main` must be reachable by the polarization in use.** With pure X̂ light and an mF=0
  ground state, only mF′=±1 is driven; choosing an mF′=0 `excited_main` makes `main_coupling` zero
  and the power→Rabi conversion divides by zero.
- **Power→Rabi is polarization-safe as written.** `power_to_rabi_rectangular_beam` returns
  `Ω = d_main(pol) · E_field`, and the coupling matrix is divided by `main_coupling`, so
  `Ω / main_coupling = E_field` is polarization-independent. Total optical power really is held
  fixed as polarization rotates — but recompute `main_coupling` per build; do not reuse it.
- **Selection rules are diagnostics, not gates.** Whether a transition is allowed is decided
  by the magnitude of the *mixed-state* dipole matrix element between the field-dressed main
  states, not by applying the E1 rules to their bare labels. In a field, mixing makes
  nominally forbidden pairs genuinely driveable — e.g. for `P2_F1_3o2_F1` at 200 V/cm the
  ΔF=2 pair X `J=2 F=3 mF=-1` → B `J=1 F1=3/2 F=1 mF=0` has `|ME| = 0.159`, and a
  parity-forbidden pair reaches 71% of the strongest coupling in the matrix. At zero field
  P, F and mF stay good quantum numbers, so such elements are *identically* zero and the
  numeric test reproduces the old rule-based behaviour exactly. `check_transitions_allowed`
  is no longer called during setup; the rules are consulted only to explain a vanishing
  matrix element. (Zero-field hyperfine mixing does exist — F₁ mixes at the ~0.8% level —
  but F₁ is not one of the quantum numbers the rules test.)
- **A weak `main_coupling` now warns.** `main_coupling` divides the whole coupling matrix in
  `generate_hamiltonian.py`, so it is the Rabi normalization reference. Naming a
  mixing-only pair as the main pair silently inflates every Rabi rate; a warning fires below
  `weak_main_fraction` (default 1e-2) of the strongest element. Tune or silence it with the
  `weak_main_fraction` argument to `generate_coupling_field`.
- **`B=[0,0,1e-5]` orders the ±mF states but does not make their eigenvectors
  well-determined.** At that field the Zeeman splitting of an mF=±1 pair is ~6e-8 MHz
  (0.37 rad/s) while `||H||₂` is ~5e11 rad/s, a relative gap of 7e-13 — only ~3000× machine
  epsilon. Eigen*values* are still exact to full precision, but the eigen*vector* error
  scales as `eps·||H||/gap`, so the mixing angle within a ±mF pair is uncertain at the
  1e-4…1e-2 level and changes with the BLAS build (kernel selection, blocking, threading —
  not just the numpy version). Measured: the same case gave a 0.54% different eigenvector on
  scipy-openblas 0.3.29 vs 0.3.34, shifting a field-mixed matrix element by 0.46%.
  This matters whenever a polarization addresses both ±mF (X̂ or Ŷ light couples an mF=0
  ground state to both mF′=±1), since the contamination then adds coherently. Strongly mixed
  couplings are the most exposed; ordinary allowed couplings sit on much larger gaps and are
  ~100× less sensitive. **For quantitative work involving ±mF pairs, use a B field that
  actually lifts the degeneracy (≥1e-3 G, ideally the real experimental field) rather than
  the 1e-5 placeholder.**
- **`method=` on the OBE builders is deprecated and does nothing.** Several committed notebooks
  still pass `method="matrix"`. Drop it in new code.
- **`generate_uniform_population_state_indices` used to be defined twice** in
  `centrex_tlf/utils/population.py` — the second definition shadowed the first. Fixed in
  0.2.5; the surviving one takes `weights=` and handles NumPy arrays. Committed notebooks
  still carry defensive `inspect.signature(...)` guards around it, which are now dead
  weight and can be dropped when you next touch them.
- **`QuantumSelector.Ω` is ignored by `get_indices`.** It only affects state *generation*.
- **`check_B_basis` raises if you ask for multiple `P` and multiple `Ω` at once.** Pick a basis:
  parity basis is `P=[-1, 1], Ω=1`; omega basis is `P=None, Ω=[-1, 1]`. Convert with
  `transform_to_omega_basis()` / `transform_to_parity_basis()`.
- **`gaussian_convolve_uniform_grid` is notebook-local**, copy-pasted across six notebooks in
  `examples/lindblad/`. It is not in the package. `utils.detuning` does provide `doppler_shift`
  and `velocity_to_detuning`.

### Default J truncation

`generate_reduced_hamiltonian_transitions` (used by `generate_OBE_system_transitions`) diagonalizes
X over `J = 0 … J′+2` and B over `J = 1 … J′+2`, then keeps only the states with dipole coupling
above `minimum_coupling=1e-3`. Measured convergence for an optical transition to B J′=1 at
250 V/cm: raising `Jmax_X` from 3 to 7 shifts X J=2 energies by **0.03 kHz** and leaves
`main_coupling` identical to six digits; raising `Jmax_B` from 3 to 6 leaves the B line offsets
identical to four decimals. The defaults are converged at few-hundred V/cm fields — do not spend
time re-deriving this, but do re-check if you move to kV/cm.

Convergence of the *eigenvectors* (state purity and assignment margin) is the stricter test, since
identification depends on them rather than on energies. For **B** the `J′+2` default is enough well
past experimental fields: for a J′=2 target, going from `Jmax = J′+2` to `J′+5` changes purity and
margin by 0.0000 at 171.6, 200, 500, 1000 and 5000 V/cm — only the `J′+1` basis is visibly off
(0.008 in margin at 5 kV/cm). For **X** the default holds to a few kV/cm but not beyond: at
30 kV/cm `Jmax=3` misses purity by 0.055 and margin by 0.092, and J=5–6 is needed (see the
identification gotchas above).

## Beyond single-laser OBE

Capabilities that already exist — check here before building something new.

- **Field curves.** Stark and Zeeman energies vs field: `generate_uncoupled_hamiltonian_X_function`
  / `generate_coupled_hamiltonian_B_function` give `H_func(E, B)`; track state identity across
  fields with `reorder_evecs` or `find_exact_states`. Examples in `examples/hamiltonian/`.
- **Branching ratios and level diagrams.** `couplings.calculate_br`,
  `couplings.generate_br_dataframe`, `utils.plotting.plot_level_diagram`.
- **Field-dressed X→B diagram for one optical transition.**
  `utils.plotting.plot_transition_level_diagram(transitions.P2_F1_3o2_F1, E=170)`, or with
  explicit quantum numbers (`J_ground=2, branch="P", F1_excited=1.5, F_excited=1`). Every
  level is drawn as a bar segmented by its zero-field parent character — hyperfine `(F1, F)`
  parents in X, the two Λ-doublet parity parents in B — so Stark mixing is visible directly.
  `E` (V/cm) and `B` (Gauss) are both along z, so mF stays good and the calculation runs per
  mF block; that is also why it needs no `B=[0,0,1e-5]` placeholder. Levels are matched to
  parents by adiabatic tracking from zero field, so labels stay right above the fields where
  one-shot matching breaks; the ramp is sized by `max_tracking_step_V_cm` (default 2.0) and
  `max_tracking_step_G` (default 1.0), whichever demands more steps.
  `utils.plotting.calculate_transition_level_structure` returns the
  same numbers without plotting, and every plotted number is on the returned
  `TransitionLevelStructure`. Cross-checked against an independent hand-rolled calculation in
  `tests/utils/test_level_diagram.py`.
- **Multi-transition systems.** Lasers and microwaves together — rotational cooling drives
  `P(2)` plus `MicrowaveTransition(1, 2)` and `(2, 3)` in one OBE system.
- **Polarization switching.** Pass two polarizations for one transition
  (`polarizations=[[pol_a, pol_b]]`); `generate_lindblad_parameters` wires them as
  `PA = square_wave(t, ...)`, `PB = 1 - PA`. More than two raises.
- **Time-dependent drives.** `RuntimeExpression` helpers: `gaussian`, `gaussian_1d/2d`,
  `gaussian_beam_rabi`, `multipass_2d_intensity`, `multipass_2d_rabi`, `rabi_from_intensity`,
  `square_wave`, `sawtooth_wave`, `sine`, `phase_modulation`,
  `resonant_polarization_modulation`, `linear_interp`, `pchip_interp`, `tabulated`.
- **Extra decay paths.** `lindblad.DecayChannel` adds loss channels outside the modelled manifold.
- **Initial populations.** `generate_thermal_population_states` (Boltzmann over J),
  `generate_uniform_population_states`, `generate_uniform_population_state_indices`.
- **Terminal events.** `lindblad.PopulationEvent` for time-to-threshold measurements.
- **Scans.** `solve_lindblad_batch`, `parameter_scan`, `initial_condition_scan`, `grid_scan` —
  all with `parallel=True` and Rust-side threading.
- **State compaction.** `qn_compact=True` collapses spectator manifolds into single levels, a
  large speedup when you only care about total population in them.

## Performance

- Defaults that are almost always right: `solver="dopri5"`,
  `execution_mode="expanded_sparse"`, `hamiltonian_representation="decomposed"`,
  `backend="rust"`, `parallel=True`. `tsit5` is a reasonable alternative.
- SciPy `scipy_bdf` / `scipy_radau` exist only as stiff fallbacks and are much slower for scans.
  `python_rk45` is for correctness checks.
- Prepare once, scan many times — symbolic lowering happens in `prepare_lindblad_problem`.
- Cache expensive rebuilds with `@lru_cache` keyed on **integer grid indices**, not floats.
- Cost anchor (29-level system, 109 µs trajectory, 8-core laptop): ~1 s per OBE build,
  ~130 ms per trajectory, so ~50 s for a 380-point detuning spectrum with `parallel=True`.
- Anything that changes the coupling matrix (polarization, E field) needs a rebuild. Detuning,
  Rabi rate and other runtime parameters do not — put them in the same `grid_scan`.
- On-disk caching convention (from `examples/ramsey_rf/benchmarks/_cache`): the filename encodes
  the physics and solver settings plus a `_v1` version bump, and the `.npz` carries a scalar
  `cache_version` plus every parameter needed to invalidate it, along with timings.

## Sanity checks worth running

- `np.trace(rho)` stays 1 to ~1e-9 across the trajectory. If it drifts, decay is leaving the
  modelled manifold.
- The number of levels and the X/B split match expectations.
- Line positions computed from `H_symbolic` match an independent
  `generate_reduced_B_hamiltonian` calculation at the same field.

## Where to find a template

| I want to… | Start from |
|---|---|
| Retain excited opposite-parity (Λ-doublet) levels | `examples/lindblad/r2_opposite_parity_retention.ipynb` |
| …with a weighted multi-level initial population | `examples/lindblad/r2_opposite_parity_retention_initial_pop.ipynb` |
| Scan X/Z polarization admixture | `examples/lindblad/r2_peak_ratio_vs_z_polarization.ipynb` |
| Scan E field *and* polarization, mF′-resolved, with npz caching | `examples/lindblad/p2_f1_3o2_f1_mf_scan.ipynb` |
| Scan optical power | `examples/lindblad/r2_peak_ratio_vs_power.ipynb` |
| Fit simulations to digitized experimental data | `examples/lindblad/r2_nltl_scan_fit_differential_evolution_cached_field.ipynb` |
| Run a 2-D Rust grid scan | `examples/lindblad/r0_f2_batch_grid_scan.ipynb` |
| Rotational cooling (laser + microwaves) | `examples/lindblad/rotational_cooling.ipynb` |
| Multi-line spectrum over a wide window, several branches at once | `examples/lindblad/r2_zero_field_s0_overlap_scan.ipynb` |
| Polarization switching | `examples/lindblad/polarization_switching.ipynb` |
| Terminal events / time-to-threshold | `examples/lindblad/rotational_cooling_terminal_event_scans.ipynb` |
| Effective-Hamiltonian models | `examples/lindblad/q1_effective_fixed_basis_vs_static_regular_rust.ipynb` |
| Stark curves vs field | `examples/hamiltonian/plot_StarkShift_TlF_J2.py`, `plot_StarkShift_B.py` |
| Detailed field-dependent level analysis | `examples/hamiltonian/j2_mf0_crossing_analysis.ipynb` |
| A field-dressed X→B diagram for one transition | `utils.plotting.plot_transition_level_diagram` |
| Branching ratios | `examples/couplings/branching_ratios.ipynb` |
| State preparation (SPA) | `examples/spa paper/state_prep_python_example.ipynb` |

Note that `examples/ramsey_rf/` contains only cached artifacts — its driver scripts are not in the
working tree.
