import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
from centrex_tlf import transitions, couplings, lindblad
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.solve import solve_lindblad
from centrex_tlf.lindblad.parameters import LindbladParameters, adapt_lindblad_parameters
from collections import OrderedDict

P2 = transitions.OpticalTransition(transitions.OpticalTransitionType.P, 2, 3 / 2, 1)
J12 = transitions.MicrowaveTransition(1, 2)
J23 = transitions.MicrowaveTransition(2, 3)
ts = couplings.generate_transition_selectors(
    [P2, J12, J23],
    [
        [couplings.polarization_Z],
        [
            (couplings.polarization_X - couplings.polarization_Z).normalize(),
            couplings.polarization_Y,
        ],
        [
            (couplings.polarization_X + couplings.polarization_Z).normalize(),
            couplings.polarization_Y,
        ],
    ],
)
system = lindblad.generate_OBE_system_transitions([P2, J12, J23], ts, method="matrix")

omega_sw = 2 * np.pi * 500e3
rabi = 2 * np.pi * 1e6
phase = np.pi / 2

n = len(system.QN)
n_ground = len(system.ground)
rho0 = np.zeros((n, n), dtype=np.complex128)
for i in range(n_ground):
    rho0[i, i] = 1.0 / n_ground
t_span = (0.0, 50e-6)
saveat = np.linspace(t_span[0], t_span[1], 501)

def make_sin_params():
    params = lindblad.generate_lindblad_parameters(ts)
    params.base_parameters["\u03c91"] = omega_sw
    params.base_parameters["\u03c92"] = omega_sw
    params.base_parameters["\u03c61"] = 0.0
    params.base_parameters["\u03c62"] = phase
    params.base_parameters["\u03a9t0"] = rabi
    params.base_parameters["\u03a9t1"] = rabi
    params.base_parameters["\u03a9t2"] = rabi
    params.base_parameters["\u03b40"] = 0.0
    params.base_parameters["\u03b41"] = 0.0
    params.base_parameters["\u03b42"] = 0.0
    return params

def make_square_wave_params():
    base = OrderedDict()
    compound = OrderedDict()
    base["\u03a9t0"] = rabi
    base["\u03a9t1"] = rabi
    base["\u03a9t2"] = rabi
    base["\u03b40"] = 0.0
    base["\u03b41"] = 0.0
    base["\u03b42"] = 0.0
    base["PZ0"] = 1.0
    base["\u03c9sw"] = omega_sw
    base["\u03c6sw"] = phase
    compound["\u03a90"] = "\u03a9t0"
    compound["\u03a91"] = "\u03a9t1"
    compound["\u03a92"] = "\u03a9t2"
    compound["PA1"] = "square_wave(t, \u03c9sw, 0)"
    compound["PY1"] = "1 - PA1"
    compound["PA2"] = "square_wave(t, \u03c9sw, \u03c6sw)"
    compound["PY2"] = "1 - PA2"
    return LindbladParameters(base, compound)

for label, params_fn in [("sin > 0", make_sin_params), ("square_wave", make_square_wave_params)]:
    params = params_fn()
    prepared = prepare_lindblad_problem(system, params, backend="rust")
    times_list = []
    for run in range(3):
        t0 = time.perf_counter()
        result = solve_lindblad(
            prepared, rho0, t_span,
            solver="dopri5", execution_mode="structured_upper",
            saveat=saveat, dt=1e-10, reltol=1e-7, abstol=1e-9,
        )
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
    median = sorted(times_list)[1]
    print(f"{label:15s}: {median*1000:.1f} ms (trace={result.populations()[-1].sum():.6f})")
