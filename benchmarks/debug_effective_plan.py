import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import (
    prepare_lindblad_safe_compact_interpolated_model,
    default_effective_density_matrix,
)
from centrex_tlf.lindblad.parameters import LindbladParameters
from centrex_tlf.lindblad.ir import lower_parameter_graph

transition = transitions.OpticalTransition(
    transitions.OpticalTransitionType.R, 0, 1/2, 1
)
polarization = couplings.polarization_Z
Gamma = 2 * np.pi * 1.56e6

params = LindbladParameters({
    smp.Symbol("Ez"): 10.0,
    smp.Symbol("\u03a90"): Gamma,
    smp.Symbol("\u03b40"): 0.0,
})

pg = lower_parameter_graph(params)
print("slot_names:", pg["slot_names"])
print("n_base:", len(pg["base_values"]))
print("n_compounds:", len(pg["compounds"]))
for i, name in enumerate(pg["slot_names"]):
    print(f"  slot {i}: {name}")
