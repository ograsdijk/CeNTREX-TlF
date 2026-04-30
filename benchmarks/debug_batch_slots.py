import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import sympy as smp
from centrex_tlf import transitions, couplings
from centrex_tlf.effective_hamiltonian import prepare_lindblad_safe_compact_interpolated_model
from centrex_tlf.effective_hamiltonian.rust_plan import prepare_effective_lindblad_rust_plan
from centrex_tlf.lindblad.parameters import (
    LindbladParameters, Parameter, Time, linear, gaussian, pchip_tabulated,
)
from centrex_tlf.lindblad.ir import lower_parameter_graph

t = transitions.Q1_F1_1o2_F0
model = prepare_lindblad_safe_compact_interpolated_model(
    field_points=[0, 5, 10, 20, 30, 40, 50],
    transition=t, optical_polarization=couplings.polarization_Z,
)
z_grid = np.linspace(-0.01, 0.01, 50)
ez = 10.0 + 15.0 * np.exp(-z_grid**2 / (2 * 0.003**2))
te = Time()
v = Parameter("v", 200.0)
z0 = Parameter("z0", -0.005)
omega0 = Parameter("omega0", 2 * np.pi * 1e6)
z_laser = Parameter("z_laser", 0.002)
w0 = Parameter("w0", 200e-6)
z_expr = linear(te, offset=z0, slope=v)
gp = Parameter("z_grid", tuple(z_grid.tolist()))
ep = Parameter("ez", tuple(ez.tolist()))
Ez = pchip_tabulated(z_expr, gp, ep)
Omega = gaussian(z_expr, center=z_laser, sigma=w0, amplitude=omega0)
params = LindbladParameters({
    smp.Symbol("Ez"): Ez,
    smp.Symbol("\u03a90"): Omega,
    smp.Symbol("\u03b40"): 0.0,
})
pg = lower_parameter_graph(params)
print("slot_names:", pg["slot_names"])
print("n_base:", pg["n_base"])
print("n_compounds:", len(pg["compounds"]))
for i, name in enumerate(pg["slot_names"]):
    is_base = i < pg["n_base"]
    kind = "base" if is_base else "compound"
    print(f"  {i}: {name} ({kind})")
