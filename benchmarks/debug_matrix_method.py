import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from centrex_tlf import transitions, couplings, lindblad

trans = transitions.R0_F1_3o2_F2
transition_selectors = couplings.generate_transition_selectors(
    [trans], [[couplings.polarization_Z]]
)

obe_system = lindblad.setup_OBE_system_transitions(
    [trans], transition_selectors, verbose=False, qn_compact=True,
)

Gamma = 2 * np.pi * 1.56e6

from centrex_tlf_julia_extension import lindblad_julia
from centrex_tlf_julia_extension.lindblad_julia.utils_julia import jl

odepars = lindblad_julia.odeParameters(
    **{"\u03a90": Gamma, "\u03b40": 0.0, "PZ0": 1.0}
)

print("Setting up matrix method...")
obe_julia = lindblad_julia.setup_OBE_system_julia(
    obe_system, transition_selectors, odepars,
    n_procs=1, method="matrix", verbose=True,
)

# Print the generated code
print("\n=== Generated Hamiltonian Code ===")
print(obe_julia.code.hamiltonian[:500])
print("...")

print("\n=== Generated Support Code ===")
print(obe_julia.code.support[:500])
print("...")

print("\n=== Generated Lindblad Code ===")
print(obe_julia.code.lindblad)

n_jl = len(obe_system.QN)
n_excited_jl = len(obe_system.excited)
n_ground_jl = n_jl - n_excited_jl
print(f"\nJulia system: {n_jl} states ({n_ground_jl} ground + {n_excited_jl} excited)")
rho = np.zeros((n_jl, n_jl), dtype=np.complex128)
for i in range(n_ground_jl):
    rho[i, i] = 1.0 / n_ground_jl

print(f"\nodepars._method = {odepars._method}")
print(f"odepars._parameters = {odepars._parameters}")
print(f"odepars.p = {odepars.p}")

# Now manually set up in Julia step by step
jl.seval("using LinearAlgebra")

# Step 1: set rho
jl.rho_debug = rho
jl.seval("rho_debug = collect(rho_debug)")
jl.seval('println("rho type: ", typeof(rho_debug), " size: ", size(rho_debug))')
jl.seval('println("rho diag sum: ", sum(diag(rho_debug)))')

# Step 2: check what p is
jl.seval('println("\\np type: ", typeof(p))')
jl.seval('println("p fields: ", fieldnames(typeof(p)))')

# Step 3: test hamiltonian! alone
jl.seval("""
    H_debug = zeros(ComplexF64, size(rho_debug))
    p.hamiltonian!(H_debug, 0.0)
    println("\\nH_debug type: ", typeof(H_debug))
    println("H_debug norm: ", norm(H_debug))
    println("H_debug is Hermitian: ", isapprox(H_debug, H_debug'))
    println("H_debug diag[1:5]: ", diag(H_debug)[1:5])
""")

# Step 4: test her2k with these specific matrices
jl.seval("""
    C_debug = zeros(ComplexF64, size(rho_debug))
    
    println("\\nBefore her2k:")
    println("  H_debug norm: ", norm(H_debug))
    println("  rho_debug norm: ", norm(rho_debug))
    println("  C_debug norm: ", norm(C_debug))
    println("  H_debug stride: ", strides(H_debug))
    println("  rho_debug stride: ", strides(rho_debug))
    println("  C_debug stride: ", strides(C_debug))
    
    alpha = ComplexF64(0.0, 1.0)
    beta = 0.0
    
    BLAS.her2k!('U', 'N', alpha, rho_debug, H_debug, beta, C_debug)
    println("\\nAfter her2k (upper only):")
    println("  C_debug norm: ", norm(C_debug))
    println("  C_debug[1,1]: ", C_debug[1,1])
    println("  C_debug[1,2]: ", C_debug[1,2])
""")

# Step 5: test with liouvillian_commutator_her2k!
jl.seval("""
    C_debug2 = zeros(ComplexF64, size(rho_debug))
    liouvillian_commutator_her2k!(C_debug2, H_debug, rho_debug)
    println("\\nliouvillian_commutator_her2k! result:")
    println("  norm: ", norm(C_debug2))
    println("  C[1,1]: ", C_debug2[1,1])
""")

# Step 6: test with liouvillian_commutator! (mul! based)
jl.seval("""
    C_debug3 = zeros(ComplexF64, size(rho_debug))
    liouvillian_commutator!(C_debug3, H_debug, rho_debug)
    println("\\nliouvillian_commutator! (mul!) result:")
    println("  norm: ", norm(C_debug3))
    println("  C[1,1]: ", C_debug3[1,1])
""")

# Step 7: test full Lindblad_rhs!
jl.seval("""
    du_debug = similar(rho_debug)
    fill!(du_debug, zero(ComplexF64))
    Lindblad_rhs!(du_debug, rho_debug, p, 0.0)
    println("\\nFull Lindblad_rhs!:")
    println("  du norm: ", norm(du_debug))
    println("  du[1,1]: ", du_debug[1,1])
    println("  du diag sum: ", sum(diag(du_debug)))
    println("  any NaN: ", any(isnan, du_debug))
    println("  any Inf: ", any(isinf, du_debug))
""")

# Step 8: check what buffer0 looks like after the RHS call
jl.seval("""
    println("\\nAfter RHS, buffer0 (should be H):")
    println("  buffer0 norm: ", norm(p.buffer0))
    println("  buffer0 === H_debug: ", p.buffer0 === H_debug)
    println("  buffer0 type: ", typeof(p.buffer0))
""")

# Step 9: quick solve test
jl.seval("""
    prob_debug = ODEProblem(Lindblad_rhs!, rho_debug, (0.0, 1e-7), p)
    t_solve = @elapsed sol_debug = solve(prob_debug, Tsit5(); dt=1e-10, abstol=1e-9, reltol=1e-7, save_everystep=false, maxiters=10000, dense=false)
    println("\\nShort solve (0.1 μs):")
    println("  time: ", round(t_solve*1000, digits=1), " ms")
    println("  retcode: ", sol_debug.retcode)
    println("  trace at end: ", real(tr(sol_debug.u[end])))
""")
