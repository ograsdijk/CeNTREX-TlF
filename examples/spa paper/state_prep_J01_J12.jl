using LinearAlgebra
using MKL
using Plots
using ProgressMeter

include("electric_field.jl")
include("J01_J12_spa_functions.jl")
include("julia_functions.jl")

field_path = "c:/Users/Olivier/Anaconda3/envs/centrex-eql-testing/Lib/site-packages/state_prep/electric_fields/Electric field components vs z-position_SPA_ExpeVer.csv"
# field_path = "c:/Users/ogras/anaconda3/envs/centrex-eql/Lib/site-packages/state_prep/electric_fields/Electric field components vs z-position_SPA_ExpeVer.csv"

function reorder_evecs!(
    V_out::AbstractMatrix{<:ComplexF64},
    E_out::AbstractVector{<:Real},
    V_in ::AbstractMatrix{<:ComplexF64},
    E_in ::AbstractVector{<:Real},
    V_ref::AbstractMatrix{<:ComplexF64},
    overlap::AbstractMatrix{<:Real},
    workspace_complex::AbstractMatrix{<:ComplexF64},
)
    # 1. Absolute overlap between every eigenvector in V_in and every reference vector
    mul!(workspace_complex, V_in', V_ref)
    overlap .= abs.(workspace_complex)

    # 2. For each eigenvector (row), find the reference column with maximum overlap
    best_ref = map(argmax, eachrow(overlap))     # vector length n_in

    # 3. Sort eigenvectors by that reference-column index
    perm     = sortperm(best_ref)                # matches Python np.argsort(…)

    # if any(perm .!= eachindex(perm))
    #     println("Re-ordering eigenvectors with permutation: ", perm)
    # end

    # 4. Reorder into the provided output buffers (buffers must be distinct from inputs)
    V_out .= @view V_in[:, perm]
    E_out .= @view E_in[perm]
    return nothing
end

@inline function calculate_probabilities!(
    out::AbstractMatrix,            # ntrack × N
    ψ::AbstractMatrix,             # N × ntrack
    ϕ::AbstractMatrix,             # N × N
    buf::AbstractMatrix)           # ntrack × N
    # buf = ψ' * ϕ   →  (ntrack × N)
    mul!(buf, ψ', ϕ)

    # element-wise |·|², in place
    @. out = abs2(buf)
    return nothing
end


function evolve(p, t_array, states_track, Ez_interp, N)
    nt      = length(t_array)
    ntrack  = length(states_track)

    # ─────────────────────────────────  helpers ─────────────────────────────────
    similar_c() = zeros(ComplexF64, N, N)         # N×N Complex work matrix
    similar_r() = zeros(Float64,   N, N)          # N×N real work matrix

    # ────────────────────────────  work buffers  ───────────────────────────────
    Hinit   = similar_c()
    H       = similar_c()
    buffer_c  = similar_c()
    buffer_r = similar_r()         # real scratch for reorder_evecs!
    Hrot    = similar_c()
    C       = similar_c()
    V       = similar_c()
    Vsort   = similar_c()
    Vref    = similar_c()
    Vrefinit= similar_c()
    U       = similar_c()          # N×N time-evolution operator

    tmp_NN  = similar_c()          # general N×N scratch
    tmp_Nk  = zeros(ComplexF64, ntrack, N)   # ntrack×N scratch
    phase_vec = zeros(ComplexF64, N)
    Phase     = Diagonal(phase_vec)

    Esort   = zeros(Float64, N)    # eigenvalues after re-order

    ψ       = zeros(ComplexF64, N, ntrack)   # current kets
    ψ_tmp   = similar(ψ)                      # scratch for ψ update

    # ───────────────────────────  result containers  ───────────────────────────
    ψt  = Array{ComplexF64,3}(undef, N, ntrack, nt)
    Et  = Array{Float64,   2}(undef, N, nt)
    Pt  = Array{Float64,   3}(undef, ntrack, N, nt)

    # ───────────────────────────  initial step (t₀)  ───────────────────────────
    Ω0, Ω1, ω0, ω1, δ0, δ1, vz, z0 = p

    Ez   = Ez_interp(z0 + vz*t_array[1])
    hamiltonian_full_nocoupling!(Hinit, Ez)

    copyto!(buffer_c, Hinit)
    Eref, _ = LAPACK.syev!('V','U', buffer_c)     # eigen-pairs in   buffer / Eref
    copyto!(Vref, buffer_c)
    copyto!(Vrefinit, buffer_c)                   # keep a pristine copy if needed

    ψ .= @view Vref[:, states_track]

    ψt[:,:,1] .= ψ
    Et[:, 1]  .= Eref
    calculate_probabilities!(view(Pt,:,:,1), ψ, Vref, tmp_Nk)

    @inbounds for i in 2:length(t_array)
        dt = t_array[i] - t_array[i-1]
        t = t_array[i]

        z = z0 + vz * t
        Ez = Ez_interp(z)

        hamiltonian_full_nocoupling!(H, Ez)

        copyto!(V, H)
        D, _ = LinearAlgebra.LAPACK.syev!('V', 'U', V)
        # now V contains the eigenvectors

        reorder_evecs!(Vsort, Esort, V, D, Vref, buffer_r, buffer_c)

        Ωval0 = Ω0/2 * gaussian_peak(z, zμ0, σμ)
        Ωval1 = Ω1/2 * gaussian_peak(z, zμ1, σμ)
        coupling_full!(C, Ωval0, Ωval1)
        H .+= C
        apply_transform!(Hrot, H, V, buffer_c)
        detuning!(Hrot, δ0, δ1, ω0, ω1)

        Drot, _ = LinearAlgebra.LAPACK.syev!('V', 'U', Hrot)

        # Hrot now contains the eigenvectors

        # ── 1.  A ← Vview · Vrot  (unchanged) ──────────────────────────────
        mul!(U, V, Hrot)                     # U ≡ A   (N×N)

        # ── 2.  tmp_NN ← A · diag(phase)  (no copy of A) ───────────────────
        @inbounds @simd for k = 1:N              # build phase_vec in place
            θ = -Drot[k] * dt
            phase_vec[k] = ComplexF64(cos(θ), sin(θ))
        end
        mul!(tmp_NN, U, Phase)                   # tmp_NN = A_phase  (N×N)

        # ── 3.  buffer ← A_phase · A′  (identical to original) ─────────────
        mul!(buffer_c, tmp_NN, U')                # buffer = Udt

        # ── 4.  ψ ← Udt · ψ  (unchanged) ───────────────────────────────────
        mul!(ψ_tmp, buffer_c, ψ)
        ψ, ψ_tmp = ψ_tmp, ψ

        ψt[:, :, i] .= ψ
        Et[:, i] .= Esort
        calculate_probabilities!(@view(Pt[:, :, i]), ψ, Vsort, tmp_Nk)

        copyto!(Vref, Vsort)
    end
    return ψt, Et, Pt
end

const Ez_interp = make_Ez_interp(field_path)

const σμ = 1.078e-2
const zμ0 = 0.0
const zμ1 = 25.4e-3 * 1.125
const Γ = 2π * 1.56e6

Ω0 = 3e-1 * Γ
Ω1 = 2e-1 * Γ
ω0 = 83817989701.69382
ω1 = 167564827658.9896
δ0 = 0.0
δ1 = 0.0
vz = 184
z0 = -0.25
zstop = 0.2
tmax = (zstop - z0) / vz
N = 64
t_array = LinRange(0.0, tmax, 6001)
states_track = [1]

p = (Ω0, Ω1, ω0, ω1, δ0, δ1, vz, z0)

ψt, Et, Pt = evolve(p, t_array, states_track, Ez_interp, N);

@time ψt, Et, Pt = evolve(p, t_array, states_track, Ez_interp, N);

plot(t_array*vz .+ z0, Pt[1,1:36,:]')

argmax(Pt[1,:,1]), argmax(Pt[1,:,end])