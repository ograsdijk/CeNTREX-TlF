using LinearAlgebra
using Plots
using ProgressMeter

include("electric_field.jl")
# include("J01_J12_spa_backgroundfunctions.jl")
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
    Ω0, Ω1, ω0, ω1, δ0, δ1, vz, z0 = p
    nt     = length(t_array)
    ntrack = length(states_track)

    # Helpers for allocations
    similar_c() = zeros(ComplexF64, N, N)
    similar_r() = zeros(Float64,   N, N)

    # Matrices
    H            = similar_c()      # Stark (field only)
    C            = similar_c()      # Couplings (bare basis)
    H_total      = similar_c()      # Full bare Hamiltonian
    H_dressed_S  = similar_c()      # Full Hamiltonian in Stark basis
    buffer_c     = similar_c()
    buffer_r     = similar_r()
    Vsort        = similar_c()

    # Bases
    V_S_prev = similar_c()
    V_S      = similar_c()
    V_D_prev = similar_c()
    V_D      = similar_c()
    W_dressed = similar_c()         # dressed eigenvectors (in Stark basis)

    # Overlap and phases
    O_overlap  = similar_c()
    phases_i   = zeros(ComplexF64, N)
    phases_ip1 = zeros(ComplexF64, N)

    # Eigenvalues
    D_S_prev = zeros(Float64, N)
    D_S      = zeros(Float64, N)
    D_D_prev = zeros(Float64, N)
    D_D      = zeros(Float64, N)

    # Coefficients and states
    c      = zeros(ComplexF64, N, ntrack)   # dressed basis coefficients
    ψ      = zeros(ComplexF64, N, ntrack)
    ψ_tmp  = zeros(ComplexF64, N, ntrack)
    tmp_Nk = zeros(ComplexF64, ntrack, N)   # for probability helper

    # Outputs
    ψt = Array{ComplexF64,3}(undef, N, ntrack, nt)
    Et = Array{Float64,2}(undef, N, nt)
    Pt = Array{Float64,3}(undef, ntrack, N, nt)

    # ---------- t0 ----------
    t0 = t_array[1]
    z  = z0 + vz * t0
    Ez = Ez_interp(z)

    # Stark Hamiltonian
    hamiltonian_full_nocoupling!(H, Ez)
    copyto!(buffer_c, H)
    D_S_prev, _ = LAPACK.syev!('V','U', buffer_c)
    copyto!(V_S_prev, buffer_c)

    # Couplings
    Ωval0 = Ω0/2 * gaussian_peak(z, zμ0, σμ)
    Ωval1 = Ω1/2 * gaussian_peak(z, zμ1, σμ)
    coupling_full!(C, Ωval0, Ωval1)

    # Total bare Hamiltonian
    H_total .= H .+ C

    # Transform to Stark basis
    apply_transform!(H_dressed_S, H_total, V_S_prev, buffer_c)

    # Detunings (acting in Stark basis)
    detuning!(H_dressed_S, δ0, δ1, ω0, ω1)

    # Dressed diagonalization (in Stark basis)
    copyto!(buffer_c, H_dressed_S)
    D_D_prev, _ = LAPACK.syev!('V','U', buffer_c)
    copyto!(W_dressed, buffer_c)

    # Bare dressed eigenvectors
    mul!(V_D_prev, V_S_prev, W_dressed)

    # Initial coefficients: start in chosen dressed eigenvectors
    ψ .= @view V_D_prev[:, states_track]
    c .= 0
    @inbounds for (j, st) in enumerate(states_track)
        c[st, j] = 1
    end

    ψt[:,:,1] .= ψ
    Et[:,1]    .= D_D_prev
    calculate_probabilities!(@view(Pt[:,:,1]), ψ, V_D_prev, tmp_Nk)

    # ---------- Time loop ----------
    @inbounds for step in 2:nt
        t_prev = t_array[step-1]
        t_cur  = t_array[step]
        dt     = t_cur - t_prev
        z      = z0 + vz * t_cur
        Ez     = Ez_interp(z)

        # Stark part
        hamiltonian_full_nocoupling!(H, Ez)
        copyto!(buffer_c, H)
        D_S_raw, _ = LAPACK.syev!('V','U', buffer_c)
        copyto!(V_S, buffer_c)

        # Single reorder (Stark) relative to previous Stark basis
        reorder_evecs!(Vsort, D_S, V_S, D_S_raw, V_S_prev, buffer_r, buffer_c)
        V_S .= Vsort
        D_S .= D_S

        # Couplings
        Ωval0 = Ω0/2 * gaussian_peak(z, zμ0, σμ)
        Ωval1 = Ω1/2 * gaussian_peak(z, zμ1, σμ)
        coupling_full!(C, Ωval0, Ωval1)

        H_total .= H .+ C

        # Transform full Hamiltonian into the (reordered) Stark basis
        apply_transform!(H_dressed_S, H_total, V_S, buffer_c)
        detuning!(H_dressed_S, δ0, δ1, ω0, ω1)

        # Dressed diagonalization in Stark basis
        copyto!(buffer_c, H_dressed_S)
        D_D_raw, _ = LAPACK.syev!('V','U', buffer_c)
        copyto!(W_dressed, buffer_c)

        # Form dressed eigenvectors (bare basis)
        mul!(V_D, V_S, W_dressed)
        D_D .= D_D_raw   # adopt ordering as produced (no reorder on dressed)

        # Overlap dressed bases and phase gauge
        mul!(O_overlap, V_D_prev', V_D)   # O = V_D_prev† V_D
        @inbounds for k in 1:N
            zkk = O_overlap[k,k]
            if zkk != 0
                ph = zkk / abs(zkk)
                V_D[:,k] .*= conj(ph)
                # adjust row k of O to reflect phase change:
                @inbounds for j in 1:N
                    O_overlap[k,j] *= ph
                end
                O_overlap[k,k] = abs(zkk)
            end
        end

        # Half step phases
        @inbounds for k in 1:N
            phases_i[k]   = ComplexF64(cos(-D_D_prev[k]*dt/2), sin(-D_D_prev[k]*dt/2))
            phases_ip1[k] = ComplexF64(cos(-D_D[k]     *dt/2), sin(-D_D[k]     *dt/2))
        end

        # Strang propagation
        # First half
        for j in 1:ntrack
            @inbounds for k in 1:N
                c[k,j] *= phases_i[k]
            end
        end
        # Rotate coefficients to new dressed basis: c' = O† c
        mul!(ψ_tmp, O_overlap', c)
        c .= ψ_tmp
        # Second half
        for j in 1:ntrack
            @inbounds for k in 1:N
                c[k,j] *= phases_ip1[k]
            end
        end

        # Reconstruct ψ = V_D * c
        mul!(ψ, V_D, c)

        ψt[:,:,step] .= ψ
        Et[:,step]   .= D_D
        calculate_probabilities!(@view(Pt[:,:,step]), ψ, V_D, tmp_Nk)

        # Advance previous bases
        V_S_prev .= V_S
        D_S_prev .= D_S
        V_D_prev .= V_D
        D_D_prev .= D_D
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