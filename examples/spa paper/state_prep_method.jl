using LinearAlgebra
using MKL
using Plots

include("electric_field.jl")
include("couplings.jl")
include("hamiltonian_nocoupling.jl")
include("julia_functions.jl")

# field_path = "c:/Users/Olivier/Anaconda3/envs/centrex-eql-testing/Lib/site-packages/state_prep/electric_fields/Electric field components vs z-position_SPA_ExpeVer.csv"
field_path = "c:/Users/ogras/anaconda3/envs/centrex-eql/Lib/site-packages/state_prep/electric_fields/Electric field components vs z-position_SPA_ExpeVer.csv"

function reorder_evecs!(
    V_out::AbstractMatrix{<:ComplexF64},
    E_out::AbstractVector{<:Real},
    V_in ::AbstractMatrix{<:ComplexF64},
    E_in ::AbstractVector{<:Real},
    V_ref::AbstractMatrix{<:ComplexF64},
)
    # 1. Absolute overlap between every eigenvector in V_in and every reference vector
    overlap  = abs.(V_in' * V_ref)               # size (n_in × n_ref)

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

@inline function calculate_probabilities!(out::AbstractMatrix,            # ntrack × N
    ψ::AbstractMatrix,             # N × ntrack
    ϕ::AbstractMatrix,             # N × N
    buf::AbstractMatrix)           # ntrack × N
    # buf = ψ' * ϕ   →  (ntrack × N)
    mul!(buf, ψ', ϕ)

    # element-wise |·|², in place
    @. out = abs2(buf)
    return nothing
end


function evolve(p, t_array, states_track, N)
    nt = length(t_array)
    ntrack = length(states_track)

    # initialize buffers and matrices
    Hinit = zeros(ComplexF64, N, N)
    H = similar(Hinit)
    buffer = similar(Hinit)
    Hrot = similar(Hinit)
    Vrot = similar(Hinit)
    C = similar(Hinit)
    Vrefinit = similar(Hinit)
    Vref = similar(Hinit)
    tmp_NN  = similar(H)                       # N×N scratch
    tmp_Nk  = zeros(ComplexF64, ntrack, N)     # N×ntrack scratch
    V = similar(Hinit)
    U = similar(Hinit)
    Udt    = similar(Hinit)
    phase_vec   = zeros(ComplexF64, N)
    Vsort = similar(Hinit)                # sorted eigenvectors
    Esort = zeros(Float64, N)             # sorted eigenvalues

    ψ      = zeros(ComplexF64, N, ntrack)      # current kets
    ψ_tmp  = similar(ψ)

    # ── data containers you want to return / analyse later ──
    ψt     = Array{ComplexF64,3}(undef, N, ntrack, nt)
    Et     = Array{Float64,2}(undef, N, nt)
    Pt     = Array{Float64,3}(undef, ntrack, N, nt)

    Ω0, ω0, δ0, vz, z0 = p
    Ez = Ez_interp(z0 + vz * t_array[1])
    hamiltonian_full_nocoupling!(Hinit, Ez)

    copyto!(buffer, Hinit)
    Eref, _ = LinearAlgebra.LAPACK.syev!('V', 'U', buffer)
    copyto!(Vref, buffer)
    copyto!(Vrefinit, buffer)

    ψ .= @view Vref[:, states_track]

    detuning = zeros(Float64, N)
    @views detuning[5:16] .= -(ω0 + δ0)

    ψt[:, :, 1] .= ψ
    Et[:, 1]    .= Eref
    calculate_probabilities!(@view(Pt[:, :, 1]), ψ, Vref, tmp_Nk)

    for i in 2:length(t_array)
        dt = t_array[i] - t_array[i-1]
        t = t_array[i]

        z = z0 + vz * t
        Ez = Ez_interp(z)

        hamiltonian_full_nocoupling!(H, Ez)
        Ωval = Ω0/2 * gaussian_peak(z0 + vz * t_array[i], zμ0, σμ)
        coupling_full!(C, Ωval)

        copyto!(buffer, H)
        D, _ = LinearAlgebra.LAPACK.syev!('V', 'U', buffer)
        copyto!(V, buffer)

        reorder_evecs!(Vsort, Esort, V, D, Vref)

        # D is already sorted in ascending order
        H .+= C

        apply_transform!(Hrot, H, V, buffer)
        Hrot .+= diagm(detuning)

        Drot, _ = LinearAlgebra.LAPACK.syev!('V', 'U', Hrot)
        copyto!(Vrot, Hrot)

        # time-evolution operator Udt
        mul!(U, V, Vrot)
        @. phase_vec = cis(-Drot * dt)
        mul!(tmp_NN, U, Diagonal(phase_vec))
        mul!(Udt, tmp_NN, U')

        # advance ψ safely:  ψ ← Udt * ψ
        mul!(ψ_tmp, Udt, ψ)                     # no aliasing (N×ntrack)
        copyto!(ψ, ψ_tmp)

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

Ω0 = 1e-1 * Γ
ω0 = 83817990535.81007
δ0 = 0.0
vz = 184
z0 = -0.25
zstop = 0.2
tmax = (zstop - z0) / vz
N = 36
Δt = 1e-7
t_array = LinRange(0.0, tmax, 10001)
states_track = [1]

p = (Ω0, ω0, δ0, vz, z0)

@time ψt, Et, Pt = evolve(p, t_array, states_track, N);

plot(t_array*vz .+ z0, Pt[1,1:16,:]')

argmax(Pt[1,:,1]), argmax(Pt[1,:,end])