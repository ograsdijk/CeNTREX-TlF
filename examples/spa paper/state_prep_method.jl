using linearAlgebra

include("electric_field.jl")
include("couplings.jl")
include("hamiltonian_nocoupling.jl")

field_path = "c:/Users/Olivier/Anaconda3/envs/centrex-eql-testing/Lib/site-packages/state_prep/electric_fields/Electric field components vs z-position_SPA_ExpeVer.csv"

function reorder_evecs(V_in::AbstractMatrix{<:ComplexF64},
    E_in::AbstractVector{<:Real},
    V_ref::AbstractMatrix{<:ComplexF64})
    # Compute absolute value of overlap matrix
    overlap = abs.(V_in' * V_ref)

    # For each row (eigenvector), find the index of the maximum overlap
    max_indices = map(i -> argmax(view(overlap, i, :)), 1:size(overlap, 1))

    # Sort eigenvectors by their maximum-overlap indices
    sorted_indices = sortperm(max_indices)

    # Reorder eigenvalues and eigenvectors accordingly
    E_out = E_in[sorted_indices]
    V_out = V_in[:, sorted_indices]

    return E_out, V_out
end

function evolve(p, tarray)
    Ω0, δ0, vz, z0 = p
    Ez = Ez_interp(z0)
    Hinit = hamiltonian_nocoupling!(Hinit, Ez)

    copyto!(buffer, Hinit)
    vals, vecs = LinearAlgebra.LAPACK.syev!('V', 'U', buffer)


    for i in 2:length(tarray)
        dt = t_array[i+1] - t_array[i]
        t = tarray[i]

        z = z0 + vz * t
        Ez = Ez_interp(z)

        hamiltonian_nocoupling!(H, Ez)
        couplings!(C, Ω0)

        copyto!(buffer, H)
        D, V = LinearAlgebra.LAPACK.syev!('V', 'U', buffer)
        # D is already sorted in ascending order
        Htot .= H .+ C
        transform!(Hrot, H + C, V, buffer)

        Drot, Vrot = LinearAlgebra.LAPACK.syev!('V', 'U', Hrot)

        U = V * Vrot

        phase = exp.(-1im .* Drot .* dt)

        Aphase = U .* permutedims(phase)

        Udt = Aphase * U'

        psis = psis * transpose(Udt)


    end
end

const Ez_interp = make_Ez_interp(field_path)

const σμ = 1.078e-2
const zμ0 = 0.0
const zμ1 = 25.4e-3 * 1.125
const Γ = 2π * 1.56e6

Ω0 = 1e-1 * Γ
δ0 = 5.26e6 * 2π
vz = 184
z0 = -0.25
zstop = 0.2
tmax = (zstop - z0) / vz
N = 36

const buffer = zeros(ComplexF64, N, N)
const Hrot = zeros(ComplexF64, N, N)
const H = zeros(ComplexF64, N, N)
const Htot = zeros(ComplexF64, N, N)
const C = zeros(ComplexF64, N, N)

p = (Ω0, δ0, vz, z0)