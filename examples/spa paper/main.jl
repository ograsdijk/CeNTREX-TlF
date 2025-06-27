using SIMD
using LinearAlgebra
using DifferentialEquations
using CSV
using DataFrames
using Interpolations
using MKL

include("julia_functions.jl")
include("hamiltonian.jl")
include("couplings.jl")

efield_path = "c:/Users/Olivier/Anaconda3/envs/centrex-eql-testing/Lib/site-packages/state_prep/electric_fields/Electric field components vs z-position_SPA_ExpeVer.csv"
# efield_path = "c:/Users/ogras/anaconda3/envs/centrex-eql/Lib/site-packages/state_prep/electric_fields/Electric field components vs z-position_SPA_ExpeVer.csv"


const σμ = 1.078e-2
const zμ0 = 0.0
const zμ1 = 25.4e-3 * 1.125

function make_Ez_interp(path::AbstractString)
    # 1) load data
    df = CSV.read(path, DataFrame)

    # 2) extract & convert with floating‐point division
    x = df[!, "Distance-250mm [mm]"] ./ 1000.0   # mm → m
    y = df[!, "Ez []"] ./ 100.0     # raw → your unit

    # 3) sanity‐checks for uniform spacing
    @assert length(x) ≥ 2 "Need at least two points for interpolation"
    Δx = x[2] - x[1]
    @assert Δx != 0 "Δx is zero—did you accidentally use .÷ instead of ./?"
    @assert all(abs.(diff(x) .- Δx) .< 1e-8) "x must be uniformly spaced"

    # 4) build & scale a cubic B-spline
    itp = interpolate(y, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, x[1]:Δx:x[end])

    # 5) wrap to fill with 0.0 outside the domain
    return extrapolate(sitp, 0.0)
end

const Ez_interp = make_Ez_interp(efield_path)

Γ = 2π*1.56e6
Ω0 = 1e-1*Γ
δ0 = 5.26e6 * 2π
vz = 184
z0 = -0.25
zstop = 0.2
tmax = (zstop - z0) / vz
N = 16
diag_idxs = [i + (i - 1) * N for i in 1:N]

u0 = zeros(ComplexF64, N, N)
u0[1,1] = 1.0 + 0.0im

const H = zeros(ComplexF64, size(u0))
const buffer = zeros(ComplexF64, size(u0))


function lindblad!(du, u, p, t)
    Ω0, δ0, vz, z0 = p

    z = vz * t + z0

    Ez = Ez_interp(z0 + vz * t)

    Ω0val = Ω0*gaussian_peak(z, zμ0, σμ)
    hamiltonian!(H, Ez, Ω0val, δ0)
    commutator_mat!(du, H, u)
    nothing
end


p = (Ω0, δ0, vz, z0)
prob = ODEProblem(
    lindblad!,
    u0,
    (0.0, tmax),
    p,
);

@time sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-7, save_idxs=diag_idxs, dtmax=1e-6);


pop = hcat(real(sol.u)...);


function prob_func(prob, i, repeat)
    p = (rabis[i], δ0, vz, z0)
    remake(prob, p=p)
end

function prob_func(prob, i, repeat)
    p = (Ω0, detunings[i], vz, z0)
    remake(prob, p=p)
end

detunings = (-6:0.25:10.0) .+ 5.25
detunings *= 2π*1e6

ens_prob = EnsembleProblem(prob, prob_func=prob_func);
@time sol = solve(
    ens_prob,
    Tsit5(),
    EnsembleSerial(),
    trajectories=length(detunings),
    save_idxs=diag_idxs,
    save_start=false,
    save_everystep=false,
    reltol=1e-5,
    abstol=1e-7,
);

pop = hcat([real(sol[i].u[1]) for i in 1:length(sol)]...);