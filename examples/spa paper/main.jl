using SIMD
using LinearAlgebra
using DifferentialEquations
using CSV
using DataFrames
using Interpolations
using MKL

include("julia_functions.jl")
include("hamiltonian.jl")

efield_path = "c:/Users/Olivier/Anaconda3/envs/centrex-eql-testing/Lib/site-packages/state_prep/electric_fields/Electric field components vs z-position_SPA_ExpeVer.csv"


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

Ez_interp = make_Ez_interp(efield_path)

Ω0 = 1e-3
Ω1 = 1e-3
δ0 = 0.0
δ1 = 0.0
vz = 184
z0 = -0.25
zstop = 0.2
tmax = (zstop - z0) / vz

u0 = zeros(ComplexF64, 64, 64)
u0[4,4] = 1.0 + 0.0im

const H = zeros(ComplexF64, 64, 64)
const buffer = zeros(ComplexF64, 64, 64)

function lindblad!(du, u, p, t)
    Ω0, Ω1, δ0, δ1, vz, z0 = p

    Ez = Ez_interp(z0 + vz * t)

    Ω0val = gaussian_peak(vz * t + z0, zμ0, σμ) .* Ω0
    Ω1val = gaussian_peak(vz * t + z0, zμ1, σμ) .* Ω1
    hamiltonian!(H, Ez, Ω0val, Ω1val, δ0, δ1)

    commutator_mat!(du, H, u, buffer)
    nothing
end

p = (Ω0, Ω1, δ0, δ1, vz, z0)
prob = ODEProblem(
    lindblad!,
    u0,
    (0.0, 1120e-6),
    p,
)

sol = solve(prob, Tsit5(), dtmax = 1e-6)