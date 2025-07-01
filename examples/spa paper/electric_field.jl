using CSV
using DataFrames
using Interpolations

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