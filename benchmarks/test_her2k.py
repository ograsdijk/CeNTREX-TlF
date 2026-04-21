import sys
sys.stdout.reconfigure(encoding="utf-8")

from juliacall import Main as jl

jl.seval("using LinearAlgebra")
jl.seval("using LinearAlgebra.BLAS")

jl.seval("""
n = 65

# Create simple Hermitian matrices
A = randn(ComplexF64, n, n)
A = A + A'  # make Hermitian

B = randn(ComplexF64, n, n)
B = B + B'  # make Hermitian

C = zeros(ComplexF64, n, n)

println("typeof(A): ", typeof(A))
println("typeof(C): ", typeof(C))
println("size(A): ", size(A))
println("BLAS threads: ", BLAS.get_num_threads())
println("BLAS vendor: ", BLAS.get_config())

# Test her2k!
α = ComplexF64(0.0, 1.0)
β = 0.0

println("\\nBefore her2k!: norm(C) = ", norm(C))
BLAS.her2k!('U', 'N', α, B, A, β, C)
println("After her2k!:  norm(C) = ", norm(C))
println("C[1,1] = ", C[1,1])
println("C[1,2] = ", C[1,2])

# Verify against manual computation
D = α * B * A' + conj(α) * A * B'
println("\\nManual norm(D) = ", norm(D))
println("Upper triangle match: ", isapprox(C[1,2], D[1,2]))

# Test with mul! (the non-BLAS commutator)
E = zeros(ComplexF64, n, n)
mul!(E, B, A)
mul!(E, A, B, -1im, 1im)
println("\\nmul! commutator norm = ", norm(E))
println("mul! vs manual match: ", isapprox(norm(E), norm(D), rtol=1e-10))

# Timing
t_her2k = @elapsed for _ in 1:1000
    BLAS.her2k!('U', 'N', α, B, A, β, C)
end
println("\\nher2k! time: ", round(t_her2k/1000*1e6, digits=1), " μs/call")

t_mul = @elapsed for _ in 1:1000
    mul!(E, B, A)
    mul!(E, A, B, -1im, 1im)
end
println("mul! time: ", round(t_mul/1000*1e6, digits=1), " μs/call")
""")
