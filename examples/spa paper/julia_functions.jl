@inline function zero_matrix!(M)
    n, m = size(M)
    T = eltype(M)
    z = zero(T)
    @inbounds @simd for i in 1:n, j in 1:m
        M[i, j] = z
    end
    nothing
end

function commutator!(C, H, ρ)
    n = size(H, 1)

    @inbounds begin
        # zero out C
        zero_matrix!(C)

        T = eltype(H)
        z = zero(T)

        # compute upper triangle of -im*[H,ρ] and conjugate mirror
        for i in 1:n
            for j in i:n
                s = z
                @simd for k in 1:n
                    s += H[i, k]*ρ[k, j] - ρ[i, k]*H[k, j]
                end
                val = -im * s
                C[i, j] = val
                if i != j
                    C[j, i] = conj(val)
                end
            end
        end
    end

    nothing
end