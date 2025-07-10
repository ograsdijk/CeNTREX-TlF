using LinearAlgebra

@inline function zero_matrix!(M)
    n, m = size(M)
    T = eltype(M)
    z = zero(T)
    @inbounds begin
        for i in 1:n
            @simd for j in 1:m
                M[i, j] = z
            end
        end
    end
    return nothing
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

function commutator_mat!(C, A, B)
    @inbounds begin
        mul!(C, B, A)
        mul!(C, A, B, -1im, 1im)
    end
    return nothing
end

function commutator_mat!(C, A, B, buf)
    @inbounds begin
        mul!(C, A, B)
        mul!(buf, B, A)
        C .-= buf
        C .*= -im
    end
    return nothing
end

@inline function gaussian_peak(x::T, μ::T=zero(T), σ::T=one(T)) where {T<:AbstractFloat}
    Δ = x - μ
    return exp(-T(0.5) * (Δ / σ)^2)
end

@inline function apply_transform!(out, in, transform, buffer)
    @inbounds begin
        # buf = in * T
        mul!(buffer, in, transform)
        # out = T' * buf   (T' is adjoint(T), no allocation)
        mul!(out, adjoint(transform), buffer)
    end
    nothing
end

function transform_couplings!(out, in, v0, v1, buffer)
    @inbounds begin
        # buf = in * v0
        mul!(buffer, in, v0)
        # out = v1' * buf   (v1' is adjoint(v1), no allocation)
        mul!(out, adjoint(v1), buffer)
    end
    nothing
end