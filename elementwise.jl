using LinearAlgebra
using Base.Threads

function mmx_mult(A::Array{Float64}, B::Array{Float64})::Array{Float64}
    dimsA = ndims(A)
    dimsB = ndims(B)

    if dimsA == 3 && dimsB == 2
        # A is 3D and B is 2D
        (m, n, p) = size(A)
        (nB, q) = size(B)
        if n != nB
            throw(DimensionMismatch("Inner dimensions must match: A (n=$n), B (nB=$nB)"))
        end
        C = zeros(Float64, m, q, p)

        @threads for i in 1:p
            C[:, :, i] = A[:, :, i] * B
        end

    elseif dimsA == 3 && dimsB == 3
        # Both A and B are 3D
        (m, n, p) = size(A)
        (nB, q, pB) = size(B)
        if n != nB || p != pB
            throw(DimensionMismatch("Dimensions must match: A (n=$n, p=$p), B (nB=$nB, pB=$pB)"))
        end
        C = zeros(Float64, m, q, p)

        @threads for i in 1:p
            C[:, :, i] = A[:, :, i] * B[:, :, i]
        end

    else
        throw(ArgumentError("Unsupported combination of dimensions: A (dims=$dimsA), B (dims=$dimsB)"))
    end

    return C
end