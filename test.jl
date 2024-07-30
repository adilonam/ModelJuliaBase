using LinearAlgebra
using Base.Threads

function mmx_mult(A::Array{Float64}, B::Array{Float64}, mod::String="n")::Array{Float64}
    dimsA = ndims(A)
    dimsB = ndims(B)
    
    # Handle transpositions based on the 'mod' parameter
    if mod == "tn"
        A_trans = A
        B_trans = (dimsB == 2) ? transpose(B) : permutedims(B, (2, 1, 3))
    elseif mod == "nt"
        A_trans = (dimsA == 3) ? permutedims(A, (2, 1, 3)) : transpose(A)
        B_trans = (dimsB == 2) ? B : permutedims(B, (2, 1, 3))
    elseif mod == "tt"
        A_trans = (dimsA == 3) ? permutedims(A, (2, 1, 3)) : transpose(A)
        B_trans = (dimsB == 2) ? transpose(B) : permutedims(B, (2, 1, 3))
    elseif mod == "n"
        A_trans = A
        B_trans = B
    else
        throw(ArgumentError("Invalid mod value: $mod. Expected 'n', 'tn', 'nt', or 'tt'."))
    end

    # Matrix multiplication based on dimensions
    if dimsA == 3 && dimsB == 2
        # A is 3D and B is 2D
        (m, n, p) = size(A_trans)
        (nB, q) = size(B_trans)
        if n != nB
            throw(DimensionMismatch("Inner dimensions must match: A (n=$n), B (nB=$nB)"))
        end
        C = zeros(Float64, m, q, p)

        @threads for i in 1:p
            C[:, :, i] = A_trans[:, :, i] * B_trans
        end

    elseif dimsA == 3 && dimsB == 3
        # Both A and B are 3D
        (m, n, p) = size(A_trans)
        (nB, q, pB) = size(B_trans)
        if n != nB || p != pB
            throw(DimensionMismatch("Dimensions must match: A (n=$n, p=$p), B (nB=$nB, pB=$pB)"))
        end
        C = zeros(Float64, m, q, p)

        @threads for i in 1:p
            C[:, :, i] = A_trans[:, :, i] * B_trans[:, :, i]
        end

    else
        throw(ArgumentError("Unsupported combination of dimensions: A (dims=$dimsA), B (dims=$dimsB)"))
    end

    return C
end

# Example usage
# 3D (71, 165, 63) and 2D (165, 165) with mod='tn'
A_3d_2d = randn(71, 165, 63)
B_2d = randn(165, 165)
C_3d_2d = mmx_mult(A_3d_2d, B_2d)
println("Result of 3D (71, 165, 63) * 2D (165, 165) with 'tn': Size $(size(C_3d_2d))")

# 3D (71, 165, 63) and 3D (165, 71, 63) with mod='tt'
A_3d = randn(71, 165, 63)
B_3d = randn(165, 71, 63)
C_3d = mmx_mult(A_3d, B_3d)
println("Result of 3D (71, 165, 63) * 3D (165, 71, 63) with 'tt': Size $(size(C_3d))")
