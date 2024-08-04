function mmx_mult(A, B, mod::String="n")
    dimsA = ndims(A)
    dimsB = ndims(B)

    # Handle transpositions based on the 'mod' parameter
    if mod == "tn"
        A_trans = A
        B_trans = (dimsB == 2) ? transpose(B) : permutedims(B, (2, 1, 3))
    elseif mod == "nt"
        A_trans = (dimsA == 3) ? permutedims(A, (2, 1, 3)) : transpose(A)
        B_trans = B
    elseif mod == "tt"
        A_trans = (dimsA == 3) ? permutedims(A, (2, 1, 3)) : transpose(A)
        B_trans = (dimsB == 2) ? transpose(B) : permutedims(B, (2, 1, 3))
    elseif mod == "n"
        A_trans = A
        B_trans = B
    else
        throw(ArgumentError("Invalid mod value: $mod. Expected 'n', 'tn', 'nt', or 'tt'."))
    end

    if ndims(A_trans) == 1 && ndims(B_trans) == 1
        return dot(A_trans, B_trans)
    elseif ndims(A_trans) == 1 && ndims(B_trans) == 2
        return A_trans' * B_trans
    elseif ndims(A_trans) == 2 && ndims(B_trans) == 1
        return A_trans * B_trans
    elseif ndims(A_trans) == 2 && ndims(B_trans) == 2
        return A_trans * B_trans
    elseif ndims(A_trans) == 1 && ndims(B_trans) == 3
        return reshape(A_trans, 1, :) * B_trans
    elseif ndims(A_trans) == 3 && ndims(B_trans) == 1
        t = map(slice -> slice * reshape(B_trans, :, 1), eachslice(A_trans, dims=3))
        stacked_t = cat(t..., dims=3)
        return stacked_t
    elseif ndims(A_trans) == 2 && ndims(B_trans) == 3
        t = map(slice -> A_trans * slice, eachslice(B_trans, dims=3))
        stacked_t = cat(t..., dims=3)
        return stacked_t
    elseif ndims(A_trans) == 3 && ndims(B_trans) == 2
        t = map(slice -> slice * B_trans, eachslice(A_trans, dims=3))
        stacked_t = cat(t..., dims=3)
        return stacked_t
    elseif ndims(A_trans) == 3 && ndims(B_trans) == 3
        t = map((Aslice, Bslice) -> Aslice * Bslice, eachslice(A_trans, dims=3), eachslice(B_trans, dims=3))
        stacked_t = cat(t..., dims=3)
        return stacked_t
    else
        throw(ArgumentError("Unsupported dimensions for multiplication"))
    end
end

# Example usage
A_3d_2d = randn(71, 165)
B_2d = randn(165,14, 56) 
C_3d_2d = mmx_mult(A_3d_2d, B_2d)
println("Result of 3D (71, 165, 63) * 2D (165, 165): Size $(size(C_3d_2d))")
# Expected size of C is (71, 165, 65)
