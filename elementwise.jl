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
        return A_trans * reshape(B_trans, :, 1)
    elseif ndims(A_trans) == 2 && ndims(B_trans) == 3
        return map(slice -> A_trans * slice, eachslice(B_trans, dims=3))
    elseif ndims(A_trans) == 3 && ndims(B_trans) == 2
        return map(slice -> slice * B_trans, eachslice(A_trans, dims=3))
    elseif ndims(A_trans) == 3 && ndims(B_trans) == 3
        if size(A_trans, 2) == size(B_trans, 1)
            C = zeros(Float64, size(A_trans, 1), size(B_trans, 2), size(A_trans, 3))
            for i in 1:size(A_trans, 3)
                C[:, :, i] = A_trans[:, :, i] * B_trans[:, :, i]
            end
            return C
        else
            throw(DimensionMismatch("Dimensions of A and B are not compatible for 3D multiplication"))
        end
    else
        throw(ArgumentError("Unsupported dimensions for multiplication"))
    end
end