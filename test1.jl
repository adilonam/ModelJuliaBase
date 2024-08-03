function mmx_mult(A, B)
    if ndims(A) == 1 && ndims(B) == 1
        # Dot product for two vectors (1D arrays)
        return dot(A, B)
    elseif ndims(A) == 1 && ndims(B) == 2
        # A is a row vector, B is a matrix (2D array)
        return A' * B
    elseif ndims(A) == 2 && ndims(B) == 1
        # A is a matrix (2D array), B is a column vector
        return A * B
    elseif ndims(A) == 2 && ndims(B) == 2
        # Matrix multiplication
        return A * B
    elseif ndims(A) == 1 && ndims(B) == 3
        # A is a row vector, B is a 3D array
        return reshape(A, 1, :) * reshape(B, size(B, 1), :) |> x -> reshape(x, 1, size(B, 2), size(B, 3))
    elseif ndims(A) == 3 && ndims(B) == 1
        # A is a 3D array, B is a column vector
        return reshape(A, size(A, 1) * size(A, 2), size(A, 3)) * B
    elseif ndims(A) == 2 && ndims(B) == 3
        # A is a matrix, B is a 3D array
        return map(slice -> A * slice, eachslice(B, dims=3))
    elseif ndims(A) == 3 && ndims(B) == 2
        # A is a 3D array, B is a matrix
        return map(slice -> slice * B, eachslice(A, dims=3))
    elseif ndims(A) == 3 && ndims(B) == 3
        # 3D matrix multiplication by slices
        if size(A, 2) == size(B, 1)
            C = zeros(Float64, size(A, 1), size(B, 2), size(A, 3))
            for i in 1:size(A, 3)
                C[:, :, i] = A[:, :, i] * B[:, :, i]
            end
            return C
        else
            throw(DimensionMismatch("Dimensions of A and B are not compatible for 3D multiplication"))
        end
    else
        throw(ArgumentError("Unsupported dimensions for multiplication"))
    end
end

# Example usage
A_3d_2d = randn(71, 165, 65)
B_2d = randn(165, 165) 
C_3d_2d = mmx_mult(A_3d_2d, B_2d)
println("Result of 3D (71, 165, 63) * 2D (165, 165) with 'tn': Size $(size(C_3d_2d))")

# 3D (71, 165, 63) and 3D (165, 71, 63) with mod='tt'
A_3d = randn(71, 165, 63)
B_3d = randn(165, 71, 63)
C_3d = mmx_mult(A_3d, B_3d)
println("Result of 3D (71, 165, 63) * 3D (165, 71, 63) with 'tt': Size $(size(C_3d))")