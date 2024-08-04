using Statistics
bsxfun(f, A, B) = broadcast(f, A, B)



# # Sample array of shape (1, 4, 69)
# arr = rand(1, 4, 69)

# # Taking the mean across the first two dimensions
# result = mean(arr, dims=(1, 2))

# # Reshaping to (69, 1)
# result_reshaped = reshape(result, 69, 1)

# println(size(result))

function squeeze(a)
    return dropdims(a, dims = tuple(findall(size(a) .== 1)...))
end





A = squeeze(mean(rand(1, 4, 69), dims=(1, 2)))
B = rand(1, 59, 165)

println(size(A))
C = bsxfun(-, A, B)
println("size of c  = ", size(C))