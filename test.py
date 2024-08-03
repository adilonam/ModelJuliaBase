import numpy as np

# 2D matrix
A = np.random.rand(3,2)  # Shape (3, 2)

# 3D matrix
B = np.random.rand(1, 4, 5)  # Shape (2, 4, 5)

# Matrix multiplication of A with each 2x4 slice of B
C = np.einsum('ij,jkl->ikl', A, B)  # Resulting shape (3, 4, 5)

# Print size of C
print("Size of C:", C.shape)
