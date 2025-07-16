import numpy as np

X = 11000  # example size
matrix_a = np.random.rand(X, X)  # uniform floats in [0.0, 1.0)
matrix_b = np.random.rand(X, X)  # uniform floats in [0.0, 1.0)

# Save to file
np.save('data/matrices/matrix_a.npy', matrix_a)
np.save('data/matrices/matrix_b.npy', matrix_b)

# multiply and save the result
matrix_c = np.dot(matrix_a, matrix_b)
np.save('data/matrices/matrix_c.npy', matrix_c)


# a = np.load('data/matrices/matrix_a.npy')
# b = np.load('data/matrices/matrix_b.npy')
# c = np.load('data/matrices/matrix_c.npy')
