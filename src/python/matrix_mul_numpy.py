import numpy as np
import time

def matrix_multiply(A, B):
    return np.dot(A, B)

def main():

    matrix_a = np.load('data/matrices/matrix_a.npy')
    matrix_b = np.load('data/matrices/matrix_b.npy')
    
    print(f"Loading two {matrix_a.shape[0]}x{matrix_a.shape[1]} matrices using NumPy...")
    
    
    actual_result = np.load('data/matrices/matrix_c.npy')

    print("Performing matrix multiplication...")
    # Start the timer
    start_time = time.time()

    # Perform the matrix multiplication
    matrix_c = matrix_multiply(matrix_a, matrix_b)

    # Stop the timer
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time
    
    
    print(matrix_a[0][0:2])
    print(matrix_b[0][0:2])
    print(matrix_c[0][0:2])
    print(actual_result[0][0:2])
    
    are_equal = np.array_equal(matrix_c, actual_result)

    print(f"Matrices are identical: {are_equal}")

    print(f"Matrix multiplication completed in {time_taken:.6f} seconds.")
    print(f"Result matrix shape: {matrix_c.shape}")

if __name__ == "__main__":
    main()
