import random
import time

fn create_matrix(rows: Int64, cols: Int64, min_val: Int64 = 0, max_val: Int64 = 100) -> List[List[Int64]]:
    """Creates a matrix (list of lists) with random integer values.

    Args:
        rows: The number of rows in the matrix.
        cols: The number of columns in the matrix.
        min_val: The minimum value for the random integers (inclusive).
        max_val: The maximum value for the random integers (inclusive).

    Returns:
        A list of lists representing the matrix with random integer values.
    """
    # random.random_si64(now())
    var matrix: List[List[Int64]] = []
    for r in range(rows):
        row = List[Int64]()
        for c in range(cols):
            row.append(random.random_si64(min_val, max_val))
        matrix.append(row)
    return matrix
# """
# Performs matrix multiplication of two matrices (list of lists)
# without using NumPy.

# matrix_a: list of lists (m x n)
# matrix_b: list of lists (n x p)

# Returns:
# list of lists (m x p)
# """

def multiply_matrices_naive(matrix_a: List[List[Int64]], matrix_b: List[List[Int64]]) -> List[List[Int64]]:
    """Multiplies two matrices using a naive approach with three nested loops."""
    # if not matrix_a or not matrix_b:
    #     raise ValueError("Both matrices must be non-empty.")
    # if len(matrix_a[0]) == 0 or len(matrix_b[0]) == 0:
    #     raise ValueError("Both matrices must have non-empty rows.")
    # if len(matrix_a) == 0 or len(matrix_b) == 0:
    #     raise ValueError("Both matrices must have non-empty columns.")
    

    var rows_a: Int64 = len(matrix_a)
    var cols_a: Int64 = len(matrix_a[0])
    var rows_b: Int64 = len(matrix_b)
    var cols_b: Int64 = len(matrix_b[0])

    # Check for valid dimensions for multiplication
    # if cols_a != rows_b:
    #     raise ValueError(f"Cannot multiply matrices: number of columns in A ({cols_a}) "
    #                      f"must equal number of rows in B ({rows_b}).")

    # Initialize the result matrix with zeros (m x p)
    var result_matrix = List[List[Int64]]()
    for _ in range(rows_a):
        var new_row = List[Int64]()
        for _ in range(cols_b):
            new_row.append(0)
        result_matrix.append(new_row)

    # Perform the multiplication using three nested loops
    for i in range(rows_a):        # Iterate over rows of matrix_a
        for j in range(cols_b):    # Iterate over columns of matrix_b
            for k in range(cols_a): # Iterate over columns of matrix_a (or rows of matrix_b)
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result_matrix

# --- Main execution for a large matrix size ---

# Define matrix dimensions
# Let's use 500x500 as 100,000 elements is too vague for matrix dimensions
# A 500x500 matrix has 250,000 elements.
# If you actually meant 1000x100, you can change it.
# 500 = 500 # For a 500x500 matrix
def main():
    var size: Int = 500
    print("Generating two square matrices ")
    var matrix_a: List[List[Int64]] = create_matrix(size, size)
    var matrix_b: List[List[Int64]] = create_matrix(size, size)
    print("Matrices generated.")

    print("Starting matrix multiplication of {500}x{500} matrices (without NumPy)...")
    var start: UInt = time.monotonic()
    var result: List[List[Int64]] = multiply_matrices_naive(matrix_a, matrix_b)
    var end: UInt = time.monotonic()
    var time_taken: UInt = end - start
    print(time_taken/1000000000, "seconds taken for multiplication.")
    # var time_taken: Int64 = time.time_function(multiply_matrices_naive(matrix_a, matrix_b))
    # result = multiply_matrices_naive(matrix_a, matrix_b)
    # end_time = time.time()

    # print("Matrix multiplication completed in {time_taken:.4f} seconds.")

# You can uncomment these lines to verify a small part of the result
# if 500 <= 10: # Only print for small matrices
#     print("\nMatrix A:")
#     for row in matrix_a:
#         print(row)
#     print("\nMatrix B:")
#     for row in matrix_b:
#         print(row)
#     print("\nResult Matrix (first 3x3 block):")
#     for r in result[:3]:
#         print(r[:3])
