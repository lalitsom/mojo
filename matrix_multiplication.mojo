from matrix_struct import Matrix
import random
import time

fn main() raises:
    var height = 1000
    var width = 1000
    var common_dim = 1000

    var matrix1_data = List[List[Float32]]()
    for i in range(height):
        var row = List[Float32]()
        for j in range(width):
            row.append(Float32(random.random_float64() * 10))
        matrix1_data.append(row)
    
    var matrix1 = Matrix(matrix1_data)

    var matrix2_data = List[List[Float32]]()
    for i in range(width):
        var row = List[Float32]()
        for j in range(common_dim):
            row.append(Float32(random.random_float64() * 10))
        matrix2_data.append(row)

    var matrix2 = Matrix(matrix2_data)

    print("Matrix 1:")
    print(matrix1.__str__())
    print("Matrix 2:")
    print(matrix2.__str__())
    start_time = time.monotonic()
    var result_matrix = matrix1 * matrix2
    end_time = time.monotonic()
    var time_taken = end_time - start_time
    print("Matrix multiplication completed in", time_taken/1000000000," seconds.")
    # print("Result Matrix:")
    # print(result_matrix.__str__())
