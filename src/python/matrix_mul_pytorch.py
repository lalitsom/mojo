import torch
import numpy as np
import time

def matrix_multiply_pytorch(A, B):
    """
    Performs matrix multiplication using PyTorch.
    """
    return torch.matmul(A, B)

def main():
    print(f"Loading two {matrix_a.shape[0]}x{matrix_a.shape[1]} matrices using NumPy...")
    
    # Load data using NumPy as it's efficient for reading text files
    matrix_a_np = np.load('data/matrices/matrix_a.npy')
    matrix_b_np = np.load('data/matrices/matrix_b.npy')
    actual_result_np = np.load('data/matrices/matrix_c.npy')

    # --- PyTorch Specific Setup ---
    # 1. Convert NumPy arrays to PyTorch tensors
    matrix_a = torch.from_numpy(matrix_a_np)
    matrix_b = torch.from_numpy(matrix_b_np)

    # 2. Check for GPU and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Move tensors to the selected device (GPU or CPU)
    matrix_a = matrix_a.to(device)
    matrix_b = matrix_b.to(device)
    
    print("Performing matrix multiplication with PyTorch...")

    # --- Accurate Timing for CPU/GPU ---
    # For GPU, we need to synchronize to get accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()

    # Perform the matrix multiplication
    matrix_c = matrix_multiply_pytorch(matrix_a, matrix_b)

    # Ensure the operation is complete before stopping the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # --- Verification ---
    # Move result back to CPU to compare with NumPy
    matrix_c_np = matrix_c.cpu().numpy()
    
    print("\n--- Slices for Verification ---")
    print(f"Matrix A (first 2 elements): {matrix_a_np[0][0:2]}")
    print(f"Matrix B (first 2 elements): {matrix_b_np[0][0:2]}")
    print(f"Calculated C (first 2 elements): {matrix_c_np[0][0:2]}")
    print(f"Actual Result (first 2 elements): {actual_result_np[0][0:2]}")
    
    # Use np.allclose for robust floating-point comparison
    are_close = np.allclose(matrix_c_np, actual_result_np)

    print(f"\nMatrices are numerically close: {are_close}")

    # --- Final Results ---
    time_taken = end_time - start_time
    print(f"Matrix multiplication completed in {time_taken:.6f} seconds.")
    print(f"Result matrix shape: {matrix_c.shape}")

if __name__ == "__main__":
    main()
