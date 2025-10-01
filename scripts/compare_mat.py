import scipy.io as sio
import numpy as np
import argparse

def compare_matrices(path1, path2, rtol=1e-5, atol=1e-8):
    mat1 = sio.loadmat(path1)
    mat2 = sio.loadmat(path2)

    A = mat1['data']
    B = mat2['data']

    print("Shape A:", A.shape)
    print("Shape B:", B.shape)
    print("Same shape:", A.shape == B.shape)

    rtol = 1e-5  # relative tolerance
    atol = 1e-8  # absolute tolerance
    print("All elements close:", np.allclose(A, B, rtol=rtol, atol=atol))

    # Elementwise closeness (for debugging which entries differ)
    close_mask = np.isclose(A, B, rtol=rtol, atol=atol)
    print("Fraction of equal elements:", np.mean(close_mask))

    # Difference summary
    diff = A - B
    print("Max abs diff:", np.max(np.abs(diff)))
    print("Mean abs diff:", np.mean(np.abs(diff)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two matrices from .mat files")
    parser.add_argument("mat1", nargs="?", help="Path to first .mat file")
    parser.add_argument("mat2", nargs="?", help="Path to second .mat file")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance (default: 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance (default: 1e-8)")
    args = parser.parse_args()
    
    
    HARDCODED_MAT1 = "/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/results/frame1_coeff_matlab.mat"
    HARDCODED_MAT2 = "/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/results/frame1_coeff_python_RAHT_optimized.mat"
    
    # Priority: CLI args > hardcoded
    mat1_path = args.mat1 if args.mat1 is not None else HARDCODED_MAT1
    mat2_path = args.mat2 if args.mat2 is not None else HARDCODED_MAT2

    if mat1_path is None or mat2_path is None:
        raise ValueError("Matrix file paths not provided. Use command line args or set HARDCODED_MAT1/2.")

    compare_matrices(mat1_path, mat2_path, rtol=args.rtol, atol=args.atol)