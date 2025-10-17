import scipy.io as sio
import numpy as np
import argparse

def compare_matrices(path1, path2, fields=("data",), rtol=1e-4, atol=1e-8, equal_nan=True):
    try:
        mat1 = sio.loadmat(path1, simplify_cells=True)
        mat2 = sio.loadmat(path2, simplify_cells=True)
    except TypeError:
        mat1 = sio.loadmat(path1)
        mat2 = sio.loadmat(path2)

    for m in (mat1, mat2):
        for k in ("__header__", "__version__", "__globals__"):
            if k in m:
                del m[k]

    def compare_arrays(A, B):
        if A.shape != B.shape:
            print("Different shapes:", A.shape, B.shape)
            return
            
        # This condition now correctly routes to the right logic path
        if A.dtype == object or B.dtype == object:
            print("Comparing as object array (element-wise)...")
            eqs = []
            for idx in np.ndindex(A.shape):
                try:
                    # This handles cases where elements are arrays themselves
                    eqs.append(np.allclose(A[idx], B[idx], rtol=rtol, atol=atol, equal_nan=equal_nan))
                except (TypeError, ValueError):
                    # Fallback for non-numeric types like strings
                    eqs.append(A[idx] == B[idx])
            frac = np.mean(eqs) if eqs else 1.0
            print("All elements close:", all(eqs))
            print("Fraction of equal elements:", frac)
        else:
            print("Comparing as numerical array (vectorized)...")
            all_close = np.allclose(A, B, rtol=rtol, atol=atol, equal_nan=equal_nan)
            close_mask = np.isclose(A, B, rtol=rtol, atol=atol, equal_nan=equal_nan)
            print("All elements close:", all_close)
            print("Fraction of equal elements:", np.mean(close_mask))
            if np.issubdtype(A.dtype, np.number) and np.issubdtype(B.dtype, np.number):
                diff = A - B
                print("Max abs diff:", np.nanmax(np.abs(diff)))
                print("Mean abs diff:", np.nanmean(np.abs(diff)))

    for f in fields:
        if f not in mat1:
            print(f"[{f}] missing in file1")
            continue
        if f not in mat2:
            print(f"[{f}] missing in file2")
            continue
        
        A = np.asarray(mat1[f])
        B = np.asarray(mat2[f])

        print(f"\n=== Field: {f} ===")
        print(f"Data type: {A.dtype}")
        print("Shape A:", A.shape)
        print("Shape B:", B.shape)
        compare_arrays(A, B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two matrices from .mat files")
    parser.add_argument("mat1", nargs="?", help="Path to first .mat file")
    parser.add_argument("mat2", nargs="?", help="Path to second .mat file")
    parser.add_argument("--rtol", type=float, default=1e-8, help="Relative tolerance (default: 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance (default: 1e-8)")
    args = parser.parse_args()
    
    HARDCODED_MAT1 = "../results/frame1_coeff_matlab.mat"
    HARDCODED_MAT2 = "../results/frame1_coeff_python.mat"
    fields=("data",)
    
    # HARDCODED_MAT1 = "/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/results/frame1_params_matlab.mat"
    # HARDCODED_MAT2 = "/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/results/frame1_params_python.mat"
    # fields=("ListC", "FlagsC", "weightsC")
    
    # Priority: CLI args > hardcoded
    mat1_path = args.mat1 if args.mat1 is not None else HARDCODED_MAT1
    mat2_path = args.mat2 if args.mat2 is not None else HARDCODED_MAT2

    if mat1_path is None or mat2_path is None:
        raise ValueError("Matrix file paths not provided. Use command line args or set HARDCODED_MAT1/2.")

    compare_matrices(mat1_path, mat2_path, fields, rtol=args.rtol, atol=args.atol)