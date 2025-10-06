"""
RAHT Debug Script - Python Version
Simple 8-point cube test for debugging RAHT transform
"""

import torch
import numpy as np
import time
from scipy.io import savemat, loadmat

from RAHT import RAHT, RAHT_optimized, RAHT_batched
from iRAHT import inverse_RAHT
from RAHT_param import RAHT_param
from voxelize import voxelize_pc

def sanity_check_vector(T: torch.Tensor, C: torch.Tensor, rtol=1e-5, atol=1e-8) -> bool:
    """
    Sanity check: max(T) == sqrt(N) * mean(C)
    T, C: 1D tensors of shape [N]
    """
    assert T.dim() == 1 and C.dim() == 1 and T.size(0) == C.size(0), \
        "T and C must be 1D with same length"
    N = T.size(0)
    
    lhs = T.max()
    rhs = torch.sqrt(torch.tensor(float(N), dtype=C.dtype, device=C.device)) * C.mean()
    
    return torch.allclose(lhs, rhs, rtol=rtol, atol=atol)

def rgb_to_yuv_torch(rgb_tensor):
    """Converts a PyTorch tensor of RGB colors [0,255] to YUV."""
    rgb_tensor = rgb_tensor.float()
    conversion_matrix = torch.tensor([
        [0.2126, 0.7152, 0.0722],
        [-0.1146, -0.3854, 0.5000],
        [0.5000, -0.4542, -0.0458]
    ]).to(rgb_tensor.device)
    
    yuv = torch.matmul(rgb_tensor, conversion_matrix.T)
    yuv[:, 1:] += 128.0 # Add offset to U and V
    return yuv

# Configuration
DEBUG = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("RAHT DEBUG MODE - Python Version")
print("=" * 60)
print(f"Device: {device}")

# Generate sample colored point cloud
N = 10000
V = torch.rand(N, 3) * 10  # xyz in [0,10]
C = torch.randint(0, 256, (N, 3), dtype=torch.float32)  # RGB attributes
PC = torch.cat([V, C], dim=1)

# Set voxelization parameters
param = {
    'vmin': [0, 0, 0],          # lower corner of bounding box
    'width': 10,                 # cube side length
    'J': 4,                      # octree depth -> 16^3 voxels
    'writeFileOut': False,       # disable file output
    'filename': 'example'        # base name if file writing enabled
}

# Voxelize point cloud
PCvox, PCsorted, voxel_indices, DeltaPC = voxelize_pc(PC, param)

# Extract voxelized and sorted coordinates and attributes
voxel_size = param['width'] / (2 ** param['J'])
vmin_tensor = torch.tensor(param['vmin'], dtype=V.dtype, device=V.device)
V0s = PCsorted[:, 0:3] - vmin_tensor  # sorted coordinates
V0i = torch.floor(V0s / voxel_size)   # sorted voxel indices
V = V0i[voxel_indices, :]             # Morton-ordered voxel coords
Cvox = PCvox[:, 3:]                   # already Morton-ordered colors
C = rgb_to_yuv_torch(Cvox)  # Convert to YUV color space

J = param['J']

print("Input Configuration:")
print(f"  Number of points: {V.shape[0]}")
print(f"  Octree depth J: {J}")
print(f"  Points (V):")
print(V.shape)
print(f"  Colors (C):")
print(C.shape)
print()

# Minimum corner and width
origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
width = 2**J

print("Processing parameters:")
print(f"  origin: {origin.tolist()}")
print(f"  width: {width}")
print(f"  Voxel size Q: {width/2**J:.4f}")
print()

# Load RAHT parameters
saved_params = loadmat("../results/debug_params.mat", simplify_cells=True)
ListC = saved_params["ListC"]
FlagsC = saved_params["FlagsC"]
weightsC = saved_params["weightsC"]
def to_tensor_list(seq, dtype, device):
    out = []
    for x in seq:
        a = np.asarray(x)            # unwrap scalar/object to ndarray
        a = np.squeeze(a)            # drop stray singleton dims
        t = torch.as_tensor(a, dtype=dtype, device=device)
        out.append(t)
    return out
ListC    = to_tensor_list(ListC,    dtype=torch.int64,   device=device)
ListC    = [t - 1 for t in ListC]  # convert to 0-based indexing
FlagsC   = to_tensor_list(FlagsC,   dtype=torch.bool, device=device)
weightsC = to_tensor_list(weightsC, dtype=torch.int64,   device=device)

# Load input data
saved_input = loadmat("../results/debug_input.mat", simplify_cells=True)
V = torch.as_tensor(saved_input["V"], dtype=torch.float32, device=device)
C = torch.as_tensor(saved_input["C"], dtype=torch.float32, device=device)
J = int(saved_input["J"])
N = V.shape[0]

print(f"  Number of levels: {len(ListC)}")
for level in range(len(ListC)):
    print(f"    Level {level}: ListC size={len(ListC[level])}, "
          f"FlagsC size={len(FlagsC[level])}, "
          f"weightsC size={len(weightsC[level])}")
print()

# Move data to device
ListC = [t.to(device) for t in ListC]
FlagsC = [t.to(device) for t in FlagsC]
weightsC = [t.to(device) for t in weightsC]
C = C.to(device)

# Test all RAHT variants
VARIANTS = {
    "RAHT":             lambda C,L,F,W,d: RAHT(C, L, F, W, d),
    "RAHT_optimized":   lambda C,L,F,W,d: RAHT_optimized(C, L, F, W, d),
    "RAHT_batched":     lambda C,L,F,W,d: RAHT_batched(C, L, F, W, d),
}

print("=" * 60)
print("Testing RAHT Variants")
print("=" * 60)

for variant_name, raht_fn in VARIANTS.items():
    print(f"\n--- {variant_name} ---")
    
    # Apply RAHT transform
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.time()
    
    result = raht_fn(C, ListC, FlagsC, weightsC, device)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t3 = time.time()
    
    # Extract coefficient matrix
    Coeff = result[0] if isinstance(result, (tuple, list)) and len(result) == 2 else result
    
    transform_time = (t3 - t2) * 1000
    print(f"  Transform time: {transform_time:.2f} ms")
    print(f"  Coefficient matrix size: {Coeff.shape}")
    print(f"  L2-norm of input C: {torch.norm(C).item():.6f}")
    print(f"  L2-norm of Coeff: {torch.norm(Coeff).item():.6f}")
    
    # Sanity checks
    print(f"  Sanity check (channel 0): {sanity_check_vector(Coeff[:, 0], C[:, 0])}")
    print(f"  Sanity check (channel 1): {sanity_check_vector(Coeff[:, 1], C[:, 1])}")
    print(f"  Sanity check (channel 2): {sanity_check_vector(Coeff[:, 2], C[:, 2])}")
    
    # Apply inverse RAHT
    t4 = time.time()
    C_recon = inverse_RAHT(Coeff, ListC, FlagsC, weightsC, device)
    t5 = time.time()
    
    inverse_time = (t5 - t4) * 1000
    print(f"  Inverse transform time: {inverse_time:.2f} ms")
    
    # Verify reconstruction
    rtol, atol = 1e-4, 1e-10
    if torch.allclose(C, C_recon, rtol=rtol, atol=atol):
        print(f"  ✓ Reconstruction check: PASSED (rtol={rtol}, atol={atol})")
    else:
        print(f"  ✗ Reconstruction check: FAILED")
    
    # Compute reconstruction error
    diff = C - C_recon
    frobenius_error = torch.norm(diff, p='fro').item()
    max_error = diff.abs().max().item()
    mean_error = diff.abs().mean().item()
    
    print(f"  Frobenius norm error: {frobenius_error:.10e}")
    print(f"  Max absolute error: {max_error:.10e}")
    print(f"  Mean absolute error: {mean_error:.10e}")
    
    # Save results
    savemat(f'../results/debug_{variant_name}_coeff.mat', 
            {'Coeff': Coeff.detach().cpu().numpy()})
    savemat(f'../results/debug_{variant_name}_recon.mat', 
            {'C_recon': C_recon.detach().cpu().numpy(),
             'frobenius_error': frobenius_error,
             'max_error': max_error,
             'mean_error': mean_error})

# Save common parameters (once)
params_dict = {
    'ListC': [t.detach().cpu().numpy() for t in ListC],
    'FlagsC': [t.detach().cpu().numpy() for t in FlagsC],
    'weightsC': [t.detach().cpu().numpy() for t in weightsC]
}
savemat('../results/debug_params.mat', params_dict)

input_dict = {
    'V': V.detach().cpu().numpy(),
    'C': C.detach().cpu().numpy(),
    'J': J
}
savemat('../results/debug_input.mat', input_dict)

print()
print("=" * 60)
print("Debug data saved to ../results/")
print("Files:")
print("  - debug_input.mat")
print("  - debug_params.mat")
for variant_name in VARIANTS.keys():
    print(f"  - debug_{variant_name}_coeff.mat")
    print(f"  - debug_{variant_name}_recon.mat")
print("=" * 60)