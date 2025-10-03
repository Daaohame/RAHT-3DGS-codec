import torch
import numpy as np
import time
import os
import glob
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

from data_util import read_ply_file
from RAHT import RAHT, RAHT_optimized, RAHT_batched, RAHT_fused_kernel
from iRAHT import inverse_RAHT
from RAHT_param import RAHT_param
# import RLGR_encoder

VARIANTS = {
    "RAHT":             lambda C,L,F,W,d: RAHT(C, L, F, W, d),
    "RAHT_optimized":   lambda C,L,F,W,d: RAHT_optimized(C, L, F, W, d),
    "RAHT_batched":     lambda C,L,F,W,d: RAHT_batched(C, L, F, W, d),
    # "RAHT_fused_kernel":lambda C,L,F,W,d: RAHT_fused_kernel(C, L, F, W, d),
}

DEBUG = True

def sanity_check_vector(T: torch.Tensor, C: torch.Tensor, rtol=1e-5, atol=1e-8) -> bool:
    """
    Sanity check: max(T) == sqrt(N) * mean(C)
    T, C: 1D tensors of shape [N]
    """
    assert T.dim() == 1 and C.dim() == 1 and T.size(0) == C.size(0), "T and C must be 1D with same length"
    N = T.size(0)

    lhs = T.max()
    rhs = torch.sqrt(torch.tensor(float(N), dtype=C.dtype, device=C.device)) * C.mean()

    return torch.allclose(lhs, rhs, rtol=rtol, atol=atol)

def save_mat(tensor: torch.Tensor, filename: str) -> None:
    savemat(filename, {"data": tensor.detach().cpu().numpy()})

def save_lists(filename, **kwargs):
    out = {}
    for key, tensor_list in kwargs.items():
        out[key] = [t.detach().cpu().numpy() for t in tensor_list]
    savemat(filename, out)

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

## ---------------------
## Configuration
## ---------------------
ply_list = ['/ssd1/haodongw/workspace/3dstream/3DGS_Compression_Adaptive_Voxelization/attributes_compressed/train_depth_15_thr_30_3DGS_adapt_lossless/train_dc.ply']
J = 18
T = len(ply_list)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

colorStep = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 64]
nSteps = len(colorStep)
bytes_log = torch.zeros((T, nSteps))
MSE = torch.zeros((T, nSteps))
Nvox = torch.zeros(T)
time_log = torch.zeros(T)

## ---------------------
## Main Processing Loop
## ---------------------
for frame_idx in range(T):
    frame = frame_idx + 1
    frame_start = time.time()
    
    V, Crgb = read_ply_file(ply_list[frame_idx])
    N = V.shape[0]
    Nvox[frame_idx] = N
    C = Crgb
    
    origin = torch.tensor([0, 0, 0], dtype=V.dtype)
    t0 = time.time()
    ListC, FlagsC, weightsC = RAHT_param(V, origin, 2**J, J)
    t1 = time.time()
    raht_param_time = t1 - t0
    save_lists(f"../results/frame{frame}_params_python.mat", ListC=ListC, FlagsC=FlagsC, weightsC=weightsC)
    
    saved_matlab_res = loadmat("../results/frame1_params_matlab.mat", simplify_cells=True)
    ListC = saved_matlab_res["ListC"]
    FlagsC = saved_matlab_res["FlagsC"]
    weightsC = saved_matlab_res["weightsC"]
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
    
    # ListC = [t.to(device) for t in ListC]
    # FlagsC = [t.to(device) for t in FlagsC]
    # weightsC = [t.to(device) for t in weightsC]
    C = C.to(device)
    
    timings = {}
    coeffs_by_variant = {}
    for name, fn in VARIANTS.items():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        out = fn(C, ListC, FlagsC, weightsC, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.time()
        Coeff = out[0] if (isinstance(out, (tuple, list)) and len(out) == 2) else out
        timings[name] = t3 - t2
        coeffs_by_variant[name] = Coeff
        save_mat(Coeff, f"../results/frame{frame}_coeff_python_{name}.mat")
        
        if DEBUG:
            print(f"Norm of C: {torch.norm(C)}")
            print(f"Norm of Coeff: {torch.norm(Coeff)}")
            print(f"Sanity check: {sanity_check_vector(Coeff[:, 0], C[:, 0])}")
            print(f"Sanity check: {sanity_check_vector(Coeff[:, 1], C[:, 1])}")
            print(f"Sanity check: {sanity_check_vector(Coeff[:, 2], C[:, 2])}")
            C_recon = inverse_RAHT(Coeff, ListC, FlagsC, weightsC, device)
            if torch.allclose(C, C_recon, rtol=1e-4, atol=1e-6):
                print("Reconstruction check: True")
            else:
                print("Reconstruction check: False")
                diff = C - C_recon
                print("Max difference:", diff.abs().max().item())
                print("Mean difference:", diff.abs().mean().item())

    # Print timing info
    print(f"Frame {frame}: RAHT_param={raht_param_time:.6f}s")
    for name, t in timings.items():
        print(f"  {name}: {t:.6f}s")
    
    # # Sort weights in descending order
    # _, IX_ref = torch.sort(w, descending=True)
    # Y = Coeff[:, 0]
    
    # # Loop through quantization steps
    # for i in range(nSteps):
    #     step = colorStep[i]
    #     Coeff_enc = torch.round(Coeff / step)
    #     Y_hat = Coeff_enc[:, 0] * step
        
    #     MSE[frame_idx, i] = (torch.linalg.norm(Y - Y_hat)**2) / (N * 255**2)
        
    #     nbytesY, _ = RLGR_encoder(Coeff_enc[IX_ref, 0])
    #     nbytesU, _ = RLGR_encoder(Coeff_enc[IX_ref, 1])
    #     nbytesV, _ = RLGR_encoder(Coeff_enc[IX_ref, 2])
        
    #     bytes_log[frame_idx, i] = nbytesY + nbytesU + nbytesV
    
    time_log[frame_idx] = time.time() - frame_start
    print(f"  Frame {frame}/{T} processed in {time_log[frame_idx]:.2f} seconds.")

# ## ---------------------
# ## Analysis, Plotting, and Saving
# ## ---------------------
# print("Analyzing results...")

# # Calculate PSNR from the mean MSE across all frames for each quantization step
# psnr = -10 * torch.log10(torch.mean(MSE, dim=0))

# # Calculate bits per voxel (bpv)
# total_bytes_per_step = torch.sum(bytes_log, dim=0)
# total_voxels = torch.sum(Nvox)
# bpv = 8 * total_bytes_per_step / total_voxels

# # --- Plotting ---
# plt.figure(figsize=(8, 6))
# plt.plot(bpv.numpy(), psnr.numpy(), 'b-x', label='O3D Load + RAHT Sim')
# plt.xlabel('Bits per Voxel (bpv)')
# plt.ylabel('Y-PSNR (dB)')
# plt.title('Rate-Distortion Curve')
# plt.grid(True)
# plt.legend()
# plt.show()
