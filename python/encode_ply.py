import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat

from data_util import read_ply_file
from utils import save_mat, save_lists
from RAHT import RAHT2, RAHT2_optimized, RAHT_batched
from iRAHT import inverse_RAHT
from RAHT_param import RAHT_param
import rlgr

VARIANTS = {
    "RAHT":             lambda C,L,F,W,d: RAHT2(C, L, F, W, d),
    "RAHT_optimized":   lambda C,L,F,W,d: RAHT2_optimized(C, L, F, W, d),
    "RAHT_batched":     lambda C,L,F,W,d: RAHT_batched(C, L, F, W, d),
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
    
    ListC = [t.to(device) for t in ListC]
    FlagsC = [t.to(device) for t in FlagsC]
    weightsC = [t.to(device) for t in weightsC]
    C = C.to(device)
    
    timings = {}
    coeffs_by_variant = {}
    for name, fn in VARIANTS.items():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        Coeff, w = fn(C, ListC, FlagsC, weightsC, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.time()
        timings[name] = t3 - t2
        coeffs_by_variant[name] = Coeff
        
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
        
        # Sort weights in descending order
        _, IX_ref = torch.sort(w, descending=True)
        Y = Coeff[:, 0]
        
        # temporary: filename for PyRLGR
        filename = 'test.bin'
        
        for i in range(nSteps):
            step = colorStep[i]
            Coeff_enc = torch.round(Coeff / step)
            Y_hat = Coeff_enc[:, 0] * step
            
            MSE[frame_idx, i] = (torch.linalg.norm(Y - Y_hat)**2) / (N * 255**2)
            
            # nbytesY, _ = RLGR_encoder(Coeff_enc[IX_ref, 0])
            # nbytesU, _ = RLGR_encoder(Coeff_enc[IX_ref, 1])
            # nbytesV, _ = RLGR_encoder(Coeff_enc[IX_ref, 2])
            # bytes_log[frame_idx, i] = nbytesY + nbytesU + nbytesV
            
            enc = rlgr.file(filename, 1)
            Y_list = [int(i) for i in Coeff_enc[IX_ref, 0].squeeze(1).tolist()]
            U_list = [int(i) for i in Coeff_enc[IX_ref, 1].squeeze(1).tolist()]
            V_list = [int(i) for i in Coeff_enc[IX_ref, 2].squeeze(1).tolist()]
            enc.rlgrWrite(Y_list, 0)
            enc.rlgrWrite(U_list, 0)
            enc.rlgrWrite(V_list, 0)
            enc.close()

    # Print timing info
    print(f"Frame {frame}: RAHT_param={raht_param_time:.6f}s")
    for name, t in timings.items():
        print(f"  {name}: {t:.6f}s")
    
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
