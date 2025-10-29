import torch
import numpy as np
import time
import os
import logging

from data_util import get_pointcloud, get_pointcloud_n_frames
from utils import rgb_to_yuv, save_mat, save_lists, sanity_check_vector
from RAHT import RAHT2_optimized
from iRAHT import inverse_RAHT_optimized
from RAHT_param import RAHT_param, RAHT_param_reorder, RAHT_param_reorder_fast
import rlgr

DEBUG = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

## ---------------------
## Configuration
## ---------------------
data_root = '/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/data'
dataset = '8iVFBv2'
sequence = 'redandblack'
T = get_pointcloud_n_frames(dataset, sequence)

colorStep = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 64]
nSteps = len(colorStep)
bytes_log = torch.zeros((T, nSteps))
rates = torch.zeros((T, nSteps), dtype=torch.float64)
raht_param_time = torch.zeros((T, nSteps), dtype=torch.float64)
raht_transform_time = torch.zeros((T, nSteps), dtype=torch.float64)
order_RAGFT_time = torch.zeros((T, nSteps), dtype=torch.float64)
quant_time = torch.zeros((T, nSteps), dtype=torch.float64)
entropy_enc_time = torch.zeros((T, nSteps), dtype=torch.float64)
entropy_dec_time = torch.zeros((T, nSteps), dtype=torch.float64)
dequant_time = torch.zeros((T, nSteps), dtype=torch.float64)
iRAHT_time = torch.zeros((T, nSteps), dtype=torch.float64)
MSE = torch.zeros((T, nSteps), dtype=torch.float64)
psnr = torch.zeros((T, nSteps), dtype=torch.float64)
Nvox = torch.zeros(T)
time_log = torch.zeros(T)


## ---------------------
## Logging setup
## ---------------------
log_filename = f'../results/runtime_{dataset}_{sequence}.csv'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w')
    ]
)
logger = logging.getLogger(__name__)
logger.info("Frame,Quantization_Step,Rate_bpp,RAHT_prelude_time,RAHT_transform_time,order_RAGFT_time,Quant_time,Entropy_enc_time,Entropy_dec_time,Dequant_time,iRAHT_time,psnr")


## ---------------------
## Precision Setup
## ---------------------
use_fp64 = True  # set True only if RAHT requires double precision
DTYPE = torch.float64 if use_fp64 else torch.float32
def to_dev(x):
    return x.to(dtype=DTYPE, device=device, non_blocking=True)


## One dummy iteration to warm up GPU
print("Warming up GPU with a dummy iteration...")
V_dummy, Crgb_dummy, J_dummy = get_pointcloud(dataset, sequence, 1, data_root)
J_dummy = int(J_dummy) if not isinstance(J_dummy, int) else J_dummy
Vw = V_dummy.to(dtype=DTYPE)  # CPU for param if RAHT_param is CPU-bound
origin_dummy = torch.zeros(3, dtype=Vw.dtype, device=Vw.device)

ListC_dummy, FlagsC_dummy, weightsC_dummy = RAHT_param(Vw, origin_dummy, 2**J_dummy, J_dummy)

ListC_dummy = [t.to(device=device, non_blocking=True) for t in ListC_dummy]
FlagsC_dummy = [t.to(device=device, non_blocking=True) for t in FlagsC_dummy]
weightsC_dummy = [t.to(device=device, non_blocking=True) for t in weightsC_dummy]

Crgb_dummy = Crgb_dummy.to(dtype=DTYPE)
C_dummy = rgb_to_yuv(Crgb_dummy).contiguous()
C_dummy = to_dev(C_dummy)

# Run transform (warm-up kernels/caches)
Coeff_dummy, w_dummy = RAHT2_optimized(C_dummy, ListC_dummy, FlagsC_dummy, weightsC_dummy)

# Cleanup
del V_dummy, Crgb_dummy, J_dummy, Vw, origin_dummy
del ListC_dummy, FlagsC_dummy, weightsC_dummy, C_dummy, Coeff_dummy, w_dummy


## ---------------------
## Main Processing Loop
## ---------------------
print(f"\nStarting processing for {T} frames...")
for frame_idx in range(T):
    frame = frame_idx + 1

    V, Crgb, J = get_pointcloud(dataset, sequence, frame, data_root)
    N = V.shape[0]
    Nvox[frame_idx] = N
    Cyuv = rgb_to_yuv(Crgb.to(dtype=DTYPE)).contiguous()
    C = to_dev(Cyuv)
    V = V.to(dtype=DTYPE).to(device)

    frame_start = time.time()
    origin = torch.tensor([0, 0, 0], dtype=V.dtype, device=device)
    start_time = time.time()
    ListC, FlagsC, weightsC, order_RAGFT = RAHT_param_reorder_fast(V, origin, 2 ** J, J)
    raht_param_time[frame_idx, :] = time.time() - start_time

    ListC = [t.to(device=device, non_blocking=True) for t in ListC]
    FlagsC = [t.to(device=device, non_blocking=True) for t in FlagsC]
    weightsC = [t.to(device=device, non_blocking=True) for t in weightsC]
    
    start_time = time.time()
    Coeff, w = RAHT2_optimized(C, ListC, FlagsC, weightsC)
    raht_transform_time[frame_idx, :] = time.time() - start_time

    if DEBUG:
        save_lists(f"../results/frame{frame}_params_python.mat", ListC=ListC, FlagsC=FlagsC, weightsC=weightsC)
        save_mat(Coeff, f"../results/frame{frame}_coeff_python.mat")
        print(f"Norm of C: {torch.norm(C)}")
        print(f"Norm of Coeff: {torch.norm(Coeff)}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 0], C[:, 0])}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 1], C[:, 1])}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 2], C[:, 2])}")
        C_recon = inverse_RAHT_optimized(Coeff, ListC, FlagsC, weightsC)
        print(f"Reconstruction check: {torch.allclose(C, C_recon, rtol=1e-5, atol=1e-8)}")

    # Sort weights in descending order
    # values, order_RAHT = torch.sort(w.squeeze(1), descending=True)
    # order_morton = torch.arange(0,V.shape[0])
    # Y = Coeff[:, 0]

    # temporary filename for PyRLGR
    filename = "test.bin"
    # Loop through quantization steps
    for i in range(nSteps):
        step = colorStep[i]
        start_time = time.time()
        Coeff_enc = torch.floor(Coeff / step + 0.5)
        quant_time[frame_idx, i] = time.time() - start_time
        Y_hat = Coeff_enc[:, 0] * step
        MSE[frame_idx, i] = (torch.linalg.norm(Coeff[:,0] - Y_hat)**2) / (N * 255**2)
        psnr[frame_idx, i] = -10 * torch.log10(MSE[frame_idx, i])

        # RLGR
        Nbits = torch.ceil(torch.log2(torch.max(torch.abs(Coeff_enc)) + 1))

        Y_list = [int(i) for i in Coeff_enc[order_RAGFT, 0].tolist()]
        enc = rlgr.file(filename, 1)
        entropy_enc_Y_time = enc.rlgrWrite(Y_list, int(1))
        enc.close()
        dec = rlgr.file(filename, 0)
        entropy_dec_Y_time, Y_list_dec = dec.rlgrRead(N, 1)
        bytesY = os.path.getsize(filename)
        dec.close()
        Y_list_dec = torch.tensor(Y_list_dec).to(device=device, dtype=torch.float64)
        
        U_list = [int(i) for i in Coeff_enc[order_RAGFT, 1].tolist()]
        enc = rlgr.file(filename, 1)
        entropy_enc_U_time = enc.rlgrWrite(U_list, int(1))
        enc.close()
        dec = rlgr.file(filename, 0)
        entropy_dec_U_time, U_list_dec = dec.rlgrRead(N, 1)
        bytesU = os.path.getsize(filename)
        dec.close()
        U_list_dec = torch.tensor(U_list_dec).to(device=device, dtype=torch.float64)
        
        V_list = [int(i) for i in Coeff_enc[order_RAGFT, 2].tolist()]
        enc = rlgr.file(filename, 1)
        entropy_enc_V_time = enc.rlgrWrite(V_list, int(1))
        enc.close()
        dec = rlgr.file(filename, 0)
        entropy_dec_V_time, V_list_dec = dec.rlgrRead(N, 1)
        bytesV = os.path.getsize(filename)
        dec.close()
        V_list_dec = torch.tensor(V_list_dec).to(device=device, dtype=torch.float64)

        size_bytes = bytesY + bytesU + bytesV
        # rate_bpp = size_bytes * 8 / N
        rates[frame_idx, i] = size_bytes

        entropy_enc_time[frame_idx, i] = (entropy_enc_Y_time + entropy_enc_U_time + entropy_enc_V_time) / 1e9
        entropy_dec_time[frame_idx, i] = (entropy_dec_Y_time + entropy_dec_U_time + entropy_dec_V_time) / 1e9

        Coeff_dec = tensor = torch.stack([Y_list_dec,
                                          U_list_dec,
                                          V_list_dec], dim=0).T

        start_time = time.time()
        Coeff_dec = Coeff_dec * step
        dequant_time[frame_idx, i] = time.time() - start_time
        
        start_time = time.time()
        order_RAGFT_dec = torch.argsort(order_RAGFT)
        Coeff_dec = Coeff_dec[order_RAGFT_dec,:]
        C_rec = inverse_RAHT_optimized(Coeff_dec, ListC, FlagsC, weightsC)
        iRAHT_time[frame_idx, i] = time.time() - start_time
        # print(f"Reconstruction check: {torch.allclose(C, C_rec, rtol=1e-5, atol=1e-8)}")
        
        logger.info(
            f"{frame},{colorStep[i]},{rates[frame_idx, i]*8/Nvox[frame_idx]:.6f},{raht_param_time[frame_idx, i]:.6f},{raht_transform_time[frame_idx, i]:.6f},"
            f"{order_RAGFT_time[frame_idx, i]:.6f},{quant_time[frame_idx, i]:.6f},{entropy_enc_time[frame_idx, i]:.6f},"
            f"{entropy_dec_time[frame_idx, i]:.6f},{dequant_time[frame_idx, i]:.6f},{iRAHT_time[frame_idx, i]:.6f},{psnr[frame_idx, i]:.6f}")

    print(f"Frame {frame}")
    os.remove(filename)

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

# # --- Saving ---
# sequence_name = "test_sequence"
# folder = f'RA-GFT/results/{sequence_name}/'
# os.makedirs(folder, exist_ok=True)
# filename = os.path.join(folder, f'{sequence_name}_RAHT.mat')
# print(f"Saving results to {filename}...")
# data_to_save = {
#     'MSE': MSE.numpy(),
#     'bytes': bytes_log.numpy(),
#     'Nvox': Nvox.numpy(),
#     'colorStep': np.array(colorStep)
# }
# savemat(filename, data_to_save)
