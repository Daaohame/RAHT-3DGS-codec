import torch
import os
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat

from data_util import get_pointcloud, get_pointcloud_n_frames
from utils import rgb_to_yuv, save_mat, save_lists, block_indices, is_frame_morton_ordered, sanity_check_vector
from RAHT import RAHT2, RAHT2_optimized
from iRAHT import inverse_RAHT
from RAHT_param import RAHT_param, RAHT_param_optimized
import rlgr

DEBUG = False

## ---------------------
## Configuration
## ---------------------
data_root = '/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/matlab'
dataset = '8iVFBv2'
sequence = 'redandblack'
T = get_pointcloud_n_frames(dataset, sequence)
T = 1

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

colorStep = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 64]
nSteps = len(colorStep)
MSE = torch.zeros((T, nSteps))
Nvox = torch.zeros(T)

## ---------------------
## Main Processing Loop
## ---------------------
print(f"\nStarting processing for {T} frames...")

for frame_idx in range(T):
    frame = frame_idx + 1   # 1-based indexing

    V, Crgb, J = get_pointcloud(dataset, sequence, frame, data_root)
    N = V.shape[0]
    Nvox[frame_idx] = N
    V = V.to(torch.float64).to(device)
    Crgb = Crgb.to(torch.float64).to(device)
    C = rgb_to_yuv(Crgb)
    
    origin = torch.tensor([0, 0, 0], dtype=V.dtype)
    t0 = time.time()
    ListC, FlagsC, weightsC = RAHT_param(V, origin, 2**J, J)
    t1 = time.time()
    raht_param_time = t1 - t0

    ListC = [t.to(device) for t in ListC]
    FlagsC = [t.to(device) for t in FlagsC]
    weightsC = [t.to(device) for t in weightsC]

    t2 = time.time()
    Coeff, w = RAHT2_optimized(C, ListC, FlagsC, weightsC)
    t3 = time.time()
    raht_transform_time = t3 - t2

    if DEBUG:
        save_lists(f"../results/frame{frame}_params_python.mat", ListC=ListC, FlagsC=FlagsC, weightsC=weightsC)
        save_mat(Coeff, f"../results/frame{frame}_coeff_python.mat")
        print(f"Norm of C: {torch.norm(C)}")
        print(f"Norm of Coeff: {torch.norm(Coeff)}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 0], C[:, 0])}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 1], C[:, 1])}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 2], C[:, 2])}")
        C_recon = inverse_RAHT(Coeff, ListC, FlagsC, weightsC)
        print(f"Reconstruction check: {torch.allclose(C, C_recon, rtol=1e-5, atol=1e-8)}")

    # Sort weights in descending order
    t50 = time.time()
    values, order_RAHT = torch.sort(w.squeeze(1), descending=True)
    t5 = time.time()
    raht_reorder_RAHT_time = t5 - t50

    # Sort weights in Morton order
    order_morton = torch.arange(0,V.shape[0])
    t6 = time.time()
    raht_reorder_morton_time = t6 - t5
    
    # Sort weights in RA-GFT order
    error, V, index = is_frame_morton_ordered(V, J)
    ac_list = []
    dc_list = []
    indices = torch.arange(0, N)
    for i in range(J):
        indices,indices_remain = block_indices(V[indices,:], 2**(i+1))
        if i == 0:
            ac_list.append(indices_remain)
            dc_list.append(indices)
        else:
            indices = dc_list[i-1][indices]
            indices_remain = dc_list[i-1][indices_remain]
            ac_list.append(indices_remain)
            dc_list.append(indices)
    ac_list.append(indices)
    ac_list = ac_list[::-1]
    order_RAGFT = torch.cat(ac_list)
    t4 = time.time()
    raht_reorder_RAGFT_time = t4 - t3


    # temporary: filename for PyRLGR
    filename = 'test.bin'
    rates = []
    Y = Coeff[:, 0]
    # Loop through quantization steps
    for i in range(nSteps):
        quant_time = time.time()
        step = colorStep[i]
        Coeff_enc = torch.floor(Coeff / step + 0.5)
        quant_time = time.time() - quant_time
        # Y_hat = Coeff_enc[:, 0] * step
        # MSE[frame_idx, i] = (torch.linalg.norm(Y - Y_hat)**2) / (N * 255**2)

        entropy_enc_Y_time = time.time()
        Y_list = [int(i) for i in Coeff_enc[order_RAGFT, 0].tolist()]
        enc = rlgr.file(filename, 1)
        enc.rlgrWrite(Y_list, int(1))
        enc.close()
        entropy_enc_Y_time = time.time() - entropy_enc_Y_time
        entropy_dec_Y_time = time.time()
        dec = rlgr.file(filename, 0)
        Y_list_dec = dec.rlgrRead(N, 1)
        bytesY = os.path.getsize(filename)
        dec.close()
        Y_list_dec = torch.tensor(Y_list_dec).to(device=device, dtype=torch.float64)
        entropy_dec_Y_time = time.time() - entropy_dec_Y_time

        entropy_enc_U_time = time.time()
        U_list = [int(i) for i in Coeff_enc[order_RAGFT, 1].tolist()]
        enc = rlgr.file(filename, 1)
        enc.rlgrWrite(U_list, int(1))
        enc.close()
        entropy_enc_U_time = time.time() - entropy_enc_U_time
        entropy_dec_U_time = time.time()
        dec = rlgr.file(filename, 0)
        U_list_dec = dec.rlgrRead(N, 1)
        bytesU = os.path.getsize(filename)
        dec.close()
        U_list_dec = torch.tensor(U_list_dec).to(device=device, dtype=torch.float64)
        entropy_dec_U_time = time.time() - entropy_dec_U_time

        entropy_enc_V_time = time.time()
        V_list = [int(i) for i in Coeff_enc[order_RAGFT, 2].tolist()]
        enc = rlgr.file(filename, 1)
        enc.rlgrWrite(V_list, int(1))
        enc.close()
        entropy_enc_V_time = time.time() - entropy_enc_V_time
        entropy_dec_V_time = time.time()
        dec = rlgr.file(filename, 0)
        V_list_dec = dec.rlgrRead(N, 1)
        bytesV = os.path.getsize(filename)
        dec.close()
        V_list_dec = torch.tensor(V_list_dec).to(device=device, dtype=torch.float64)
        entropy_dec_V_time = time.time() - entropy_dec_V_time

        size_bytes = bytesY + bytesU + bytesV
        rates.append(size_bytes * 8 / N)
        
        # stat
        entropy_enc_time = entropy_enc_Y_time + entropy_enc_U_time + entropy_enc_V_time
        entropy_dec_time = entropy_dec_Y_time + entropy_dec_U_time + entropy_dec_V_time
        
        
        Coeff_dec = tensor = torch.stack([Y_list_dec,
                                          U_list_dec,
                                          V_list_dec], dim=0).T

        # dequantization
        dequant_time = time.time()
        Coeff_dec = Coeff_dec * step
        dequant_time = time.time() - dequant_time
        
        iRAHT_time = time.time()
        order_RAGFT = torch.argsort(order_RAGFT)
        Coeff_dec = Coeff_dec[order_RAGFT,:]
        C_rec = inverse_RAHT(Coeff_dec, ListC, FlagsC, weightsC)
        iRAHT_time = time.time() - iRAHT_time
        print(f"Reconstruction check: {torch.allclose(C, C_rec, rtol=1e-5, atol=1e-8)}")
    
    print("rates (bpv): ", end="")
    print("\t".join(map(str, rates)))
    # Print timing information
    print(f"Frame {frame}: RAHT_param={raht_param_time:.6f}s, "
          f"RAHT_optimized={raht_transform_time:.6f}s, "
          f"order_RAGFT_time={raht_reorder_RAGFT_time:.6f}s, "
          f"quant_time={quant_time:.6f}s, "
          f"entropy_enc_time={entropy_enc_time:.6f}s, "
          f"entropy_dec_time={entropy_dec_time:.6f}s, "
          f"dequant_time={dequant_time:.6f}s, "
          f"iRAHT_time={iRAHT_time:.6f}s, ")
    print("\t".join(map(str, rates)))
    print("\t".join(map(str, [raht_param_time,raht_transform_time,raht_transform_time,quant_time,entropy_enc_time,entropy_dec_time,
                              dequant_time,iRAHT_time])))

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
