import torch
import numpy as np
import time
import os
import glob
from scipy.io import savemat
import matplotlib.pyplot as plt
from scipy.io import loadmat

from data_util import get_pointcloud, get_pointcloud_n_frames, read_ply_file
from utils import rgb_to_yuv_torch2, save_mat, save_lists
from RAHT import RAHT2, RAHT_optimized, RAHT2_optimized,is_frame_morton_ordered,block_indices
from iRAHT import inverse_RAHT
from RAHT_param import RAHT_param2
from crosscheck import compare_lists,load_raht_param_from_mat,compare_raht_param,load_raht_out_mat,compare_RAHT_outputs
import rlgr

DEBUG = False


## ---------------------
## Configuration
## ---------------------
ply_list = ['C:\\Users\\hhrho\\Downloads\\train_dc.ply']
J = 18
T = len(ply_list)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

colorStep = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 64]
# colorStep = [1]
nSteps = len(colorStep)
bytes_log = torch.zeros((T, nSteps))
MSE = torch.zeros((T, nSteps))
Nvox = torch.zeros(T)
time_log = torch.zeros(T)

## ---------------------
## Main Processing Loop
## ---------------------
print(f"\nStarting processing for {T} frames...")

for frame_idx in range(T):
    frame = frame_idx + 1
    frame_start = time.time()

    V, Crgb = read_ply_file(ply_list[frame_idx])
    N = V.shape[0]
    Nvox[frame_idx] = N
    Crgb = Crgb.to(torch.float64).to(device)
    C = rgb_to_yuv_torch2(Crgb)

    origin = torch.tensor([0, 0, 0], dtype=V.dtype)
    t0 = time.time()
    ListC, FlagsC, weightsC = RAHT_param2(V, origin, 2**J, J)
    t1 = time.time()
    raht_param_time = t1 - t0

    # L_ref, F_ref, W_ref = load_raht_param_from_mat("F:\\Desktop\\Motion_Vector_Database\\ref_raht_param.mat",
    #                                                device='cpu', one_based=True)
    # L_py, F_py, W_py = ListC, FlagsC, weightsC

    # 若 Python 是 0-based、MATLAB 是 1-based：先统一
    # if L_py and L_py[0].numel() > 0 and L_py[0].min().item() == 0:
    #     L_py = [t + 1 for t in L_py]  # 统一为 1-based 再比较

    # okL = compare_lists(L_ref, L_py, "ListC")
    # okF = compare_lists(F_ref, F_py, "FlagsC")  # 布尔整等比较
    # okW = compare_lists(W_ref, W_py, "weightsC")  # 整型整等比较
    # all_ok = okL and okF and okW

    ListC = [t.to(device) for t in ListC]
    FlagsC = [t.to(device) for t in FlagsC]
    weightsC = [t.to(device) for t in weightsC]

    t2 = time.time()
    Coeff, w = RAHT2_optimized(C, ListC, FlagsC, weightsC)
    # Coeff_m, w_m = load_raht_out_mat("F:\\Desktop\\Motion_Vector_Database\\ref_raht_out.mat", device='cpu')
    # compare_RAHT_outputs(Coeff_m, w_m, Coeff, w, rtol=1e-12, atol=1e-12)

    t3 = time.time()
    raht_transform_time = t3 - t2

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
    # Sort weights in descending order

    values, order_RAHT = torch.sort(w.squeeze(1), descending=True)
    t5 = time.time()
    raht_reorder_RAHT_time = t5 - t4

    order_morton = torch.arange(0,V.shape[0])
    t6 = time.time()
    raht_reorder_morton_time = t6 - t5

    print(f"Frame {frame}: RAHT_param={raht_param_time:.6f}s, "
          f"RAHT_optimized={raht_transform_time:.6f}s, "
          f"RAHT_reorder_RAGFT={raht_reorder_RAGFT_time:.6f}s, "
          f"RAHT_reorder_RAHT={raht_reorder_RAHT_time:.6f}s, "
          f"RAHT_reorder_morton={raht_reorder_morton_time:.6f}s")

    Y = Coeff[:, 0]

    # temporary: filename for PyRLGR
    filename = 'test.bin'
    rates = []
    # Loop through quantization steps
    for i in range(nSteps):
        step = colorStep[i]
        Coeff_enc = torch.floor(Coeff / step + 0.5)
        Y_hat = Coeff_enc[:, 0] * step

        MSE[frame_idx, i] = (torch.linalg.norm(Y - Y_hat)**2) / (N * 255**2)

        # nbytesY, _ = RLGR_encoder(Coeff_enc[IX_ref, 0])
        # nbytesU, _ = RLGR_encoder(Coeff_enc[IX_ref, 1])
        # nbytesV, _ = RLGR_encoder(Coeff_enc[IX_ref, 2])
        # bytes_log[frame_idx, i] = nbytesY + nbytesU + nbytesV

        enc = rlgr.file(filename, 1)
        Y_list = [int(i) for i in Coeff_enc[order_RAHT, 0].tolist()] #order_RAHT order_RAGFT order_morton
        U_list = [int(i) for i in Coeff_enc[order_RAHT, 1].tolist()]
        V_list = [int(i) for i in Coeff_enc[order_RAHT, 2].tolist()]
        Nbits = torch.ceil(torch.log2(torch.max(torch.abs(Coeff_enc)) + 1))
        enc.rlgrWrite(Y_list, int(1))
        enc.rlgrWrite(U_list, int(1))
        enc.rlgrWrite(V_list, int(1))
        enc.close()
        size_bytes = os.path.getsize(filename)
        rates.append(size_bytes*8/N)

        dec = rlgr.file(filename, 0)
        Y_list_dec = dec.rlgrRead(N, 1)
        U_list_dec = dec.rlgrRead(N, 1)
        V_list_dec = dec.rlgrRead(N, 1)
        dec.close()

        # rates.append(size_bytes)

    time_log[frame_idx] = time.time() - frame_start
    print(f"  Frame {frame}/{T} processed in {time_log[frame_idx]:.2f} seconds.")
    print("\t".join(map(str, rates)))

