import torch
import numpy as np
import time
import os
import logging

from data_util import get_pointcloud, get_pointcloud_n_frames
from utils import rgb_to_yuv, save_mat, save_lists, is_frame_morton_ordered, block_indices, sanity_check_vector
from RAHT import RAHT2_optimized
from iRAHT import inverse_RAHT
from RAHT_param import RAHT_param
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
MSE = torch.zeros((T, nSteps))
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
logger.info("Frame,Quantization_Step,Rate_bpp,RAHT_prelude_time,RAHT_transform_time,order_RAGFT_time,Quant_time,Entropy_enc_time,Entropy_dec_time,Dequant_time,iRAHT_time")


## ---------------------
## Main Processing Loop
## ---------------------
print(f"\nStarting processing for {T} frames...")

for frame_idx in range(T):
    frame = frame_idx + 1

    V, Crgb, J = get_pointcloud(dataset, sequence, frame, data_root)
    N = V.shape[0]
    Nvox[frame_idx] = N
    Crgb = Crgb.to(torch.float64).to(device)
    C = rgb_to_yuv(Crgb)

    frame_start = time.time()
    origin = torch.tensor([0, 0, 0], dtype=V.dtype)
    raht_param_time = time.time()
    ListC, FlagsC, weightsC = RAHT_param(V, origin, 2**J, J)
    raht_param_time = time.time() - raht_param_time

    ListC = [t.to(device) for t in ListC]
    FlagsC = [t.to(device) for t in FlagsC]
    weightsC = [t.to(device) for t in weightsC]
    V = V.to(torch.float64).to(device)

    raht_transform_time = time.time()
    Coeff, w = RAHT2_optimized(C, ListC, FlagsC, weightsC)
    raht_transform_time = time.time() - raht_transform_time

    order_RAGFT_time = time.time()
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
    order_RAGFT_time = time.time() - order_RAGFT_time

    if DEBUG:
        save_lists(f"../results/frame{frame}_params_python.mat", ListC=ListC, FlagsC=FlagsC, weightsC=weightsC)
        save_mat(Coeff, f"../results/frame{frame}_coeff_python.mat")
        print(f"Norm of C: {torch.norm(C)}")
        print(f"Norm of Coeff: {torch.norm(Coeff)}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 0], C[:, 0])}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 1], C[:, 1])}")
        print(f"Sanity check: {sanity_check_vector(Coeff[:, 2], C[:, 2])}")
        C_recon = inverse_RAHT(Coeff, ListC, FlagsC, weightsC, device)
        print(f"Reconstruction check: {torch.allclose(C, C_recon, rtol=1e-5, atol=1e-8)}")

    # Sort weights in descending order
    # values, order_RAHT = torch.sort(w.squeeze(1), descending=True)
    # order_morton = torch.arange(0,V.shape[0])
    # Y = Coeff[:, 0]

    # temporary filename for PyRLGR
    filename = "test.bin"
    rates = []
    # Loop through quantization steps
    for i in range(nSteps):
        # print(i)
        quant_time = time.time()
        step = colorStep[i]
        Coeff_enc = torch.floor(Coeff / step + 0.5)
        quant_time = time.time() - quant_time
        # Y_hat = Coeff_enc[:, 0] * step
        # MSE[frame_idx, i] = (torch.linalg.norm(Y - Y_hat)**2) / (N * 255**2)

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
        rate_bpp = size_bytes * 8 / N
        rates.append(rate_bpp)

        ## lzma
        # Step 1️⃣: 序列化 tensor 到内存字节流
        # buf = io.BytesIO()
        # torch.save(Coeff_enc, buf)
        # buf.seek(0)
        # data = buf.getvalue()
        #
        # # Step 2️⃣: 创建新的内存流用于存放压缩数据
        # compressed_buf = io.BytesIO()
        #
        # # ✅ 用 writestr，arcname 必须是简单文件名（不能带路径！）
        # with py7zr.SevenZipFile(compressed_buf, mode='w') as archive:
        #     archive.writestr(data,"tensor.pt")
        #
        # # 压缩字节流
        # compressed_bytes = compressed_buf.getvalue()
        # size_bytes = len(compressed_bytes)
        # rates.append(size_bytes * 8 / N)

        # # cabac
        # # === 1. 准备数据 ===
        # Coeff_enc = Coeff_enc[order_RAGFT,:]
        # # Coeff_enc = Coeff_enc[3:-1,:]
        # Coeff_enc = signed_to_unsigned(Coeff_enc)
        # Coeff_enc = Coeff_enc.flatten().abs().int().tolist()
        #
        # enc = cabac.cabacSimpleSequenceEncoder()
        # enc.initCtx(1, 0.5, 8)
        # enc.start()
        #
        # k = 0
        # num_max_val = int(np.max(Coeff_enc))
        # num_bins = math.ceil(math.log2(num_max_val + 1))
        # ctx_ids = np.zeros(num_max_val + 2, dtype=np.uint32)
        # for m in Coeff_enc:
        #     enc.encodeBinsEGk(m, k, ctx_ids)  # 编码幅度（非负整数）
        #
        # enc.encodeBinTrm(1)
        # enc.finish()
        # enc.writeByteAlignment()
        # compressed_bytes = enc.getBitstream()
        # size_bytes = len(compressed_bytes)
        # rates.append(size_bytes * 8 / N)
        #
        # # === 4. 解码验证 ===
        # dec = cabac.cabacSimpleSequenceDecoder(compressed_bytes)
        # dec.initCtx(1, 0.5, 8)
        # dec.start()
        #
        # Coeff_dec = [dec.decodeBinsEGk(k, ctx_ids) for _ in Coeff_enc]
        # Coeff_dec = torch.tensor(Coeff_dec).reshape(-1, 3)
        # Coeff_dec = unsigned_to_signed(Coeff_dec)

        entropy_enc_time = (entropy_enc_Y_time + entropy_enc_U_time + entropy_enc_V_time) / 1e9
        entropy_dec_time = (entropy_dec_Y_time + entropy_dec_U_time + entropy_dec_V_time) / 1e9

        Coeff_dec = tensor = torch.stack([Y_list_dec,
                                          U_list_dec,
                                          V_list_dec], dim=0).T

        dequant_time = time.time()
        Coeff_dec = Coeff_dec * step
        dequant_time = time.time() - dequant_time

        iRAHT_time = time.time()
        order_RAGFT = torch.argsort(order_RAGFT)
        Coeff_dec = Coeff_dec[order_RAGFT,:]
        C_rec = inverse_RAHT(Coeff_dec, ListC, FlagsC, weightsC)
        iRAHT_time = time.time() - iRAHT_time
        # print(f"Reconstruction check: {torch.allclose(C, C_rec, rtol=1e-5, atol=1e-8)}")
        
        # Log timing data as CSV row
        logger.info(f"{frame},{colorStep[i]},{rate_bpp:.6f},{raht_param_time:.6f},{raht_transform_time:.6f},"
                   f"{order_RAGFT_time:.6f},{quant_time:.6f},{entropy_enc_time:.6f},"
                   f"{entropy_dec_time:.6f},{dequant_time:.6f},{iRAHT_time:.6f}")

    os.remove(filename)
    # Print timing information
    print(f"Frame {frame}: RAHT_param={raht_param_time:.6f}s, "
          f"RAHT_optimized={raht_transform_time:.6f}s, "
          f"order_RAGFT_time={order_RAGFT_time:.6f}s, "
          f"quant_time={quant_time:.6f}s, "
          f"entropy_enc_time={entropy_enc_time:.6f}s, "
          f"entropy_dec_time={entropy_dec_time:.6f}s, "
          f"dequant_time={dequant_time:.6f}s, "
          f"iRAHT_time={iRAHT_time:.6f}s, ")
    print("\t".join(map(str, rates)))
    print("\t".join(map(str, [raht_param_time,raht_transform_time,order_RAGFT_time,quant_time,entropy_enc_time,entropy_dec_time,
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
