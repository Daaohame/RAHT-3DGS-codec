import torch
import numpy as np
import time
import logging

from data_util import get_pointcloud, get_pointcloud_n_frames
from utils import rgb_to_yuv, save_mat, save_lists, sanity_check_vector
from RAHT import RAHT2_optimized
from iRAHT import inverse_RAHT_optimized
from RAHT_param import RAHT_param, RAHT_param_reorder_fast
import rlgr


## ---------------------
## Configuration
## ---------------------
torch.backends.cudnn.benchmark=False # for benchmarking
DEBUG = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
raht_fn = {
    "RAHT": RAHT2_optimized,
    "iRAHT": inverse_RAHT_optimized,
    "RAHT_param": RAHT_param_reorder_fast
}

data_root = '/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/data'
dataset = '8iVFBv2'
sequence = 'redandblack'
T = get_pointcloud_n_frames(dataset, sequence)
colorStep = [1, 4, 8, 12, 16, 20, 24, 32, 64]


nSteps = len(colorStep)
rates = torch.zeros((T, nSteps), dtype=torch.float64)
raht_param_time = torch.zeros((T, nSteps), dtype=torch.float64)
raht_transform_time = torch.zeros((T, nSteps), dtype=torch.float64)
quant_time = torch.zeros((T, nSteps), dtype=torch.float64)
entropy_enc_time = torch.zeros((T, nSteps), dtype=torch.float64)
entropy_dec_time = torch.zeros((T, nSteps), dtype=torch.float64)
dequant_time = torch.zeros((T, nSteps), dtype=torch.float64)
iRAHT_time = torch.zeros((T, nSteps), dtype=torch.float64)
MSE = torch.zeros((T, nSteps), dtype=torch.float64)
psnr = torch.zeros((T, nSteps), dtype=torch.float64)
Nvox = torch.zeros(T)


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
logger.info("Frame,Quantization_Step,Rate_bpp,RAHT_prelude_time,RAHT_transform_time,Quant_time,Entropy_enc_time,Entropy_dec_time,Dequant_time,iRAHT_time,psnr")


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
Vw = V_dummy.to(dtype=DTYPE).to(device)
origin_dummy = torch.zeros(3, dtype=Vw.dtype, device=Vw.device)

ListC_dummy, FlagsC_dummy, weightsC_dummy, order_RAGFT_dummy = raht_fn["RAHT_param"](Vw, origin_dummy, 2**J_dummy, J_dummy)

ListC_dummy = [t.to(device=device, non_blocking=True) for t in ListC_dummy]
FlagsC_dummy = [t.to(device=device, non_blocking=True) for t in FlagsC_dummy]
weightsC_dummy = [t.to(device=device, non_blocking=True) for t in weightsC_dummy]

Crgb_dummy = Crgb_dummy.to(dtype=DTYPE)
C_dummy = rgb_to_yuv(Crgb_dummy).contiguous()
C_dummy = to_dev(C_dummy)

# Run transform (warm-up kernels/caches)
Coeff_dummy, w_dummy = raht_fn["RAHT"](C_dummy, ListC_dummy, FlagsC_dummy, weightsC_dummy)
order_RAGFT_dec_dummy = torch.argsort(order_RAGFT_dummy)
Coeff_enc_dummy = torch.floor(Coeff_dummy / 10.5)
Coeff_dec_dummy = Coeff_enc_dummy[order_RAGFT_dec_dummy,:]
C_rec = raht_fn["iRAHT"](Coeff_dec_dummy, ListC_dummy, FlagsC_dummy, weightsC_dummy)

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
    ListC, FlagsC, weightsC, order_RAGFT = raht_fn["RAHT_param"](V, origin, 2 ** J, J)
    raht_param_time[frame_idx, :] = time.time() - start_time

    ListC = [t.to(device=device, non_blocking=True) for t in ListC]
    FlagsC = [t.to(device=device, non_blocking=True) for t in FlagsC]
    weightsC = [t.to(device=device, non_blocking=True) for t in weightsC]
    
    start_time = time.time()
    Coeff, w = raht_fn["RAHT"](C, ListC, FlagsC, weightsC)
    raht_transform_time[frame_idx, :] = time.time() - start_time

    if DEBUG:
        save_lists(f"../results/frame{frame}_params_python.mat", ListC=ListC, FlagsC=FlagsC, weightsC=weightsC)
        save_mat(Coeff, f"../results/frame{frame}_coeff_python.mat")
        print(f"Norm of C: {torch.norm(C)}")
        print(f"Norm of Coeff: {torch.norm(Coeff)}")
        print(f"Sanity check Y: {sanity_check_vector(Coeff[:, 0], C[:, 0])}")
        print(f"Sanity check U: {sanity_check_vector(Coeff[:, 1], C[:, 1])}")
        print(f"Sanity check V: {sanity_check_vector(Coeff[:, 2], C[:, 2])}")
        # Verify lossless RAHT
        C_recon = raht_fn["iRAHT"](Coeff, ListC, FlagsC, weightsC)
        raht_error = torch.abs(C - C_recon).max()
        print(f"Lossless RAHT max error: {raht_error:.2e}")
        print(f"Lossless RAHT check passes: {torch.allclose(C, C_recon, rtol=1e-5, atol=1e-8)}")

    # Sort weights in descending order
    # values, order_RAHT = torch.sort(w.squeeze(1), descending=True)
    # order_morton = torch.arange(0,V.shape[0])

    # Loop through quantization steps
    for i in range(nSteps):
        step = colorStep[i]
        start_time = time.time()
        Coeff_enc = torch.floor(Coeff / step + 0.5)
        quant_time[frame_idx, i] = time.time() - start_time
        Y_hat = Coeff_enc[:, 0] * step
        MSE[frame_idx, i] = (torch.linalg.norm(Coeff[:,0] - Y_hat)**2) / (N * 255**2)
        psnr[frame_idx, i] = -10 * torch.log10(MSE[frame_idx, i])
        # Nbits = torch.ceil(torch.log2(torch.max(torch.abs(Coeff_enc)) + 1))
        
        # get reoredered coefficients
        coeff_reordered = Coeff_enc.index_select(0, order_RAGFT)
        coeff_cpu_i32 = coeff_reordered.to('cpu', dtype=torch.int32, non_blocking=True)
        np_coeff = coeff_cpu_i32.numpy()                        # zero-copy view on CPU
        Y_list = np_coeff[:, 0].tolist()
        U_list = np_coeff[:, 1].tolist()
        V_list = np_coeff[:, 2].tolist()

        # RLGR settings
        channels = {"Y": Y_list, "U": U_list, "V": V_list}
        flag_signed = 1  # 1 => signed integers
        compressed = {}     # name -> {"buf": list[uint8], "time_ns": int}
        decoded = {}        # name -> {"out": list[int], "time_ns": int}

        # encode
        for name, data in channels.items():
            m_write = rlgr.membuf()                 # write buffer
            encode_time_ns = m_write.rlgrWrite(data, flag_signed)
            m_write.close()
            buf = m_write.get_buffer()              # list[uint8] (bytes-like)
            compressed[name] = {"buf": buf, "time_ns": encode_time_ns}
        
        # decode
        for name, original in channels.items():
            original_len = len(original)
            m_read = rlgr.membuf(compressed[name]["buf"])
            decode_time_ns, out = m_read.rlgrRead(original_len, flag_signed)
            m_read.close()
            assert len(out) == original_len, f"Length mismatch for {name}: {len(out)} != {original_len}"
            decoded[name] = {"out": out, "time_ns": decode_time_ns}
            # Verify RLGR roundtrip correctness
            assert decoded[name]["out"] == original, f"RLGR roundtrip failed for {name}: decoded values don't match encoded values"
        
        size_bytes = sum(len(b['buf']) for b in compressed.values())
        rates[frame_idx, i] = size_bytes
        entropy_enc_time[frame_idx, i] = sum(b["time_ns"] for b in compressed.values()) / 1e9
        entropy_dec_time[frame_idx, i] = sum(b["time_ns"] for b in decoded.values()) / 1e9
        
        coeff_dec_cpu = np.stack(
            (decoded["Y"]["out"], decoded["U"]["out"], decoded["V"]["out"]),
            axis=1
        ).astype(np.int32, copy=False)
        Coeff_dec = torch.from_numpy(coeff_dec_cpu).pin_memory().to(device=device, dtype=DTYPE, non_blocking=True)

        start_time = time.time()
        Coeff_dec = Coeff_dec * step
        dequant_time[frame_idx, i] = time.time() - start_time
        
        start_time = time.time()
        order_RAGFT_dec = torch.argsort(order_RAGFT)
        Coeff_dec = Coeff_dec[order_RAGFT_dec,:]
        C_rec = raht_fn["iRAHT"](Coeff_dec, ListC, FlagsC, weightsC)
        iRAHT_time[frame_idx, i] = time.time() - start_time

        # Verify full pipeline reconstruction (quantization causes expected loss)
        if DEBUG and i == 0:  # Only check for step=1 (minimal quantization)
            reconstruction_error = torch.abs(C - C_rec).max()
            print(f"Full pipeline reconstruction error (Quantization={step}): {reconstruction_error:.6e}")
            print(f"Reconstruction check passes: {torch.allclose(C, C_rec, rtol=1e-3, atol=step)}")
        
        logger.info(
            f"{frame},{colorStep[i]},{rates[frame_idx, i]*8/Nvox[frame_idx]:.6f},{raht_param_time[frame_idx, i]:.6f},{raht_transform_time[frame_idx, i]:.6f},"
            f"{quant_time[frame_idx, i]:.6f},{entropy_enc_time[frame_idx, i]:.6f},"
            f"{entropy_dec_time[frame_idx, i]:.6f},{dequant_time[frame_idx, i]:.6f},{iRAHT_time[frame_idx, i]:.6f},{psnr[frame_idx, i]:.6f}")

    print(f"Frame {frame}")

