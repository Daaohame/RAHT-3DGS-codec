import torch
import numpy as np
import time
import logging
import math
import os

from data_util import read_compressed_3dgs_ply
from utils import save_mat, save_lists, sanity_check_vector
from RAHT import RAHT2_optimized
from iRAHT import inverse_RAHT_optimized
from RAHT_param import RAHT_param, RAHT_param_reorder_fast
from quality_eval import save_ply, try_render_comparison
import rlgr


## ---------------------
## Configuration
## ---------------------
torch.backends.cudnn.benchmark=False # for benchmarking
DEBUG = True  # Enable for correctness checks
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
raht_fn = {
    "RAHT": RAHT2_optimized,
    "iRAHT": inverse_RAHT_optimized,
    "RAHT_param": RAHT_param_reorder_fast
}

ply_list = ['/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/python/output_compressed/compressed_Nvox_gaussians.ply']
J = 10
T = len(ply_list)
colorStep = [1, 4, 8, 12, 16, 20, 24, 32, 64]
output_dir = 'output_compressed'


nSteps = len(colorStep)
rates = torch.zeros((T, nSteps), dtype=torch.float64)
raht_param_time = torch.zeros((T, nSteps), dtype=torch.float64)
raht_transform_time = torch.zeros((T, nSteps), dtype=torch.float64)
quant_time = torch.zeros((T, nSteps), dtype=torch.float64)
entropy_enc_time = torch.zeros((T, nSteps), dtype=torch.float64)
entropy_dec_time = torch.zeros((T, nSteps), dtype=torch.float64)
dequant_time = torch.zeros((T, nSteps), dtype=torch.float64)
iRAHT_time = torch.zeros((T, nSteps), dtype=torch.float64)
psnr = torch.zeros((T, nSteps), dtype=torch.float64)
Nvox = torch.zeros(T)


## ---------------------
## Logging setup
## ---------------------
log_filename = f'../results/runtime_3dgs.csv'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w')
    ]
)
logger = logging.getLogger(__name__)
logger.info("Frame,Quantization_Step,Rate_bpp,RAHT_prelude_time,RAHT_transform_time, Quant_time,Entropy_enc_time,Entropy_dec_time,Dequant_time,iRAHT_time,PSNR_all,PSNR_quats,PSNR_scales,PSNR_opacity,PSNR_colors")


## ---------------------
## Precision Setup
## ---------------------
use_fp64 = True  # set True only if RAHT requires double precision
DTYPE = torch.float64 if use_fp64 else torch.float32
def to_dev(x):
    return x.to(dtype=DTYPE, device=device, non_blocking=True)


## One dummy iteration to warm up GPU
print("Warming up GPU with a dummy iteration...")
result_dummy = read_compressed_3dgs_ply(ply_list[0])
if result_dummy is None:
    raise RuntimeError(f"Failed to load dummy frame from {ply_list[0]}")
V_dummy, attributes_dummy, _, _ = result_dummy  # Unpack 4 values, ignore voxel_size and vmin for warmup

attributes_dummy = attributes_dummy.to(dtype=DTYPE).contiguous()
C_dummy = to_dev(attributes_dummy)
V_dummy = V_dummy.to(dtype=DTYPE).to(device)

origin_dummy = torch.tensor([0, 0, 0], dtype=V_dummy.dtype, device=device)
ListC_dummy, FlagsC_dummy, weightsC_dummy, order_RAGFT_dummy = raht_fn["RAHT_param"](V_dummy, origin_dummy, 2 ** J, J)

ListC_dummy = [t.to(device=device, non_blocking=True) for t in ListC_dummy]
FlagsC_dummy = [t.to(device=device, non_blocking=True) for t in FlagsC_dummy]
weightsC_dummy = [t.to(device=device, non_blocking=True) for t in weightsC_dummy]

# Run through quantize/reorder/dequant path to mirror the main loop
Coeff_dummy, _ = raht_fn["RAHT"](C_dummy, ListC_dummy, FlagsC_dummy, weightsC_dummy)
step_dummy = colorStep[0]
Coeff_enc_dummy = torch.floor(Coeff_dummy / step_dummy + 0.5)
Coeff_enc_reordered = Coeff_enc_dummy.index_select(0, order_RAGFT_dummy)
order_RAGFT_dec_dummy = torch.argsort(order_RAGFT_dummy)
Coeff_dec_dummy = Coeff_enc_reordered[order_RAGFT_dec_dummy, :] * step_dummy
_ = raht_fn["iRAHT"](Coeff_dec_dummy, ListC_dummy, FlagsC_dummy, weightsC_dummy)

# Cleanup
del V_dummy, attributes_dummy, C_dummy, origin_dummy
del ListC_dummy, FlagsC_dummy, weightsC_dummy, Coeff_dummy, Coeff_enc_dummy
del Coeff_enc_reordered, Coeff_dec_dummy, order_RAGFT_dummy


## ---------------------
## Main Processing Loop
## ---------------------
print(f"\nStarting processing for {T} frames...")

for frame_idx in range(T):
    frame = frame_idx + 1

    V_quantized, attributes, voxel_size, vmin = read_compressed_3dgs_ply(ply_list[frame_idx])
    print(f"Loaded PLY: {V_quantized.shape[0]} Gaussians")
    print(f"  Integer positions shape: {V_quantized.shape}, range: [{V_quantized.min()}, {V_quantized.max()}]")
    print(f"  Attributes shape: {attributes.shape}")
    print(f"  Voxel metadata: voxel_size={voxel_size:.6f}, vmin={vmin.tolist()}")

    N = V_quantized.shape[0]
    n_channels = attributes.shape[1]
    Nvox[frame_idx] = N

    # Treat all attributes as "colors" for RAHT
    attributes = attributes.to(dtype=DTYPE).contiguous()
    C = to_dev(attributes)
    V = V_quantized.to(dtype=DTYPE).to(device)

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
    print(f"RAHT transform complete. Coeff shape: {Coeff.shape}")

    if DEBUG:
        save_lists(f"../results/frame{frame}_params_python.mat", ListC=ListC, FlagsC=FlagsC, weightsC=weightsC)
        save_mat(Coeff, f"../results/frame{frame}_coeff_python.mat")
        print(f"\n=== DEBUG INFO ===")
        print(f"Position range: [{V.min():.4f}, {V.max():.4f}]")
        print(f"Expected position range for J={J}: [0, {2**J - 1}]")

        # Check for duplicate positions
        V_cpu = V.cpu().long()
        unique_positions = torch.unique(V_cpu, dim=0)
        n_duplicates = N - unique_positions.shape[0]
        print(f"Unique positions: {unique_positions.shape[0]} / {N} (duplicates: {n_duplicates})")

        print(f"Attribute value range: [{C.min():.4f}, {C.max():.4f}]")
        print(f"Attribute channels: {n_channels}")
        print(f"  Quats (ch 0-3): [{C[:, 0:4].min():.4f}, {C[:, 0:4].max():.4f}]")
        print(f"  Scales (ch 4-6): [{C[:, 4:7].min():.4f}, {C[:, 4:7].max():.4f}]")
        print(f"  Opacity (ch 7): [{C[:, 7].min():.4f}, {C[:, 7].max():.4f}]")
        print(f"  Colors (ch 8+): [{C[:, 8:].min():.4f}, {C[:, 8:].max():.4f}]")
        print(f"Norm of attributes: {torch.norm(C):.2f}")
        print(f"Norm of Coeff: {torch.norm(Coeff):.2f}")

        # Verify lossless RAHT (use same function as production code)
        C_recon = raht_fn["iRAHT"](Coeff, ListC, FlagsC, weightsC)
        raht_error = torch.abs(C - C_recon).max()
        raht_error_mean = torch.abs(C - C_recon).mean()
        raht_error_rel = (raht_error / C.abs().max()).item()
        print(f"\nLossless RAHT max error: {raht_error:.2e}")
        print(f"Lossless RAHT mean error: {raht_error_mean:.2e}")
        print(f"Lossless RAHT relative error: {raht_error_rel:.2e} ({raht_error_rel*100:.2f}%)")
        print(f"Lossless RAHT check passes (loose): {torch.allclose(C, C_recon, rtol=1e-3, atol=1e-2)}")
        print(f"Lossless RAHT check passes (strict): {torch.allclose(C, C_recon, rtol=1e-5, atol=1e-8)}")
        print(f"===================\n")

    # Loop through quantization steps
    for i in range(nSteps):
        step = colorStep[i]

        # Analyze quantization appropriateness
        if DEBUG and i == 0:
            print(f"\n=== QUANTIZATION ANALYSIS (step={step}) ===")
            # Compute coefficient ranges per attribute type
            coeff_quats = Coeff[:, 0:4]
            coeff_scales = Coeff[:, 4:7]
            coeff_opacity = Coeff[:, 7]
            coeff_colors = Coeff[:, 8:]

            print(f"Coefficient ranges (before quantization):")
            print(f"  Quats (ch 0-3):   [{coeff_quats.min():.4f}, {coeff_quats.max():.4f}], range={coeff_quats.max()-coeff_quats.min():.4f}")
            print(f"  Scales (ch 4-6):  [{coeff_scales.min():.4f}, {coeff_scales.max():.4f}], range={coeff_scales.max()-coeff_scales.min():.4f}")
            print(f"  Opacity (ch 7):   [{coeff_opacity.min():.4f}, {coeff_opacity.max():.4f}], range={coeff_opacity.max()-coeff_opacity.min():.4f}")
            print(f"  Colors (ch 8+):   [{coeff_colors.min():.4f}, {coeff_colors.max():.4f}], range={coeff_colors.max()-coeff_colors.min():.4f}")

            print(f"\nQuantization step={step} relative to coefficient ranges:")
            print(f"  Quats:   step/range = {step/(coeff_quats.max()-coeff_quats.min()+1e-10):.4f} ({step/(coeff_quats.max()-coeff_quats.min()+1e-10)*100:.1f}%)")
            print(f"  Scales:  step/range = {step/(coeff_scales.max()-coeff_scales.min()+1e-10):.4f} ({step/(coeff_scales.max()-coeff_scales.min()+1e-10)*100:.1f}%)")
            print(f"  Opacity: step/range = {step/(coeff_opacity.max()-coeff_opacity.min()+1e-10):.4f} ({step/(coeff_opacity.max()-coeff_opacity.min()+1e-10)*100:.1f}%)")
            print(f"  Colors:  step/range = {step/(coeff_colors.max()-coeff_colors.min()+1e-10):.4f} ({step/(coeff_colors.max()-coeff_colors.min()+1e-10)*100:.1f}%)")

            print(f"\nNumber of quantization levels:")
            print(f"  Quats:   {int((coeff_quats.max()-coeff_quats.min())/step + 1)} levels")
            print(f"  Scales:  {int((coeff_scales.max()-coeff_scales.min())/step + 1)} levels")
            print(f"  Opacity: {int((coeff_opacity.max()-coeff_opacity.min())/step + 1)} levels")
            print(f"  Colors:  {int((coeff_colors.max()-coeff_colors.min())/step + 1)} levels")
            print(f"===========================================\n")

            # Study: Per-attribute quantization strategies
            print(f"\n=== PER-ATTRIBUTE QUANTIZATION STUDY ===")

            # Compute coefficient ranges
            range_quats = coeff_quats.max() - coeff_quats.min()
            range_scales = coeff_scales.max() - coeff_scales.min()
            range_opacity = coeff_opacity.max() - coeff_opacity.min()
            range_colors = coeff_colors.max() - coeff_colors.min()

            print(f"Current uniform quantization (step={step}):")
            print(f"  Problem: Same step size for all attributes ignores range differences")
            print(f"  Result: Scales get only {int(range_scales/step + 1)} levels vs Colors get {int(range_colors/step + 1)} levels")

            # Strategy 1: Range-normalized quantization
            # Keep the same number of quantization levels across all attributes
            print(f"\n--- Strategy 1: RANGE-NORMALIZED QUANTIZATION ---")
            print(f"Goal: Equal quantization levels for all attributes")

            # Choose target number of levels (e.g., 256 levels like 8-bit quantization)
            target_levels = 256
            step_quats_norm = range_quats / (target_levels - 1)
            step_scales_norm = range_scales / (target_levels - 1)
            step_opacity_norm = range_opacity / (target_levels - 1)
            step_colors_norm = range_colors / (target_levels - 1)

            print(f"Target quantization levels: {target_levels}")
            print(f"Per-attribute steps:")
            print(f"  Quats:   step={step_quats_norm:.4f} → {target_levels} levels")
            print(f"  Scales:  step={step_scales_norm:.4f} → {target_levels} levels (vs current {int(range_scales/step + 1)} levels)")
            print(f"  Opacity: step={step_opacity_norm:.4f} → {target_levels} levels")
            print(f"  Colors:  step={step_colors_norm:.4f} → {target_levels} levels")

            # Strategy 2: Visual importance weighted quantization
            # Allocate more bits to visually important attributes (from ablation study)
            print(f"\n--- Strategy 2: VISUAL IMPORTANCE WEIGHTED ---")
            print(f"Goal: More quantization levels for visually important attributes")
            print(f"From ablation study:")
            print(f"  Quats:   21.93 dB (most impactful → needs more levels)")
            print(f"  Scales:  26.36 dB (minimal impact)")
            print(f"  Opacity: 42.22 dB (least impactful → can use fewer levels)")
            print(f"  Colors:  38.67 dB (low impact)")

            # Importance weights (inverse of ablation PSNR - lower PSNR = more important)
            # Normalize so that total bitrate is comparable
            ablation_psnr = {'quats': 21.93, 'scales': 26.36, 'opacity': 42.22, 'colors': 38.67}
            # Lower PSNR = more important = needs more levels
            importance = {k: 1.0 / v for k, v in ablation_psnr.items()}
            total_importance = sum(importance.values())

            # Distribute total levels budget (e.g., 4 * 256 = 1024 levels total) based on importance
            total_levels_budget = 1024
            levels_quats_weighted = int(total_levels_budget * importance['quats'] / total_importance)
            levels_scales_weighted = int(total_levels_budget * importance['scales'] / total_importance)
            levels_opacity_weighted = int(total_levels_budget * importance['opacity'] / total_importance)
            levels_colors_weighted = int(total_levels_budget * importance['colors'] / total_importance)

            step_quats_weighted = range_quats / max(levels_quats_weighted - 1, 1)
            step_scales_weighted = range_scales / max(levels_scales_weighted - 1, 1)
            step_opacity_weighted = range_opacity / max(levels_opacity_weighted - 1, 1)
            step_colors_weighted = range_colors / max(levels_colors_weighted - 1, 1)

            print(f"Importance weights (normalized):")
            print(f"  Quats:   {importance['quats']/total_importance:.3f} → {levels_quats_weighted} levels, step={step_quats_weighted:.4f}")
            print(f"  Scales:  {importance['scales']/total_importance:.3f} → {levels_scales_weighted} levels, step={step_scales_weighted:.4f}")
            print(f"  Opacity: {importance['opacity']/total_importance:.3f} → {levels_opacity_weighted} levels, step={step_opacity_weighted:.4f}")
            print(f"  Colors:  {importance['colors']/total_importance:.3f} → {levels_colors_weighted} levels, step={step_colors_weighted:.4f}")

            # Strategy 3: Hybrid approach
            print(f"\n--- Strategy 3: HYBRID (Range-normalized + Importance-weighted) ---")
            print(f"Goal: Balance equal distortion per attribute with visual importance")

            # Combine both strategies: normalize ranges first, then weight by importance
            # Start with range-normalized steps, then scale by importance
            hybrid_weight = 0.5  # 50% range normalization, 50% importance weighting

            step_quats_hybrid = step_quats_norm * (1 - hybrid_weight) + step_quats_weighted * hybrid_weight
            step_scales_hybrid = step_scales_norm * (1 - hybrid_weight) + step_scales_weighted * hybrid_weight
            step_opacity_hybrid = step_opacity_norm * (1 - hybrid_weight) + step_opacity_weighted * hybrid_weight
            step_colors_hybrid = step_colors_norm * (1 - hybrid_weight) + step_colors_weighted * hybrid_weight

            levels_quats_hybrid = int(range_quats / step_quats_hybrid + 1)
            levels_scales_hybrid = int(range_scales / step_scales_hybrid + 1)
            levels_opacity_hybrid = int(range_opacity / step_opacity_hybrid + 1)
            levels_colors_hybrid = int(range_colors / step_colors_hybrid + 1)

            print(f"Hybrid steps (50% range-norm + 50% importance):")
            print(f"  Quats:   step={step_quats_hybrid:.4f} → {levels_quats_hybrid} levels")
            print(f"  Scales:  step={step_scales_hybrid:.4f} → {levels_scales_hybrid} levels")
            print(f"  Opacity: step={step_opacity_hybrid:.4f} → {levels_opacity_hybrid} levels")
            print(f"  Colors:  step={step_colors_hybrid:.4f} → {levels_colors_hybrid} levels")

            # Summary and recommendations
            print(f"\n--- RECOMMENDATIONS ---")
            print(f"Current uniform quantization issues:")
            print(f"  • Scales severely under-quantized: only {int(range_scales/step + 1)} levels")
            print(f"  • Quats are bottleneck: lowest rendering PSNR (21.93 dB)")
            print(f"  • Opacity/Colors over-quantized: high rendering PSNR despite reconstruction errors")

            print(f"\nRecommended strategy: Visual Importance Weighted")
            print(f"  • Allocate more bits to quats (most impactful): {levels_quats_weighted} levels")
            print(f"  • Reduce bits for opacity (least impactful): {levels_opacity_weighted} levels")
            print(f"  • Expected benefit: Better visual quality at same bitrate")

            print(f"\nImplementation: Use per-attribute quantization steps:")
            print(f"  Coeff_enc_quats = floor(Coeff_quats / {step_quats_weighted:.4f} + 0.5)")
            print(f"  Coeff_enc_scales = floor(Coeff_scales / {step_scales_weighted:.4f} + 0.5)")
            print(f"  Coeff_enc_opacity = floor(Coeff_opacity / {step_opacity_weighted:.4f} + 0.5)")
            print(f"  Coeff_enc_colors = floor(Coeff_colors / {step_colors_weighted:.4f} + 0.5)")
            print(f"===========================================\n")

        start_time = time.time()

        # Per-attribute quantization
        if i == 0 and DEBUG:
            print(f"\n=== APPLYING PER-ATTRIBUTE QUANTIZATION ===")

        # Define attribute channel ranges (adjust for different 3DGS formats)
        # Standard 3DGS: quats(4) + scales(3) + opacity(1) + colors(48) = 56
        attr_ranges = {
            'quats': (0, 4),
            'scales': (4, 7),
            'opacity': (7, 8),
            'colors': (8, n_channels)
        }

        # Visual importance weights (from ablation study)
        # Lower rendering PSNR = more important = needs more quantization levels
        # These weights are generally applicable to most 3DGS scenes
        importance_weights = {
            'quats': 1.0 / 21.93,    # Most impactful (lowest ablation PSNR)
            'scales': 1.0 / 26.36,   # Moderate impact
            'opacity': 1.0 / 42.22,  # Least impactful (highest ablation PSNR)
            'colors': 1.0 / 38.67    # Low impact
        }

        # Compute per-attribute quantization steps
        per_attr_steps = {}
        total_importance = sum(importance_weights.values())
        total_levels_budget = 1024  # Total quantization levels budget

        for attr_name, (start_ch, end_ch) in attr_ranges.items():
            if start_ch >= n_channels:
                continue  # Skip if attribute doesn't exist in this 3DGS

            # Get coefficient range for this attribute
            coeff_attr = Coeff[:, start_ch:end_ch]
            range_attr = coeff_attr.max() - coeff_attr.min()

            # Allocate quantization levels based on importance
            levels_attr = int(total_levels_budget * importance_weights[attr_name] / total_importance)
            levels_attr = max(levels_attr, 2)  # At least 2 levels

            # Compute quantization step for this attribute
            step_attr = range_attr / max(levels_attr - 1, 1)
            step_attr = max(step_attr, 1e-6)  # Avoid division by zero

            per_attr_steps[attr_name] = {
                'step': step_attr.item(),
                'levels': levels_attr,
                'range': range_attr.item(),
                'channels': (start_ch, end_ch)
            }

            if i == 0 and DEBUG:
                print(f"  {attr_name:8s}: step={step_attr.item():.4f}, levels={levels_attr}, range={range_attr.item():.2f}")

        # Apply per-attribute quantization
        Coeff_enc = torch.zeros_like(Coeff)
        for attr_name, info in per_attr_steps.items():
            start_ch, end_ch = info['channels']
            step_attr = info['step']
            Coeff_enc[:, start_ch:end_ch] = torch.floor(Coeff[:, start_ch:end_ch] / step_attr + 0.5)

        if i == 0 and DEBUG:
            print(f"===========================================\n")

        quant_time[frame_idx, i] = time.time() - start_time

        # Get reordered coefficients
        coeff_reordered = Coeff_enc.index_select(0, order_RAGFT)
        coeff_cpu_i32 = coeff_reordered.to('cpu', dtype=torch.int32, non_blocking=True)
        np_coeff = coeff_cpu_i32.numpy()  # zero-copy view on CPU

        # RLGR settings - encode all channels
        channels = {}
        for ch in range(n_channels):
            channels[f"ch{ch}"] = np_coeff[:, ch].tolist()

        flag_signed = 1  # 1 => signed integers
        compressed = {}     # name -> {"buf": list[uint8], "time_ns": int}
        decoded = {}        # name -> {"out": list[int], "time_ns": int}

        # encode all channels
        for name, data in channels.items():
            m_write = rlgr.membuf()
            encode_time_ns = m_write.rlgrWrite(data, flag_signed)
            m_write.close()
            buf = m_write.get_buffer()
            compressed[name] = {"buf": buf, "time_ns": encode_time_ns}

        # decode all channels
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

        # Reconstruct coefficient array from all decoded channels
        coeff_dec_list = [decoded[f"ch{ch}"]["out"] for ch in range(n_channels)]
        coeff_dec_cpu = np.stack(coeff_dec_list, axis=1).astype(np.int32, copy=False)
        Coeff_dec = torch.from_numpy(coeff_dec_cpu).pin_memory().to(device=device, dtype=DTYPE, non_blocking=True)

        start_time = time.time()
        # Apply per-attribute dequantization (each attribute has its own step)
        for attr_name, info in per_attr_steps.items():
            start_ch, end_ch = info['channels']
            step_attr = info['step']
            Coeff_dec[:, start_ch:end_ch] = Coeff_dec[:, start_ch:end_ch] * step_attr
        dequant_time[frame_idx, i] = time.time() - start_time

        start_time = time.time()
        order_RAGFT_dec = torch.argsort(order_RAGFT)
        Coeff_dec = Coeff_dec[order_RAGFT_dec,:]
        C_rec = raht_fn["iRAHT"](Coeff_dec, ListC, FlagsC, weightsC)
        iRAHT_time[frame_idx, i] = time.time() - start_time

        # Compute PSNR on decoded attributes (all channels)
        mse_all = torch.mean((C - C_rec) ** 2).item()
        psnr[frame_idx, i] = -10 * math.log10(mse_all + 1e-10)

        # Also compute per-attribute PSNR for analysis
        mse_quats = torch.mean((C[:, 0:4] - C_rec[:, 0:4]) ** 2).item()
        mse_scales = torch.mean((C[:, 4:7] - C_rec[:, 4:7]) ** 2).item()
        mse_opacity = torch.mean((C[:, 7] - C_rec[:, 7]) ** 2).item()
        mse_colors = torch.mean((C[:, 8:] - C_rec[:, 8:]) ** 2).item()

        psnr_quats = -10 * math.log10(mse_quats + 1e-10)
        psnr_scales = -10 * math.log10(mse_scales + 1e-10)
        psnr_opacity = -10 * math.log10(mse_opacity + 1e-10)
        psnr_colors = -10 * math.log10(mse_colors + 1e-10)

        # Verify full pipeline reconstruction (quantization causes expected loss)
        if DEBUG and i == 0:  # Only check for step=1 (minimal quantization)
            reconstruction_error = torch.abs(C - C_rec).max()
            print(f"Full pipeline reconstruction error (step={step}): {reconstruction_error:.6e}")
            print(f"Reconstruction check passes: {torch.allclose(C, C_rec, rtol=1e-3, atol=step)}")
            print(f"PSNR breakdown: All={psnr[frame_idx, i]:.2f}, Quats={psnr_quats:.2f}, Scales={psnr_scales:.2f}, Opacity={psnr_opacity:.2f}, Colors={psnr_colors:.2f}")

            # Render reconstructed 3DGS for visual quality check
            print(f"\n=== RENDERING RECONSTRUCTED 3DGS (step={step}) ===")

            # Convert integer positions to world coordinates using stored voxel_size and vmin
            voxel_positions_world = (V_quantized.float() + 0.5) * voxel_size + vmin

            print(f"  Integer position range: [{V_quantized.min()}, {V_quantized.max()}]")
            print(f"  World position range: [{voxel_positions_world.min():.4f}, {voxel_positions_world.max():.4f}]")
            print(f"  Voxel metadata: voxel_size={voxel_size:.6f}, vmin={vmin.tolist()}")

            # Split reconstructed attributes
            C_rec_cpu = C_rec.cpu()
            recon_quats = C_rec_cpu[:, 0:4]
            recon_scales = C_rec_cpu[:, 4:7]
            recon_opacities = C_rec_cpu[:, 7]
            recon_colors = C_rec_cpu[:, 8:]

            # Normalize quaternions (handle zero-norm case)
            quat_norms = recon_quats.norm(dim=1, keepdim=True)
            zero_norm_mask = (quat_norms.squeeze() < 1e-8)
            if zero_norm_mask.any():
                print(f"  Warning: {zero_norm_mask.sum()} quaternions have zero norm after reconstruction")
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=recon_quats.dtype, device=recon_quats.device)
                recon_quats[zero_norm_mask] = identity_quat
                quat_norms = recon_quats.norm(dim=1, keepdim=True)
            recon_quats = recon_quats / quat_norms

            # Ensure scales are positive and opacity in [0, 1]
            recon_scales = torch.abs(recon_scales)
            recon_opacities = torch.clamp(recon_opacities, 0, 1)

            # Prepare params for rendering (convert to float32 for gsplat)
            recon_params = {
                'means': voxel_positions_world.float().to(device),
                'quats': recon_quats.float().to(device),
                'scales': recon_scales.float().to(device),
                'opacities': recon_opacities.float().to(device),
                'colors': recon_colors.float().to(device)
            }

            # Original params for comparison
            C_orig_cpu = C.cpu()
            orig_quats = C_orig_cpu[:, 0:4]
            orig_scales = C_orig_cpu[:, 4:7]
            orig_opacities = C_orig_cpu[:, 7]
            orig_colors = C_orig_cpu[:, 8:]

            # Normalize original quaternions
            orig_quat_norms = orig_quats.norm(dim=1, keepdim=True)
            zero_norm_mask_orig = (orig_quat_norms.squeeze() < 1e-8)
            if zero_norm_mask_orig.any():
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=orig_quats.dtype)
                orig_quats[zero_norm_mask_orig] = identity_quat
                orig_quat_norms = orig_quats.norm(dim=1, keepdim=True)
            orig_quats = orig_quats / orig_quat_norms
            orig_scales = torch.abs(orig_scales)
            orig_opacities = torch.clamp(orig_opacities, 0, 1)

            orig_params = {
                'means': voxel_positions_world.float().to(device),
                'quats': orig_quats.float().to(device),
                'scales': orig_scales.float().to(device),
                'opacities': orig_opacities.float().to(device),
                'colors': orig_colors.float().to(device)
            }

            # Render comparison
            render_output_dir = os.path.join(output_dir, f"renders_step{step}_frame{frame}")
            print(f"  Rendering comparison (original attributes vs reconstructed)...")
            rendering_metrics = try_render_comparison(
                orig_params,
                recon_params,
                n_views=50,
                output_dir=render_output_dir
            )

            if rendering_metrics:
                print(f"  Rendering PSNR: {rendering_metrics['psnr_avg']:.2f} ± {rendering_metrics['psnr_std']:.2f} dB")
                print(f"  Renders saved to: {render_output_dir}")
            else:
                print(f"  ⚠ Rendering unavailable")

            # Ablation study: test each attribute individually
            print(f"\n  === ABLATION STUDY: Individual Attribute Impact ===")
            ablation_results = {}

            # Test 1: Only reconstructed quats (original scales/opacity/colors)
            ablation_params_quats = {
                'means': voxel_positions_world.float().to(device),
                'quats': recon_quats.float().to(device),  # Reconstructed
                'scales': orig_scales.float().to(device),  # Original
                'opacities': orig_opacities.float().to(device),  # Original
                'colors': orig_colors.float().to(device)  # Original
            }
            print(f"  Testing: reconstructed quats only...")
            metrics_quats = try_render_comparison(
                orig_params, ablation_params_quats, n_views=50,
                output_dir=os.path.join(output_dir, f"ablation_quats_step{step}_frame{frame}")
            )
            if metrics_quats:
                ablation_results['quats'] = metrics_quats['psnr_avg']
                print(f"    Quats only: {metrics_quats['psnr_avg']:.2f} dB")

            # Test 2: Only reconstructed scales (original quats/opacity/colors)
            ablation_params_scales = {
                'means': voxel_positions_world.float().to(device),
                'quats': orig_quats.float().to(device),  # Original
                'scales': recon_scales.float().to(device),  # Reconstructed
                'opacities': orig_opacities.float().to(device),  # Original
                'colors': orig_colors.float().to(device)  # Original
            }
            print(f"  Testing: reconstructed scales only...")
            metrics_scales = try_render_comparison(
                orig_params, ablation_params_scales, n_views=50,
                output_dir=os.path.join(output_dir, f"ablation_scales_step{step}_frame{frame}")
            )
            if metrics_scales:
                ablation_results['scales'] = metrics_scales['psnr_avg']
                print(f"    Scales only: {metrics_scales['psnr_avg']:.2f} dB")

            # Test 3: Only reconstructed opacity (original quats/scales/colors)
            ablation_params_opacity = {
                'means': voxel_positions_world.float().to(device),
                'quats': orig_quats.float().to(device),  # Original
                'scales': orig_scales.float().to(device),  # Original
                'opacities': recon_opacities.float().to(device),  # Reconstructed
                'colors': orig_colors.float().to(device)  # Original
            }
            print(f"  Testing: reconstructed opacity only...")
            metrics_opacity = try_render_comparison(
                orig_params, ablation_params_opacity, n_views=50,
                output_dir=os.path.join(output_dir, f"ablation_opacity_step{step}_frame{frame}")
            )
            if metrics_opacity:
                ablation_results['opacity'] = metrics_opacity['psnr_avg']
                print(f"    Opacity only: {metrics_opacity['psnr_avg']:.2f} dB")

            # Test 4: Only reconstructed colors (original quats/scales/opacity)
            ablation_params_colors = {
                'means': voxel_positions_world.float().to(device),
                'quats': orig_quats.float().to(device),  # Original
                'scales': orig_scales.float().to(device),  # Original
                'opacities': orig_opacities.float().to(device),  # Original
                'colors': recon_colors.float().to(device)  # Reconstructed
            }
            print(f"  Testing: reconstructed colors only...")
            metrics_colors = try_render_comparison(
                orig_params, ablation_params_colors, n_views=50,
                output_dir=os.path.join(output_dir, f"ablation_colors_step{step}_frame{frame}")
            )
            if metrics_colors:
                ablation_results['colors'] = metrics_colors['psnr_avg']
                print(f"    Colors only: {metrics_colors['psnr_avg']:.2f} dB")

            # Summary
            print(f"\n  === ABLATION SUMMARY ===")
            print(f"  Baseline (all original): inf dB")
            if rendering_metrics:
                print(f"  All reconstructed: {rendering_metrics['psnr_avg']:.2f} dB")
            for attr, psnr in ablation_results.items():
                print(f"  Reconstructed {attr:8s} only: {psnr:.2f} dB")

            # Identify the most impactful attribute
            if ablation_results:
                worst_attr = min(ablation_results.items(), key=lambda x: x[1])
                best_attr = max(ablation_results.items(), key=lambda x: x[1])
                print(f"\n  Most impactful attribute (lowest PSNR): {worst_attr[0]} ({worst_attr[1]:.2f} dB)")
                print(f"  Least impactful attribute (highest PSNR): {best_attr[0]} ({best_attr[1]:.2f} dB)")
            print(f"  ========================")

            print(f"===============================================\n")

        logger.info(
            f"{frame},{colorStep[i]},{rates[frame_idx, i].item()*8/Nvox[frame_idx].item():.6f},{raht_param_time[frame_idx, i].item():.6f},{raht_transform_time[frame_idx, i].item():.6f},"
            f"{quant_time[frame_idx, i].item():.6f},{entropy_enc_time[frame_idx, i].item():.6f},"
            f"{entropy_dec_time[frame_idx, i].item():.6f},{iRAHT_time[frame_idx, i].item():.6f},"
            f"{psnr[frame_idx, i].item():.6f},{psnr_quats:.6f},{psnr_scales:.6f},{psnr_opacity:.6f},{psnr_colors:.6f}")

    print(f"Frame {frame}")
