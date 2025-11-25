#!/usr/bin/env python3
"""
Test script for actual 3DGS compression using Nvox Gaussians.

This script demonstrates Goal 2: Actual Compression/Deployment
- Compresses N Gaussians to Nvox merged Gaussians
- Renders directly with Nvox (no expansion to N)
- Measures compression quality (how much quality loss from reducing Gaussian count)
- Evaluates file size reduction and rendering speedup
"""

import torch
import os
import time

# Import merge_gaussian_clusters from the installed merge_cluster_cuda library
from merge_cluster_cuda import merge_gaussian_clusters_with_indices

# Import from local python directory
from voxelize_pc import voxelize_pc_batched
from quality_eval import (
    save_ply,
    try_render_comparison
)


def load_3dgs_checkpoint(ckpt_path, device='cuda'):
    """Load 3DGS checkpoint and extract Gaussian parameters."""
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    return checkpoint


def extract_gaussian_params(checkpoint, device='cuda'):
    """Extract Gaussian parameters from checkpoint."""
    if 'splats' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'splats' key")

    splats = checkpoint['splats']
    params = {}

    # Extract means (positions)
    if 'means' not in splats:
        raise ValueError("Missing 'means' in splats")
    params['means'] = splats['means'].to(device).float()

    # Extract quaternions (rotations)
    if 'quats' not in splats:
        raise ValueError("Missing 'quats' in splats")
    params['quats'] = splats['quats'].to(device).float()
    # Normalize quaternions
    params['quats'] = params['quats'] / params['quats'].norm(dim=1, keepdim=True)

    # Extract scales
    if 'scales' not in splats:
        raise ValueError("Missing 'scales' in splats")
    params['scales'] = splats['scales'].to(device).float()
    # Scales might be in log space, exponentiate if needed
    if params['scales'].min() < 0:
        params['scales'] = torch.exp(params['scales'])

    # Extract opacities
    if 'opacities' not in splats:
        raise ValueError("Missing 'opacities' in splats")
    params['opacities'] = splats['opacities'].to(device).float().squeeze()
    # Opacities might be in logit space, apply sigmoid if needed
    if params['opacities'].min() < 0 or params['opacities'].max() > 1:
        params['opacities'] = torch.sigmoid(params['opacities'])

    # Extract colors from SH coefficients
    if 'sh0' in splats:
        sh0 = splats['sh0'].to(device).float()
        # Flatten if needed (e.g., [N, 3, 1] -> [N, 3])
        if sh0.ndim > 2:
            sh0 = sh0.reshape(sh0.shape[0], -1)

        if 'shN' in splats and splats['shN'] is not None:
            shN = splats['shN'].to(device).float()
            # Flatten if needed
            if shN.ndim > 2:
                shN = shN.reshape(shN.shape[0], -1)
            # Concatenate sh0 and shN
            params['colors'] = torch.cat([sh0, shN], dim=1)
        else:
            # Only use sh0 if shN is not available
            params['colors'] = sh0
    else:
        raise ValueError("Missing 'sh0' in splats")

    return params


def warmup_cuda_kernels(params, J, device):
    """
    Warmup CUDA kernels to avoid JIT compilation overhead during timing.

    Args:
        params: Gaussian parameters dictionary
        J: Octree depth for voxelization
        device: CUDA device
    """
    N = params['means'].shape[0]
    positions = params['means']

    # Warmup voxelization
    for _ in range(3):
        voxelize_pc_batched(positions, J=J, device=device)

    # Warmup merge - need dummy cluster indices from voxelization
    dummy_pcvox, dummy_pcsorted, dummy_voxel_indices, dummy_deltapc, dummy_info = voxelize_pc_batched(
        positions, J=J, device=device
    )
    dummy_sort_idx = dummy_info['sort_idx']
    dummy_cluster_indices = dummy_sort_idx.int()
    dummy_cluster_offsets = torch.cat([
        dummy_voxel_indices,
        torch.tensor([N], dtype=torch.int32, device=device)
    ]).int()

    for _ in range(3):
        _ = merge_gaussian_clusters_with_indices(
            params['means'],
            params['quats'],
            params['scales'],
            params['opacities'],
            params['colors'],
            dummy_cluster_indices,
            dummy_cluster_offsets,
            weight_by_opacity=True
        )

    # Ensure all warmup operations complete before returning
    torch.cuda.synchronize()


def compress_to_nvox(ckpt_path, J=10, output_dir="output_compressed", device='cuda'):
    """
    Compress 3DGS from N to Nvox Gaussians.

    This function demonstrates actual compression:
    - Voxelize positions
    - Merge all attributes
    - Render with Nvox Gaussians (no expansion)
    - Compare quality: N original vs Nvox compressed

    Args:
        ckpt_path: Path to the 3DGS checkpoint
        J: Octree depth for voxelization
        output_dir: Directory to save output PLY files
        device: CUDA device to use (e.g., 'cuda', 'cuda:0', 'cuda:1')
    """
    print("=" * 80)
    print("3DGS Compression: N → Nvox Gaussians")
    print("=" * 80)
    print(f"Using device: {device}")

    # Load checkpoint and extract parameters
    checkpoint = load_3dgs_checkpoint(ckpt_path, device=device)
    params = extract_gaussian_params(checkpoint, device=device)

    N = params['means'].shape[0]
    print(f"Number of Gaussians: {N}")

    # Warmup CUDA kernels to avoid JIT compilation overhead
    warmup_cuda_kernels(params, J, device)

    # ========== COMPRESSION PIPELINE ==========
    print(f"\n" + "=" * 80)
    print(f"COMPRESSION PIPELINE (J={J})")
    print("=" * 80)

    # Start timing the full compression pipeline (voxelization + cluster construction + merge)
    # Synchronize once at the start to ensure clean slate
    torch.cuda.synchronize()
    compression_start_time = time.time()
    voxel_start_time = time.time()

    # 1. Voxelize positions
    PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc_batched(
        params['means'], J=J, device=device
    )

    # Measure synchronization time (= GPU execution time)
    voxel_pre_sync = time.time()
    torch.cuda.synchronize()
    voxel_post_sync = time.time()
    voxel_sync_time = voxel_post_sync - voxel_pre_sync
    voxel_elapsed_time = voxel_post_sync - voxel_start_time

    Nvox = voxel_info['Nvox']

    print(f"⏱️  Voxelization time: {voxel_elapsed_time*1000:.2f} ms (GPU wait: {voxel_sync_time*1000:.2f} ms)")
    print(f"📊 Compression ratio: {N / Nvox:.2f}x ({N} → {Nvox} Gaussians)")
    print(f"📏 Voxel size: {voxel_info['voxel_size']:.6f}")

    # 2. Construct cluster indices directly from voxelization output
    cluster_start_time = time.time()

    # sort_idx tells us which original Gaussian each sorted position corresponds to
    # This is exactly what cluster_indices needs - an indirection array!
    sort_idx = voxel_info['sort_idx']
    cluster_indices = sort_idx.int()

    # cluster_offsets marks the boundaries of each cluster (voxel)
    # voxel_indices already tells us where each voxel starts, just append N
    cluster_offsets = torch.cat([
        voxel_indices,
        torch.tensor([N], dtype=torch.int32, device=device)
    ]).int()

    # Measure synchronization time
    cluster_pre_sync = time.time()
    torch.cuda.synchronize()
    cluster_post_sync = time.time()
    cluster_sync_time = cluster_post_sync - cluster_pre_sync
    cluster_elapsed_time = cluster_post_sync - cluster_start_time

    print(f"⏱️  Cluster construction time: {cluster_elapsed_time*1000:.2f} ms (GPU wait: {cluster_sync_time*1000:.2f} ms)")

    # 3. Merge all attributes
    merge_start_time = time.time()

    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = \
        merge_gaussian_clusters_with_indices(
            params['means'],
            params['quats'],
            params['scales'],
            params['opacities'],
            params['colors'],
            cluster_indices,
            cluster_offsets,
            weight_by_opacity=True
        )

    # Measure synchronization time and calculate total
    merge_pre_sync = time.time()
    torch.cuda.synchronize()
    merge_post_sync = time.time()
    merge_sync_time = merge_post_sync - merge_pre_sync
    merge_elapsed_time = merge_post_sync - merge_start_time
    compression_elapsed_time = merge_post_sync - compression_start_time

    print(f"⏱️  Attribute merging time: {merge_elapsed_time*1000:.2f} ms (GPU wait: {merge_sync_time*1000:.2f} ms)")
    print(f"⏱️  Total compression time: {compression_elapsed_time*1000:.2f} ms")

    # Use PCvox integer positions instead of merged_means
    # PCvox[:, :3] contains integer voxel coordinates - this is what RAHT needs
    # For rendering/PLY, convert back to world coordinates (voxel centers)
    voxel_positions_int = PCvox[:, :3]  # Integer voxel coordinates [0, 2^J - 1]
    voxel_positions_world = (voxel_positions_int + 0.5) * voxel_info['voxel_size'] + voxel_info['vmin']

    # 4. Save compressed PLY files
    os.makedirs(output_dir, exist_ok=True)

    # Save original N Gaussians
    original_ply_path = os.path.join(output_dir, "original_N_gaussians.ply")
    save_ply(original_ply_path, params['means'], params['quats'], params['scales'],
             params['opacities'], params['colors'])

    # Save compressed Nvox Gaussians using integer voxel coordinates
    compressed_ply_path = os.path.join(output_dir, "compressed_Nvox_gaussians.ply")
    save_ply(compressed_ply_path, voxel_positions_int, merged_quats, merged_scales,
             merged_opacities, merged_colors)

    # 5. File size comparison
    import os as os_module
    original_size = os_module.path.getsize(original_ply_path)
    compressed_size = os_module.path.getsize(compressed_ply_path)
    size_reduction = (1 - compressed_size / original_size) * 100

    print(f"\n" + "=" * 80)
    print("FILE SIZE COMPARISON")
    print("=" * 80)
    print(f"📁 Original (N={N}): {original_size / 1024 / 1024:.2f} MB")
    print(f"📁 Compressed (Nvox={Nvox}): {compressed_size / 1024 / 1024:.2f} MB")
    print(f"💾 Size reduction: {size_reduction:.1f}%")

    # 6. Rendering comparison: N original vs Nvox compressed
    print(f"\n" + "=" * 80)
    print("QUALITY EVALUATION")
    print("=" * 80)
    print(f"Comparing: {N} original Gaussians vs {Nvox} compressed Gaussians")

    # Prepare original params (N Gaussians)
    original_params = {
        'means': params['means'],
        'quats': params['quats'],
        'scales': params['scales'],
        'opacities': params['opacities'],
        'colors': params['colors']
    }

    # Prepare compressed params (Nvox Gaussians)
    compressed_params = {
        'means': voxel_positions_world,
        'quats': merged_quats,
        'scales': merged_scales,
        'opacities': merged_opacities,
        'colors': merged_colors
    }

    render_output_dir = os.path.join(output_dir, "renders")
    rendering_metrics = try_render_comparison(
        original_params,
        compressed_params,
        n_views=50,
        output_dir=render_output_dir
    )

    return {
        'original_count': N,
        'compressed_count': Nvox,
        'compression_ratio': N / Nvox,
        'voxel_time_ms': voxel_elapsed_time * 1000,
        'voxel_sync_ms': voxel_sync_time * 1000,
        'cluster_time_ms': cluster_elapsed_time * 1000,
        'cluster_sync_ms': cluster_sync_time * 1000,
        'merge_time_ms': merge_elapsed_time * 1000,
        'merge_sync_ms': merge_sync_time * 1000,
        'total_time_ms': compression_elapsed_time * 1000,
        'original_size_mb': original_size / 1024 / 1024,
        'compressed_size_mb': compressed_size / 1024 / 1024,
        'size_reduction_percent': size_reduction,
        'rendering_metrics': rendering_metrics,
        'original_ply_path': original_ply_path,
        'compressed_ply_path': compressed_ply_path,
    }


if __name__ == '__main__':
    ckpt_path = "/ssd1/rajrup/Project/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt"

    try:
        results = compress_to_nvox(
            ckpt_path,
            J=10,  # Octree depth for voxelization
            output_dir="output_compressed",
            device="cuda:0"  # Change to "cuda:0", "cuda:1", etc. to use a specific GPU
        )

        print("\n" + "=" * 80)
        print("COMPRESSION RESULTS SUMMARY")
        print("=" * 80)
        print(f"Gaussians: {results['original_count']} → {results['compressed_count']} ({results['compression_ratio']:.2f}x)")
        print(f"⏱️  Total compression time: {results['total_time_ms']:.2f} ms")
        print(f"  ├─ Voxelization: {results['voxel_time_ms']:.2f} ms (GPU: {results['voxel_sync_ms']:.2f} ms)")
        print(f"  ├─ Cluster construction: {results['cluster_time_ms']:.2f} ms (GPU: {results['cluster_sync_ms']:.2f} ms)")
        print(f"  └─ Merging: {results['merge_time_ms']:.2f} ms (GPU: {results['merge_sync_ms']:.2f} ms)")
        print(f"💾 File size: {results['original_size_mb']:.2f} MB → {results['compressed_size_mb']:.2f} MB ({results['size_reduction_percent']:.1f}% reduction)")

        if results['rendering_metrics']:
            render_metrics = results['rendering_metrics']
            print(f"🎨 PSNR: {render_metrics['psnr_avg']:.2f} ± {render_metrics['psnr_std']:.2f} dB")
            print(f"   Range: [{render_metrics['psnr_min']:.2f}, {render_metrics['psnr_max']:.2f}] dB")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
