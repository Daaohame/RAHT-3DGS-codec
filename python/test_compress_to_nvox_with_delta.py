#!/usr/bin/env python3
"""
Test script for 3DGS compression using Nvox Gaussians with position delta encoding.

This script demonstrates compression with position residuals:
- Uses integer voxel positions (PCvox) for RAHT tree structure
- Encodes position delta (merged_means - voxel_center) as an attribute
- Reconstructs high-quality positions: voxel_center + delta

This approach preserves position accuracy while using integer coordinates for RAHT.
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


def compress_to_nvox_with_delta(ckpt_path, J=10, output_dir="output_compressed_delta", device='cuda'):
    """
    Compress 3DGS from N to Nvox Gaussians with position delta encoding.

    This function demonstrates compression with position residuals:
    - Uses integer voxel positions (PCvox) for RAHT tree structure
    - Computes position delta: merged_means - voxel_center_world
    - Reconstructs positions for rendering: voxel_center + delta

    Args:
        ckpt_path: Path to the 3DGS checkpoint
        J: Octree depth for voxelization
        output_dir: Directory to save output PLY files
        device: CUDA device to use (e.g., 'cuda', 'cuda:0', 'cuda:1')
    """
    print("=" * 80)
    print("3DGS Compression: N → Nvox Gaussians (with Position Delta)")
    print("=" * 80)
    print(f"Using device: {device}")

    # Load checkpoint and extract parameters
    checkpoint = load_3dgs_checkpoint(ckpt_path, device=device)
    params = extract_gaussian_params(checkpoint, device=device)

    N = params['means'].shape[0]
    print(f"Number of Gaussians: {N}")

    # 1. Voxelize positions
    positions = params['means']

    # Warmup
    for _ in range(3):
        voxelize_pc_batched(positions, J=J, device=device)

    # Timed voxelization
    print(f"\n" + "=" * 80)
    print(f"COMPRESSION PIPELINE (J={J})")
    print("=" * 80)

    torch.cuda.synchronize()
    voxel_start_time = time.time()

    PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc_batched(
        positions, J=J, device=device
    )

    torch.cuda.synchronize()
    voxel_elapsed_time = time.time() - voxel_start_time

    Nvox = voxel_info['Nvox']

    print(f"Voxelization time: {voxel_elapsed_time*1000:.2f} ms")
    print(f"Compression ratio: {N / Nvox:.2f}x ({N} → {Nvox} Gaussians)")
    print(f"Voxel size: {voxel_info['voxel_size']:.6f}")

    # 2. Construct cluster indices directly from voxelization output
    sort_idx = voxel_info['sort_idx']
    cluster_indices = sort_idx.int()

    cluster_offsets = torch.cat([
        voxel_indices,
        torch.tensor([N], dtype=torch.int32, device=device)
    ]).int()

    # 3. Merge all attributes
    # Warmup
    for _ in range(3):
        _ = merge_gaussian_clusters_with_indices(
            params['means'],
            params['quats'],
            params['scales'],
            params['opacities'],
            params['colors'],
            cluster_indices,
            cluster_offsets,
            weight_by_opacity=True
        )

    torch.cuda.synchronize()
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

    torch.cuda.synchronize()
    merge_elapsed_time = time.time() - merge_start_time

    print(f"Attribute merging time: {merge_elapsed_time*1000:.2f} ms")
    print(f"Total compression time: {(voxel_elapsed_time + merge_elapsed_time)*1000:.2f} ms")

    # 4. Compute position delta
    # PCvox[:, :3] contains integer voxel coordinates
    # Convert to world coordinates (voxel centers)
    voxel_positions_int = PCvox[:, :3]  # Integer voxel coordinates [0, 2^J - 1]
    voxel_positions_world = (voxel_positions_int + 0.5) * voxel_info['voxel_size'] + voxel_info['vmin']

    # Position delta: difference between merged_means and voxel center
    # This will be encoded as an attribute via RAHT
    position_delta = merged_means - voxel_positions_world

    # Statistics on position delta
    delta_magnitude = torch.norm(position_delta, dim=1)
    print(f"\nPosition Delta Statistics:")
    print(f"  Mean magnitude: {delta_magnitude.mean().item():.6f}")
    print(f"  Max magnitude: {delta_magnitude.max().item():.6f}")
    print(f"  Relative to voxel size: {delta_magnitude.mean().item() / voxel_info['voxel_size']:.2%}")

    # 5. Reconstruct positions for rendering
    # In actual pipeline: decode voxel_positions_int from RAHT tree structure
    #                     decode position_delta from RAHT attribute encoding
    #                     final_position = voxel_center + delta
    reconstructed_positions = voxel_positions_world + position_delta

    # Verify reconstruction matches merged_means
    reconstruction_error = torch.norm(reconstructed_positions - merged_means, dim=1).max().item()
    print(f"  Reconstruction error (should be ~0): {reconstruction_error:.2e}")

    # 6. Save compressed PLY files
    os.makedirs(output_dir, exist_ok=True)

    # Save original N Gaussians
    original_ply_path = os.path.join(output_dir, "original_N_gaussians.ply")
    save_ply(original_ply_path, params['means'], params['quats'], params['scales'],
             params['opacities'], params['colors'])

    # Save compressed Nvox Gaussians (using reconstructed positions)
    compressed_ply_path = os.path.join(output_dir, "compressed_Nvox_gaussians.ply")
    save_ply(compressed_ply_path, reconstructed_positions, merged_quats, merged_scales,
             merged_opacities, merged_colors)

    # 7. File size comparison
    import os as os_module
    original_size = os_module.path.getsize(original_ply_path)
    compressed_size = os_module.path.getsize(compressed_ply_path)
    size_reduction = (1 - compressed_size / original_size) * 100

    print(f"\n" + "=" * 80)
    print("FILE SIZE COMPARISON")
    print("=" * 80)
    print(f"Original (N={N}): {original_size / 1024 / 1024:.2f} MB")
    print(f"Compressed (Nvox={Nvox}): {compressed_size / 1024 / 1024:.2f} MB")
    print(f"Size reduction: {size_reduction:.1f}%")
    print(f"\nNote: Position delta (3 floats per Gaussian) adds {Nvox * 3 * 4 / 1024 / 1024:.2f} MB")
    print(f"      This will be RAHT-encoded for additional compression")

    # 8. Rendering comparison
    print(f"\n" + "=" * 80)
    print("QUALITY EVALUATION")
    print("=" * 80)
    print(f"Comparing: {N} original Gaussians vs {Nvox} compressed Gaussians")
    print(f"Using reconstructed positions (voxel_center + delta)")

    # Prepare original params (N Gaussians)
    original_params = {
        'means': params['means'],
        'quats': params['quats'],
        'scales': params['scales'],
        'opacities': params['opacities'],
        'colors': params['colors']
    }

    # Prepare compressed params (Nvox Gaussians with reconstructed positions)
    compressed_params = {
        'means': reconstructed_positions,
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
        'merge_time_ms': merge_elapsed_time * 1000,
        'total_time_ms': (voxel_elapsed_time + merge_elapsed_time) * 1000,
        'original_size_mb': original_size / 1024 / 1024,
        'compressed_size_mb': compressed_size / 1024 / 1024,
        'size_reduction_percent': size_reduction,
        'rendering_metrics': rendering_metrics,
        'original_ply_path': original_ply_path,
        'compressed_ply_path': compressed_ply_path,
        # Position delta info (for RAHT encoding)
        'voxel_positions_int': voxel_positions_int,
        'position_delta': position_delta,
        'delta_mean_magnitude': delta_magnitude.mean().item(),
        'delta_max_magnitude': delta_magnitude.max().item(),
    }


if __name__ == '__main__':
    ckpt_path = "/ssd1/rajrup/Project/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt"

    try:
        results = compress_to_nvox_with_delta(
            ckpt_path,
            J=10,  # Octree depth for voxelization
            output_dir="output_compressed_delta",
            device="cuda:0"
        )

        print("\n" + "=" * 80)
        print("COMPRESSION RESULTS SUMMARY")
        print("=" * 80)
        print(f"Gaussians: {results['original_count']} → {results['compressed_count']} ({results['compression_ratio']:.2f}x)")
        print(f"Total compression time: {results['total_time_ms']:.2f} ms")
        print(f"  Voxelization: {results['voxel_time_ms']:.2f} ms")
        print(f"  Merging: {results['merge_time_ms']:.2f} ms")
        print(f"File size: {results['original_size_mb']:.2f} MB → {results['compressed_size_mb']:.2f} MB ({results['size_reduction_percent']:.1f}% reduction)")
        print(f"\nPosition Delta:")
        print(f"  Mean magnitude: {results['delta_mean_magnitude']:.6f}")
        print(f"  Max magnitude: {results['delta_max_magnitude']:.6f}")

        if results['rendering_metrics']:
            render_metrics = results['rendering_metrics']
            print(f"\nRendering Quality:")
            print(f"  PSNR: {render_metrics['psnr_avg']:.2f} ± {render_metrics['psnr_std']:.2f} dB")
            print(f"  Range: [{render_metrics['psnr_min']:.2f}, {render_metrics['psnr_max']:.2f}] dB")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
