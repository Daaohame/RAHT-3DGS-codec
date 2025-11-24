#!/usr/bin/env python3
"""
Test script for merge_cluster_cuda using real 3DGS checkpoint data.

This script loads a pre-trained 3DGS model checkpoint and tests the cluster merging functionality.
"""

import torch
import os
import time

# Import merge_gaussian_clusters from the installed merge_cluster_cuda library
from merge_cluster_cuda import merge_gaussian_clusters

# Import from local python directory
from voxelize_pc import voxelize_pc_batched
from quality_eval import (
    save_ply,
    compute_attribute_metrics,
    print_metrics,
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
    # Combine sh0 (DC component) and shN (higher order) if available
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


def create_cluster_labels(N, num_clusters, strategy='random'):
    """Create cluster labels for testing."""
    if strategy == 'random':
        return torch.randint(0, num_clusters, (N,), device='cuda')
    elif strategy == 'spatial':
        # This would require actual spatial positions
        # For now, just return random
        return torch.randint(0, num_clusters, (N,), device='cuda')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def test_merge_cluster(ckpt_path, num_clusters=100, weight_by_opacity=True, J=10, device='cuda'):
    """Main test function using voxelize_pc for positions and merge_cluster for attributes."""

    print("=" * 80)
    print("Testing Quantization + Merging Pipeline")
    print("=" * 80)
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = load_3dgs_checkpoint(ckpt_path, device=device)
    params = extract_gaussian_params(checkpoint, device=device)

    N = params['means'].shape[0]
    color_dim = params['colors'].shape[1]
    print(f"Number of Gaussians: {N}")

    # Step 1: Quantize positions only
    print(f"\n" + "=" * 80)
    print(f"QUANTIZATION PIPELINE (J={J})")
    print("=" * 80)

    # Warmup
    for _ in range(3):
        voxelize_pc_batched(params['means'], J=J, device=device)

    torch.cuda.synchronize()
    voxel_start_time = time.time()

    PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc_batched(
        params['means'],  # Only positions, shape [N, 3]
        J=J,
        device=device
    )

    torch.cuda.synchronize()
    voxel_elapsed_time = time.time() - voxel_start_time

    Nvox = voxel_info['Nvox']

    print(f"⏱️  Voxelization time: {voxel_elapsed_time*1000:.2f} ms")
    print(f"📊 Compression ratio: {N / Nvox:.2f}x ({N} → {Nvox} Gaussians)")
    print(f"📏 Voxel size: {voxel_info['voxel_size']:.6f}")

    # Step 2: Create cluster labels from voxel assignments
    voxel_counts_int = torch.diff(
        torch.cat([voxel_indices, torch.tensor([N], device=device)])
    )

    voxel_id_sorted = torch.repeat_interleave(
        torch.arange(Nvox, device=device),
        voxel_counts_int
    )

    sort_idx = voxel_info['sort_idx']
    cluster_labels = torch.zeros(N, dtype=torch.long, device=device)
    cluster_labels[sort_idx] = voxel_id_sorted

    # Step 3: Merge attributes using cluster labels
    # Warmup
    for _ in range(3):
        _ = merge_gaussian_clusters(
            params['means'],
            params['quats'],
            params['scales'],
            params['opacities'],
            params['colors'],
            cluster_labels,
            weight_by_opacity=weight_by_opacity
        )

    torch.cuda.synchronize()
    start_time = time.time()

    merged_means_unused, merged_quats, merged_scales, merged_opacities, merged_colors = \
        merge_gaussian_clusters(
            params['means'],
            params['quats'],
            params['scales'],
            params['opacities'],
            params['colors'],
            cluster_labels,
            weight_by_opacity=weight_by_opacity
        )

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    print(f"⏱️  Attribute merging time: {elapsed_time*1000:.2f} ms")
    print(f"⏱️  Total time: {(voxel_elapsed_time + elapsed_time)*1000:.2f} ms")

    # Step 4: Use voxelized positions instead of merged positions
    # PCvox is [Nvox, 3] with integer voxel coordinates
    # convert back to world coordinates at voxel centers: world_pos = vmin + (voxel_coords + 0.5) * voxel_size
    merged_means = voxel_info['vmin'].unsqueeze(0) + (PCvox + 0.5) * voxel_info['voxel_size']

    # Verify results
    quat_norms = merged_quats.norm(dim=1)
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5), "Quaternions not normalized!"
    assert merged_opacities.min() >= 0 and merged_opacities.max() <= 1, "Opacities out of range!"
    for name, tensor in [('means', merged_means), ('quats', merged_quats),
                          ('scales', merged_scales), ('opacities', merged_opacities),
                          ('colors', merged_colors)]:
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        assert not has_nan and not has_inf, f"{name} contains NaN or Inf!"

    # Quality Evaluation
    print(f"\n" + "=" * 80)
    print("QUALITY EVALUATION")
    print("=" * 80)

    # Compute attribute quality metrics
    original_voxel_indices_direct = torch.floor((params['means'] - voxel_info['vmin'].unsqueeze(0)) / voxel_info['voxel_size']).long()
    reconstructed_means_correct = voxel_info['vmin'].unsqueeze(0) + (original_voxel_indices_direct.float() + 0.5) * voxel_info['voxel_size']

    metrics = compute_attribute_metrics(
        params['means'],
        params['quats'],
        params['scales'],
        params['opacities'],
        params['colors'],
        reconstructed_means_correct,
        merged_quats,
        merged_scales,
        merged_opacities,
        merged_colors,
        cluster_labels
    )

    # Save PLY files
    output_dir = "output_ply"
    os.makedirs(output_dir, exist_ok=True)

    original_ply_path = os.path.join(output_dir, "original_gaussians.ply")
    save_ply(
        original_ply_path,
        params['means'],
        params['quats'],
        params['scales'],
        params['opacities'],
        params['colors']
    )

    merged_ply_path = os.path.join(output_dir, "merged_gaussians.ply")
    save_ply(
        merged_ply_path,
        merged_means,
        merged_quats,
        merged_scales,
        merged_opacities,
        merged_colors
    )

    quantized_params = {
        'means': reconstructed_means_correct,  # Quantized positions (expanded to N)
        'quats': merged_quats[cluster_labels],  # Merged quats (expanded to N)
        'scales': merged_scales[cluster_labels],  # Merged scales (expanded to N)
        'opacities': merged_opacities[cluster_labels],  # Merged opacities (expanded to N)
        'colors': merged_colors[cluster_labels]  # Merged colors (expanded to N)
    }

    render_output_dir = os.path.join(output_dir, "renders")
    rendering_metrics = try_render_comparison(
        {
            'means': params['means'],
            'quats': params['quats'],
            'scales': params['scales'],
            'opacities': params['opacities'],
            'colors': params['colors']
        },
        quantized_params,
        n_views=50,
        output_dir=render_output_dir
    )

    return {
        'original_count': N,
        'merged_count': merged_means.shape[0],
        'compression_ratio': N / merged_means.shape[0],
        'voxel_time_ms': voxel_elapsed_time * 1000,
        'merge_time_ms': elapsed_time * 1000,
        'total_time_ms': (voxel_elapsed_time + elapsed_time) * 1000,
        'merged_means': merged_means,
        'merged_quats': merged_quats,
        'merged_scales': merged_scales,
        'merged_opacities': merged_opacities,
        'merged_colors': merged_colors,
        'quality_metrics': metrics,
        'rendering_metrics': rendering_metrics,
        'original_ply_path': original_ply_path,
        'merged_ply_path': merged_ply_path,
    }


if __name__ == '__main__':
    ckpt_path = "/ssd1/rajrup/Project/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt"

    try:
        results = test_merge_cluster(
            ckpt_path,
            J=10,  # Octree depth for voxelization
            weight_by_opacity=True,
            device="cuda:1"
        )

        print("\n" + "=" * 80)
        print("QUANTIZATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"Gaussians: {results['original_count']} → {results['merged_count']} ({results['compression_ratio']:.2f}x)")
        print(f"⏱️  Total time: {results['total_time_ms']:.2f} ms")
        print(f"  ├─ Voxelization: {results['voxel_time_ms']:.2f} ms")
        print(f"  └─ Merging: {results['merge_time_ms']:.2f} ms")

        # Show rendering metrics if available
        if results['rendering_metrics']:
            render_metrics = results['rendering_metrics']
            print(f"🎨 PSNR: {render_metrics['psnr_avg']:.2f} ± {render_metrics['psnr_std']:.2f} dB")
            print(f"   Range: [{render_metrics['psnr_min']:.2f}, {render_metrics['psnr_max']:.2f}] dB")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
