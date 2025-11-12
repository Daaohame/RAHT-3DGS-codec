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


def load_3dgs_checkpoint(ckpt_path):
    """Load 3DGS checkpoint and extract Gaussian parameters."""
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=True)

    # Print checkpoint structure to understand the data format
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: {checkpoint[key].shape} ({checkpoint[key].dtype})")
        else:
            print(f"  {key}: {type(checkpoint[key])}")

    return checkpoint


def extract_gaussian_params(checkpoint):
    """Extract Gaussian parameters from checkpoint."""
    # Checkpoint structure: checkpoint['splats'] contains the 3DGS data
    # with keys: ['means', 'opacities', 'quats', 'scales', 'sh0', 'shN']

    if 'splats' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'splats' key")

    splats = checkpoint['splats']

    print("\nSplats keys:")
    for key in splats.keys():
        if isinstance(splats[key], torch.Tensor):
            print(f"  {key}: {splats[key].shape} ({splats[key].dtype})")

    params = {}

    # Extract means (positions)
    if 'means' not in splats:
        raise ValueError("Missing 'means' in splats")
    params['means'] = splats['means'].cuda().float()

    # Extract quaternions (rotations)
    if 'quats' not in splats:
        raise ValueError("Missing 'quats' in splats")
    params['quats'] = splats['quats'].cuda().float()
    # Normalize quaternions
    params['quats'] = params['quats'] / params['quats'].norm(dim=1, keepdim=True)

    # Extract scales
    if 'scales' not in splats:
        raise ValueError("Missing 'scales' in splats")
    params['scales'] = splats['scales'].cuda().float()
    # Scales might be in log space, exponentiate if needed
    if params['scales'].min() < 0:
        params['scales'] = torch.exp(params['scales'])

    # Extract opacities
    if 'opacities' not in splats:
        raise ValueError("Missing 'opacities' in splats")
    params['opacities'] = splats['opacities'].cuda().float().squeeze()
    # Opacities might be in logit space, apply sigmoid if needed
    if params['opacities'].min() < 0 or params['opacities'].max() > 1:
        params['opacities'] = torch.sigmoid(params['opacities'])

    # Extract colors from SH coefficients
    # Combine sh0 (DC component) and shN (higher order) if available
    if 'sh0' in splats:
        sh0 = splats['sh0'].cuda().float()
        # Flatten if needed (e.g., [N, 3, 1] -> [N, 3])
        if sh0.ndim > 2:
            sh0 = sh0.reshape(sh0.shape[0], -1)

        if 'shN' in splats and splats['shN'] is not None:
            shN = splats['shN'].cuda().float()
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


def test_merge_cluster(ckpt_path, num_clusters=100, weight_by_opacity=True, J=10):
    """Main test function using voxelize_pc for positions and merge_cluster for attributes."""

    print("=" * 80)
    print("Testing voxelize_pc + merge_cluster with real 3DGS checkpoint")
    print("=" * 80)

    # Load checkpoint
    checkpoint = load_3dgs_checkpoint(ckpt_path)

    # Extract Gaussian parameters
    print("\n" + "=" * 80)
    print("Extracting Gaussian parameters...")
    print("=" * 80)
    params = extract_gaussian_params(checkpoint)

    N = params['means'].shape[0]
    color_dim = params['colors'].shape[1]

    print(f"\nExtracted parameters:")
    print(f"  Number of Gaussians: {N}")
    print(f"  Means shape: {params['means'].shape}")
    print(f"  Quats shape: {params['quats'].shape}")
    print(f"  Scales shape: {params['scales'].shape}")
    print(f"  Opacities shape: {params['opacities'].shape}")
    print(f"  Colors shape: {params['colors'].shape}")
    print(f"  Color dimension: {color_dim}")

    # Step 1: Quantize positions only
    print(f"\n" + "=" * 80)
    print(f"Step 1: Quantizing positions (J={J})...")
    print("=" * 80)

    print(f"  Warming up CUDA kernels...")
    for _ in range(3):
        voxelize_pc_batched(params['means'], J=J, device='cuda')

    torch.cuda.synchronize()
    voxel_start_time = time.time()

    PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc_batched(
        params['means'],  # Only positions, shape [N, 3]
        J=J,
        device='cuda'
    )

    torch.cuda.synchronize()
    voxel_elapsed_time = time.time() - voxel_start_time

    Nvox = voxel_info['Nvox']

    print(f"  Quantization completed in {voxel_elapsed_time*1000:.2f} ms")
    print(f"  Input Gaussians: {N}")
    print(f"  Output voxels: {Nvox}")
    print(f"  Voxel size: {voxel_info['voxel_size']:.6f}")
    print(f"  Compression ratio: {N / Nvox:.2f}x")

    # Step 2: Create cluster labels from voxel assignments
    print(f"\n" + "=" * 80)
    print(f"Step 2: Creating cluster labels from voxel assignments...")
    print("=" * 80)

    # Each point between voxel_indices[i] and voxel_indices[i+1] belongs to voxel i
    # We need to map this back to the original unsorted order
    voxel_counts_int = torch.diff(
        torch.cat([voxel_indices, torch.tensor([N], device='cuda')])
    )

    # Create voxel_id for sorted points [N]
    voxel_id_sorted = torch.repeat_interleave(
        torch.arange(Nvox, device='cuda'),
        voxel_counts_int
    )

    # Use the sort indices returned by voxelize_pc (no need to recalculate Morton codes)
    sort_idx = voxel_info['sort_idx']

    # Create inverse mapping to unsort cluster labels
    cluster_labels = torch.zeros(N, dtype=torch.long, device='cuda')
    cluster_labels[sort_idx] = voxel_id_sorted

    unique_clusters = torch.unique(cluster_labels)
    print(f"  Created {len(unique_clusters)} unique clusters (should equal {Nvox} voxels)")

    # Step 3: Merge attributes using cluster labels
    print(f"\n" + "=" * 80)
    print(f"Step 3: Merging attributes with merge_cluster (weight_by_opacity={weight_by_opacity})...")
    print("=" * 80)

    # Warmup to avoid CUDA JIT compilation overhead
    print(f"  Warming up CUDA kernels...")
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

    # Step 4: Use voxelized positions instead of merged positions
    print(f"\n" + "=" * 80)
    print(f"Step 4: Replacing merged positions with quantized voxelized positions...")
    print("=" * 80)

    # Verify that all quantized positions are integers (as expected from voxelization)
    is_integer = torch.all(PCvox == torch.floor(PCvox))
    print(f"  All quantized positions are integers: {is_integer.item()}")

    if is_integer:
        print(f"  Quantized position range:")
        print(f"    Min: {PCvox.min(dim=0)[0].cpu().numpy()}")
        print(f"    Max: {PCvox.max(dim=0)[0].cpu().numpy()}")

    # PCvox is [Nvox, 3] with integer voxel coordinates
    # convert back to world coordinates: world_pos = vmin + voxel_coords * voxel_size
    merged_means = voxel_info['vmin'].unsqueeze(0) + PCvox * voxel_info['voxel_size']

    # Print results
    print(f"\nMerge completed successfully!")
    print(f"  Voxelization time: {voxel_elapsed_time*1000:.2f} ms")
    print(f"  Attribute merging time: {elapsed_time*1000:.2f} ms")
    print(f"  Total time: {(voxel_elapsed_time + elapsed_time)*1000:.2f} ms")
    print(f"  Input Gaussians: {N}")
    print(f"  Output clusters: {merged_means.shape[0]}")
    print(f"  Compression ratio: {N / merged_means.shape[0]:.2f}x")

    print(f"\nMerged tensor shapes:")
    print(f"  Means (from voxelize_pc): {merged_means.shape}")
    print(f"  Quats (from merge_cluster): {merged_quats.shape}")
    print(f"  Scales (from merge_cluster): {merged_scales.shape}")
    print(f"  Opacities (from merge_cluster): {merged_opacities.shape}")
    print(f"  Colors (from merge_cluster): {merged_colors.shape}")


    # Verify results
    print(f"\n" + "=" * 80)
    print("Verifying results...")
    print("=" * 80)

    # Check quaternion normalization
    quat_norms = merged_quats.norm(dim=1)
    print(f"  Quaternion norms: min={quat_norms.min():.6f}, max={quat_norms.max():.6f}, mean={quat_norms.mean():.6f}")
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5), "Quaternions not normalized!"

    # Check opacity range
    print(f"  Opacity range: min={merged_opacities.min():.6f}, max={merged_opacities.max():.6f}")
    assert merged_opacities.min() >= 0 and merged_opacities.max() <= 1, "Opacities out of range!"

    # Check for NaN/Inf
    for name, tensor in [('means', merged_means), ('quats', merged_quats),
                          ('scales', merged_scales), ('opacities', merged_opacities),
                          ('colors', merged_colors)]:
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        print(f"  {name}: NaN={has_nan}, Inf={has_inf}")
        assert not has_nan and not has_inf, f"{name} contains NaN or Inf!"

    print(f"\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)

    # Quality Evaluation
    print(f"\n" + "=" * 80)
    print("Quality Evaluation")
    print("=" * 80)

    # 1. Compute attribute quality metrics
    print("\nComputing attribute quality metrics...")
    metrics = compute_attribute_metrics(
        params['means'],
        params['quats'],
        params['scales'],
        params['opacities'],
        params['colors'],
        merged_means,
        merged_quats,
        merged_scales,
        merged_opacities,
        merged_colors,
        cluster_labels
    )
    print_metrics(metrics, "Attribute Quality Metrics")

    # 2. Save PLY files for visualization
    print(f"\n" + "=" * 80)
    print("Saving PLY files for visualization...")
    print("=" * 80)

    # Create output directory
    output_dir = "output_ply"
    os.makedirs(output_dir, exist_ok=True)

    # Save original Gaussians
    original_ply_path = os.path.join(output_dir, "original_gaussians.ply")
    print(f"\nSaving original Gaussians to {original_ply_path}...")
    save_ply(
        original_ply_path,
        params['means'],
        params['quats'],
        params['scales'],
        params['opacities'],
        params['colors']
    )

    # Save merged Gaussians
    merged_ply_path = os.path.join(output_dir, "merged_gaussians.ply")
    print(f"\nSaving merged Gaussians to {merged_ply_path}...")
    save_ply(
        merged_ply_path,
        merged_means,
        merged_quats,
        merged_scales,
        merged_opacities,
        merged_colors
    )

    # 3. Try rendering comparison (if gsplat is available)
    rendering_metrics = try_render_comparison(
        {
            'means': params['means'],
            'quats': params['quats'],
            'scales': params['scales'],
            'opacities': params['opacities'],
            'colors': params['colors']
        },
        {
            'means': merged_means,
            'quats': merged_quats,
            'scales': merged_scales,
            'opacities': merged_opacities,
            'colors': merged_colors
        }
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
            weight_by_opacity=True
        )

        print("\n" + "=" * 80)
        print("Summary:")
        print("=" * 80)
        print(f"  Original Gaussians: {results['original_count']}")
        print(f"  Merged Clusters: {results['merged_count']}")
        print(f"  Compression: {results['compression_ratio']:.2f}x")
        print(f"  Voxelization time: {results['voxel_time_ms']:.2f} ms")
        print(f"  Attribute merging time: {results['merge_time_ms']:.2f} ms")
        print(f"  Total execution time: {results['total_time_ms']:.2f} ms")

        print("\nQuality Metrics:")
        metrics = results['quality_metrics']
        print(f"  Position RMSE: {metrics['position_rmse']:.6e}")
        print(f"  Quaternion mean dist: {metrics['quaternion_mean_dist']:.6e}")
        print(f"  Scale log RMSE: {metrics['scale_log_rmse']:.6e}")
        print(f"  Opacity RMSE: {metrics['opacity_rmse']:.6e}")
        print(f"  Color RMSE: {metrics['color_rmse']:.6e}")

        print("\nOutput Files:")
        print(f"  Original PLY: {results['original_ply_path']}")
        print(f"  Merged PLY: {results['merged_ply_path']}")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
