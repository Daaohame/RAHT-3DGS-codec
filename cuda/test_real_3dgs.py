#!/usr/bin/env python3
"""
Test script for merge_cluster_cuda using real 3DGS checkpoint data.

This script loads a pre-trained 3DGS model checkpoint and tests the cluster merging functionality.
"""

import torch
from merge_cluster import merge_gaussian_clusters
from voxelize_pc import voxelize_pc_batched
import time


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


def test_merge_cluster(ckpt_path, num_clusters=100, weight_by_opacity=True):
    """Main test function."""

    print("=" * 80)
    print("Testing merge_cluster_cuda with real 3DGS checkpoint")
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

    # Create cluster labels
    print(f"\n" + "=" * 80)
    print(f"Creating cluster labels (num_clusters={num_clusters})...")
    print("=" * 80)
    cluster_labels = create_cluster_labels(N, num_clusters, strategy='random')
    unique_clusters = torch.unique(cluster_labels)
    print(f"  Created {len(unique_clusters)} unique clusters")

    # Test merging
    print(f"\n" + "=" * 80)
    print(f"Testing cluster merging (weight_by_opacity={weight_by_opacity})...")
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

    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = \
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

    # Print results
    print(f"\nMerge completed successfully!")
    print(f"  Time taken: {elapsed_time*1000:.2f} ms")
    print(f"  Input Gaussians: {N}")
    print(f"  Output clusters: {merged_means.shape[0]}")
    print(f"  Compression ratio: {N / merged_means.shape[0]:.2f}x")

    print(f"\nMerged tensor shapes:")
    print(f"  Means: {merged_means.shape}")
    print(f"  Quats: {merged_quats.shape}")
    print(f"  Scales: {merged_scales.shape}")
    print(f"  Opacities: {merged_opacities.shape}")
    print(f"  Colors: {merged_colors.shape}")

    # Test voxelization on merged Gaussians
    print(f"\n" + "=" * 80)
    print("Testing voxelization on merged Gaussians...")
    print("=" * 80)

    # Prepare point cloud data: concatenate means with colors (use SH DC component if available)
    # For voxelization, we'll use means (xyz) and the first 3 channels of colors (RGB equivalent)
    if merged_colors.shape[1] >= 3:
        # Use first 3 color channels (DC component of SH)
        pc_colors = merged_colors[:, :3]
    else:
        # Use all available color channels
        pc_colors = merged_colors

    # Create point cloud tensor [N, 3+d] where d is color dimension
    pc_data = torch.cat([merged_means, pc_colors], dim=1)

    # Warmup to avoid CUDA JIT compilation overhead
    print("\n  Warming up CUDA kernels...")
    for _ in range(3):
        voxelize_pc_batched(pc_data, J=8, device='cuda')
        voxelize_pc_batched(pc_data, J=10, device='cuda')

    # Test with different octree depths
    for J in [8, 10]:
        print(f"\n  Testing with octree depth J={J}:")

        torch.cuda.synchronize()
        voxel_start_time = time.time()

        PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc_batched(
            pc_data,
            J=J,
            device='cuda'
        )

        torch.cuda.synchronize()
        voxel_elapsed_time = time.time() - voxel_start_time

        print(f"    Voxelization time: {voxel_elapsed_time*1000:.2f} ms")
        print(f"    Input points: {voxel_info['N']}")
        print(f"    Output voxels: {voxel_info['Nvox']}")
        print(f"    Voxel size: {voxel_info['voxel_size']:.6f}")
        print(f"    Compression ratio: {voxel_info['N'] / voxel_info['Nvox']:.2f}x")
        print(f"    PCvox shape: {PCvox.shape}")
        print(f"      ↳ [Nvox, 6] = [{voxel_info['Nvox']}, 3 xyz coords + 3 RGB colors]")
        print(f"      ↳ One row per unique voxel with averaged attributes")
        print(f"    DeltaPC shape: {DeltaPC.shape}")
        print(f"      ↳ [N, 6] = [{voxel_info['N']}, 3 xyz deltas + 3 RGB deltas]")
        print(f"      ↳ Quantization error for each original point")

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

    return {
        'original_count': N,
        'merged_count': merged_means.shape[0],
        'compression_ratio': N / merged_means.shape[0],
        'time_ms': elapsed_time * 1000,
        'merged_means': merged_means,
        'merged_quats': merged_quats,
        'merged_scales': merged_scales,
        'merged_opacities': merged_opacities,
        'merged_colors': merged_colors,
    }


if __name__ == '__main__':
    ckpt_path = "/ssd1/rajrup/Project/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt"
    
    try:
        results = test_merge_cluster(
            ckpt_path,
            num_clusters=50000,
            weight_by_opacity=True
        )

        print("\n" + "=" * 80)
        print("Summary:")
        print("=" * 80)
        print(f"  Original Gaussians: {results['original_count']}")
        print(f"  Merged Clusters: {results['merged_count']}")
        print(f"  Compression: {results['compression_ratio']:.2f}x")
        print(f"  Execution time: {results['time_ms']:.2f} ms")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
