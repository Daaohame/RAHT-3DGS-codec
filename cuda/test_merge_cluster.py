"""
Test script for merge_cluster CUDA extension.

This script tests the functionality and correctness of the cluster merging implementation.
"""

import torch
import time


def test_basic_functionality():
    """Test basic functionality of cluster merging."""
    print("=" * 70)
    print("Test 1: Basic Functionality")
    print("=" * 70)

    # Use JIT compilation for testing
    try:
        from build_merge_cluster import merge_gaussian_clusters
    except Exception as e:
        print(f"Error importing: {e}")
        print("Trying direct import...")
        from merge_cluster import merge_gaussian_clusters

    # Create test data
    N = 1000
    num_clusters = 100

    print(f"Creating {N} test Gaussians...")
    means = torch.randn(N, 3, device='cuda', dtype=torch.float32)
    quats = torch.randn(N, 4, device='cuda', dtype=torch.float32)
    quats = quats / quats.norm(dim=1, keepdim=True)  # Normalize
    scales = torch.rand(N, 3, device='cuda', dtype=torch.float32)
    opacities = torch.rand(N, device='cuda', dtype=torch.float32)
    colors = torch.randn(N, 48, device='cuda', dtype=torch.float32)  # 16 SH coeffs * 3

    # Create cluster labels
    cluster_labels = torch.randint(0, num_clusters, (N,), device='cuda', dtype=torch.long)

    print(f"Cluster labels: min={cluster_labels.min()}, max={cluster_labels.max()}")
    print(f"Number of unique clusters: {torch.unique(cluster_labels).numel()}")

    # Merge clusters
    print("\nMerging clusters...")
    start = time.time()
    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = \
        merge_gaussian_clusters(
            means, quats, scales, opacities, colors, cluster_labels,
            weight_by_opacity=True
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Merging took {elapsed * 1000:.2f} ms")
    print(f"\nOutput shapes:")
    print(f"  merged_means: {merged_means.shape}")
    print(f"  merged_quats: {merged_quats.shape}")
    print(f"  merged_scales: {merged_scales.shape}")
    print(f"  merged_opacities: {merged_opacities.shape}")
    print(f"  merged_colors: {merged_colors.shape}")

    # Verify outputs
    assert merged_means.shape[0] == torch.unique(cluster_labels).numel()
    assert merged_quats.shape[0] == torch.unique(cluster_labels).numel()
    assert merged_scales.shape[0] == torch.unique(cluster_labels).numel()
    assert merged_opacities.shape[0] == torch.unique(cluster_labels).numel()
    assert merged_colors.shape[0] == torch.unique(cluster_labels).numel()

    # Check quaternion normalization
    quat_norms = merged_quats.norm(dim=1)
    print(f"\nQuaternion norms: min={quat_norms.min():.6f}, max={quat_norms.max():.6f}")
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5), \
        "Quaternions should be normalized"

    # Check opacity clamping
    assert merged_opacities.min() >= 0.0, "Opacities should be >= 0"
    assert merged_opacities.max() <= 1.0, "Opacities should be <= 1"

    print("\n✓ Test 1 passed!")


def test_correctness():
    """Test correctness by comparing with CPU reference implementation."""
    print("\n" + "=" * 70)
    print("Test 2: Correctness (vs CPU reference)")
    print("=" * 70)

    try:
        from build_merge_cluster import merge_gaussian_clusters
    except:
        from merge_cluster import merge_gaussian_clusters

    # Create simple test case
    N = 20
    num_clusters = 4

    torch.manual_seed(42)
    means = torch.randn(N, 3, device='cuda')
    quats = torch.randn(N, 4, device='cuda')
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.rand(N, 3, device='cuda')
    opacities = torch.rand(N, device='cuda')
    colors = torch.randn(N, 3, device='cuda')  # Simple RGB

    # Create controlled cluster labels
    cluster_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                   0, 0, 1, 1, 2, 2, 3, 3], device='cuda')

    # Merge with CUDA
    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = \
        merge_gaussian_clusters(
            means, quats, scales, opacities, colors, cluster_labels,
            weight_by_opacity=False  # Use equal weights for simplicity
        )

    # CPU reference implementation
    def cpu_merge(cluster_id):
        mask = cluster_labels == cluster_id
        indices = torch.where(mask)[0]

        if len(indices) == 0:
            return None

        # Simple average (weight_by_opacity=False)
        merged_mean = means[indices].mean(dim=0)
        merged_quat = quats[indices].mean(dim=0)
        merged_quat = merged_quat / merged_quat.norm()  # Normalize
        merged_scale = scales[indices].mean(dim=0)
        merged_opacity = opacities[indices].sum().clamp(max=1.0)
        merged_color = colors[indices].mean(dim=0)

        return merged_mean, merged_quat, merged_scale, merged_opacity, merged_color

    # Compare with CPU reference
    print("Comparing cluster 0...")
    ref = cpu_merge(0)
    if ref is not None:
        ref_mean, ref_quat, ref_scale, ref_opacity, ref_color = ref
        cuda_idx = 0

        print(f"  Mean diff: {(merged_means[cuda_idx].cpu() - ref_mean).abs().max():.6f}")
        print(f"  Quat diff: {(merged_quats[cuda_idx].cpu() - ref_quat).abs().max():.6f}")
        print(f"  Scale diff: {(merged_scales[cuda_idx].cpu() - ref_scale).abs().max():.6f}")
        print(f"  Opacity diff: {(merged_opacities[cuda_idx].cpu() - ref_opacity).abs():.6f}")
        print(f"  Color diff: {(merged_colors[cuda_idx].cpu() - ref_color).abs().max():.6f}")

        assert torch.allclose(merged_means[cuda_idx].cpu(), ref_mean, atol=1e-5)
        assert torch.allclose(merged_quats[cuda_idx].cpu(), ref_quat, atol=1e-4)

    print("\n✓ Test 2 passed!")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 70)
    print("Test 3: Edge Cases")
    print("=" * 70)

    try:
        from build_merge_cluster import merge_gaussian_clusters
    except:
        from merge_cluster import merge_gaussian_clusters

    # Test 1: Single Gaussian per cluster
    print("Testing single Gaussian per cluster...")
    N = 10
    means = torch.randn(N, 3, device='cuda')
    quats = torch.randn(N, 4, device='cuda')
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.rand(N, 3, device='cuda')
    opacities = torch.rand(N, device='cuda')
    colors = torch.randn(N, 3, device='cuda')
    cluster_labels = torch.arange(N, device='cuda')  # Each in its own cluster

    merged = merge_gaussian_clusters(
        means, quats, scales, opacities, colors, cluster_labels
    )

    # Should be identical (or very close due to floating point)
    assert torch.allclose(merged[0], means, atol=1e-5), "Single-element clusters should preserve means"
    assert torch.allclose(merged[1], quats, atol=1e-5), "Single-element clusters should preserve quats"
    print("✓ Single Gaussian per cluster test passed")

    # Test 2: All Gaussians in one cluster
    print("\nTesting all Gaussians in one cluster...")
    cluster_labels = torch.zeros(N, device='cuda', dtype=torch.long)
    merged = merge_gaussian_clusters(
        means, quats, scales, opacities, colors, cluster_labels
    )

    assert merged[0].shape[0] == 1, "Should have exactly 1 cluster"
    assert merged[0].shape == (1, 3), "Merged means should be [1, 3]"
    print("✓ All Gaussians in one cluster test passed")

    print("\n✓ Test 3 passed!")


def test_performance():
    """Test performance with larger datasets."""
    print("\n" + "=" * 70)
    print("Test 4: Performance")
    print("=" * 70)

    try:
        from build_merge_cluster import merge_gaussian_clusters
    except:
        from merge_cluster import merge_gaussian_clusters

    sizes = [1000, 10000, 100000]

    for N in sizes:
        num_clusters = N // 10

        print(f"\nTesting with {N} Gaussians, {num_clusters} clusters...")
        means = torch.randn(N, 3, device='cuda')
        quats = torch.randn(N, 4, device='cuda')
        quats = quats / quats.norm(dim=1, keepdim=True)
        scales = torch.rand(N, 3, device='cuda')
        opacities = torch.rand(N, device='cuda')
        colors = torch.randn(N, 48, device='cuda')
        cluster_labels = torch.randint(0, num_clusters, (N,), device='cuda')

        # Warmup
        _ = merge_gaussian_clusters(
            means, quats, scales, opacities, colors, cluster_labels
        )

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        num_iters = 10

        for _ in range(num_iters):
            merged = merge_gaussian_clusters(
                means, quats, scales, opacities, colors, cluster_labels
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_time = (elapsed / num_iters) * 1000  # ms
        print(f"  Average time: {avg_time:.2f} ms ({N / avg_time:.0f} Gaussians/ms)")

    print("\n✓ Test 4 passed!")


if __name__ == '__main__':
    print("Testing merge_cluster CUDA extension\n")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}\n")

    try:
        test_basic_functionality()
        test_correctness()
        test_edge_cases()
        test_performance()

        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n\nTest failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
