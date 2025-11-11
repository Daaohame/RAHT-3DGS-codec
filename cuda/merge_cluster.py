"""
Python interface for merging 3D Gaussian clusters.

This module provides functionality to merge clusters of 3D Gaussians using a weighted
mean strategy. The merging is performed on GPU using CUDA kernels.
"""

import torch
from typing import Tuple
import warnings

# Try to import the compiled extension
try:
    import merge_cluster_cuda
    _extension_available = True
except ImportError:
    _extension_available = False
    warnings.warn(
        "merge_cluster_cuda extension not found. Please build it first using setup.py"
    )


def prepare_cluster_data(cluster_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert cluster labels to the format expected by the CUDA kernel.

    Args:
        cluster_labels: [N] tensor of cluster IDs for each Gaussian

    Returns:
        cluster_indices: [total_clustered] flat indices of Gaussians in each cluster
        cluster_offsets: [num_clusters + 1] boundaries marking cluster starts/ends
    """
    device = cluster_labels.device

    # Get unique cluster IDs and create a mapping to consecutive IDs
    unique_clusters, inverse_indices = torch.unique(cluster_labels, return_inverse=True)
    num_clusters = len(unique_clusters)

    # Sort indices by cluster ID
    sorted_indices = torch.argsort(inverse_indices)
    sorted_cluster_ids = inverse_indices[sorted_indices]

    # Find cluster boundaries
    # Add a sentinel at the beginning to make diff work
    boundaries = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int64),
        torch.where(sorted_cluster_ids[1:] != sorted_cluster_ids[:-1])[0] + 1,
        torch.tensor([len(sorted_indices)], device=device, dtype=torch.int64)
    ])

    cluster_offsets = boundaries.to(torch.int32)
    cluster_indices = sorted_indices.to(torch.int32)

    return cluster_indices, cluster_offsets


def merge_gaussian_clusters(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    cluster_labels: torch.Tensor,
    weight_by_opacity: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge 3D Gaussian clusters using weighted mean strategy.

    Args:
        means: [N, 3] tensor of Gaussian means (positions)
        quats: [N, 4] tensor of quaternions (rotations)
        scales: [N, 3] tensor of scales
        opacities: [N] tensor of opacities
        colors: [N, color_dim] tensor of colors (e.g., SH coefficients or RGB)
        cluster_labels: [N] tensor of cluster IDs for each Gaussian
        weight_by_opacity: If True, weight contributions by opacity; otherwise equal weights

    Returns:
        Tuple of (merged_means, merged_quats, merged_scales, merged_opacities, merged_colors)
        Each tensor has shape [num_clusters, ...] where num_clusters is the number of unique clusters

    Example:
        >>> means = torch.randn(1000, 3, device='cuda')
        >>> quats = torch.randn(1000, 4, device='cuda')
        >>> quats = quats / quats.norm(dim=1, keepdim=True)  # normalize
        >>> scales = torch.rand(1000, 3, device='cuda')
        >>> opacities = torch.rand(1000, device='cuda')
        >>> colors = torch.randn(1000, 48, device='cuda')  # e.g., 16 SH coefficients * 3 channels
        >>> cluster_labels = torch.randint(0, 100, (1000,), device='cuda')
        >>> merged = merge_gaussian_clusters(means, quats, scales, opacities, colors, cluster_labels)
    """
    if not _extension_available:
        raise RuntimeError(
            "merge_cluster_cuda extension is not available. "
            "Please build it first using: python setup.py install"
        )

    # Input validation
    assert means.is_cuda, "All inputs must be CUDA tensors"
    assert quats.is_cuda, "All inputs must be CUDA tensors"
    assert scales.is_cuda, "All inputs must be CUDA tensors"
    assert opacities.is_cuda, "All inputs must be CUDA tensors"
    assert colors.is_cuda, "All inputs must be CUDA tensors"
    assert cluster_labels.is_cuda, "All inputs must be CUDA tensors"

    N = means.shape[0]
    assert quats.shape[0] == N, "All inputs must have same number of Gaussians"
    assert scales.shape[0] == N, "All inputs must have same number of Gaussians"
    assert opacities.shape[0] == N, "All inputs must have same number of Gaussians"
    assert colors.shape[0] == N, "All inputs must have same number of Gaussians"
    assert cluster_labels.shape[0] == N, "cluster_labels must have same length as inputs"

    # Ensure correct dtypes
    means = means.contiguous().float()
    quats = quats.contiguous().float()
    scales = scales.contiguous().float()
    opacities = opacities.contiguous().float()
    colors = colors.contiguous().float()
    cluster_labels = cluster_labels.contiguous().long()

    # Prepare cluster data
    cluster_indices, cluster_offsets = prepare_cluster_data(cluster_labels)

    # Call CUDA kernel
    result = merge_cluster_cuda.merge_clusters_cuda(
        cluster_indices,
        cluster_offsets,
        means,
        quats,
        scales,
        opacities,
        colors,
        weight_by_opacity
    )

    return tuple(result)


def merge_gaussian_clusters_with_indices(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    cluster_indices: torch.Tensor,
    cluster_offsets: torch.Tensor,
    weight_by_opacity: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Low-level interface: merge clusters using pre-computed cluster_indices and cluster_offsets.

    This is useful if you already have the cluster data in the required format.
    Most users should use merge_gaussian_clusters() instead.

    Args:
        means: [N, 3] tensor of Gaussian means
        quats: [N, 4] tensor of quaternions
        scales: [N, 3] tensor of scales
        opacities: [N] tensor of opacities
        colors: [N, color_dim] tensor of colors
        cluster_indices: [total_clustered] flat indices of Gaussians in each cluster
        cluster_offsets: [num_clusters + 1] boundaries marking cluster starts/ends
        weight_by_opacity: If True, weight contributions by opacity

    Returns:
        Tuple of (merged_means, merged_quats, merged_scales, merged_opacities, merged_colors)
    """
    if not _extension_available:
        raise RuntimeError(
            "merge_cluster_cuda extension is not available. "
            "Please build it first using: python setup.py install"
        )

    # Ensure correct dtypes
    means = means.contiguous().float()
    quats = quats.contiguous().float()
    scales = scales.contiguous().float()
    opacities = opacities.contiguous().float()
    colors = colors.contiguous().float()
    cluster_indices = cluster_indices.contiguous().int()
    cluster_offsets = cluster_offsets.contiguous().int()

    # Call CUDA kernel
    result = merge_cluster_cuda.merge_clusters_cuda(
        cluster_indices,
        cluster_offsets,
        means,
        quats,
        scales,
        opacities,
        colors,
        weight_by_opacity
    )

    return tuple(result)
