"""
Quality evaluation utilities for 3D Gaussian Splatting.

Provides functions to:
- Export 3DGS to PLY format
- Compute attribute quality metrics (MSE, RMSE, etc.)
- Map merged Gaussians back to original indices for comparison
"""

import torch
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


def save_ply(
    filepath: str,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor
) -> None:
    """
    Save 3D Gaussians to a PLY file.

    Args:
        filepath: Output .ply file path
        means: [N, 3] Gaussian centers
        quats: [N, 4] Quaternions (normalized)
        scales: [N, 3] Scales
        opacities: [N] Opacities
        colors: [N, C] Colors (spherical harmonics coefficients)
    """
    N = means.shape[0]
    color_dim = colors.shape[1]

    # Convert to numpy and move to CPU
    means_np = means.detach().cpu().float().numpy()
    quats_np = quats.detach().cpu().float().numpy()
    scales_np = scales.detach().cpu().float().numpy()
    opacities_np = opacities.detach().cpu().float().numpy()
    colors_np = colors.detach().cpu().float().numpy()

    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Write PLY file
    with open(filepath, 'wb') as f:
        # Write header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())

        # Position properties
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")

        # Normal (we'll use zeros as placeholder)
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")

        # Spherical harmonics (colors)
        # First 3 channels are DC (RGB equivalent)
        for i in range(color_dim):
            f.write(f"property float f_dc_{i}\n".encode())

        # Opacity
        f.write(b"property float opacity\n")

        # Scale
        f.write(b"property float scale_0\n")
        f.write(b"property float scale_1\n")
        f.write(b"property float scale_2\n")

        # Rotation (quaternion)
        f.write(b"property float rot_0\n")
        f.write(b"property float rot_1\n")
        f.write(b"property float rot_2\n")
        f.write(b"property float rot_3\n")

        f.write(b"end_header\n")

        # Write data
        normals = np.zeros((N, 3), dtype=np.float32)

        for i in range(N):
            # Position
            f.write(means_np[i].astype(np.float32).tobytes())
            # Normal (placeholder)
            f.write(normals[i].astype(np.float32).tobytes())
            # Colors (SH coefficients)
            f.write(colors_np[i].astype(np.float32).tobytes())
            # Opacity
            f.write(opacities_np[i:i+1].astype(np.float32).tobytes())
            # Scale
            f.write(scales_np[i].astype(np.float32).tobytes())
            # Rotation (quaternion)
            f.write(quats_np[i].astype(np.float32).tobytes())

    print(f"Saved {N} Gaussians to {filepath}")


def compute_attribute_metrics(
    original_means: torch.Tensor,
    original_quats: torch.Tensor,
    original_scales: torch.Tensor,
    original_opacities: torch.Tensor,
    original_colors: torch.Tensor,
    merged_means: torch.Tensor,
    merged_quats: torch.Tensor,
    merged_scales: torch.Tensor,
    merged_opacities: torch.Tensor,
    merged_colors: torch.Tensor,
    cluster_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute quality metrics between original and merged Gaussians.

    For each original Gaussian, compare it to its merged cluster representative.

    Args:
        original_*: Original Gaussian attributes [N, ...]
        merged_*: Merged Gaussian attributes [K, ...]
        cluster_labels: [N] mapping from original Gaussians to cluster IDs

    Returns:
        Dictionary of metrics
    """
    N = original_means.shape[0]

    # Map merged attributes back to original indices
    reconstructed_means = merged_means[cluster_labels]
    reconstructed_quats = merged_quats[cluster_labels]
    reconstructed_scales = merged_scales[cluster_labels]
    reconstructed_opacities = merged_opacities[cluster_labels]
    reconstructed_colors = merged_colors[cluster_labels]

    # Compute MSE and RMSE for each attribute
    metrics = {}

    # Position error
    pos_mse = torch.mean((original_means - reconstructed_means) ** 2).item()
    pos_rmse = np.sqrt(pos_mse)
    metrics['position_mse'] = pos_mse
    metrics['position_rmse'] = pos_rmse

    # Quaternion error (geodesic distance on unit sphere)
    # d(q1, q2) = 1 - |<q1, q2>|^2
    quat_dot = torch.abs(torch.sum(original_quats * reconstructed_quats, dim=1))
    quat_dist = 1.0 - quat_dot ** 2
    metrics['quaternion_mean_dist'] = torch.mean(quat_dist).item()
    metrics['quaternion_max_dist'] = torch.max(quat_dist).item()

    # Scale error (in log space since scales are typically in log domain)
    scale_log_original = torch.log(original_scales + 1e-8)
    scale_log_reconstructed = torch.log(reconstructed_scales + 1e-8)
    scale_mse = torch.mean((scale_log_original - scale_log_reconstructed) ** 2).item()
    scale_rmse = np.sqrt(scale_mse)
    metrics['scale_log_mse'] = scale_mse
    metrics['scale_log_rmse'] = scale_rmse

    # Opacity error
    opacity_mse = torch.mean((original_opacities - reconstructed_opacities) ** 2).item()
    opacity_rmse = np.sqrt(opacity_mse)
    metrics['opacity_mse'] = opacity_mse
    metrics['opacity_rmse'] = opacity_rmse

    # Color error (MSE over all channels)
    color_mse = torch.mean((original_colors - reconstructed_colors) ** 2).item()
    color_rmse = np.sqrt(color_mse)
    metrics['color_mse'] = color_mse
    metrics['color_rmse'] = color_rmse

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Quality Metrics") -> None:
    """Pretty print quality metrics."""
    print(f"\n{'=' * 80}")
    print(title)
    print('=' * 80)

    print("\nPosition:")
    print(f"  MSE:  {metrics['position_mse']:.6e}")
    print(f"  RMSE: {metrics['position_rmse']:.6e}")

    print("\nQuaternion (rotation):")
    print(f"  Mean distance: {metrics['quaternion_mean_dist']:.6e}")
    print(f"  Max distance:  {metrics['quaternion_max_dist']:.6e}")

    print("\nScale (log space):")
    print(f"  MSE:  {metrics['scale_log_mse']:.6e}")
    print(f"  RMSE: {metrics['scale_log_rmse']:.6e}")

    print("\nOpacity:")
    print(f"  MSE:  {metrics['opacity_mse']:.6e}")
    print(f"  RMSE: {metrics['opacity_rmse']:.6e}")

    print("\nColor (SH coefficients):")
    print(f"  MSE:  {metrics['color_mse']:.6e}")
    print(f"  RMSE: {metrics['color_rmse']:.6e}")


def try_render_comparison(
    original_params: Dict[str, torch.Tensor],
    merged_params: Dict[str, torch.Tensor],
    camera_params: Dict[str, torch.Tensor] = None
) -> Dict[str, float]:
    """
    Attempt to render both original and merged Gaussians and compute image metrics.

    Requires gsplat library to be installed.

    Args:
        original_params: Dict with keys: means, quats, scales, opacities, colors
        merged_params: Dict with keys: means, quats, scales, opacities, colors
        camera_params: Optional camera parameters (if None, skips rendering)

    Returns:
        Dictionary with PSNR, SSIM, LPIPS metrics (or empty dict if rendering fails)
    """
    try:
        import gsplat
        print("\ngsplat found! Rendering comparison...")

        # TODO: Implement rendering logic
        # This requires camera parameters and a rendering loop
        # For now, return empty dict
        print("  (Rendering not implemented yet - requires camera parameters)")
        return {}

    except ImportError:
        print("\ngsplat not available - skipping rendering comparison")
        return {}
