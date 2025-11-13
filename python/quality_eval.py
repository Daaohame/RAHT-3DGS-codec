"""
Quality evaluation utilities for 3D Gaussian Splatting.

Provides functions to:
- Export 3DGS to PLY format
- Compute attribute quality metrics (MSE, RMSE, etc.)
- Map merged Gaussians back to original indices for comparison
- Render Gaussians and compute PSNR
"""

import torch
import numpy as np
import time
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

    print("\nQuaternion (rotation):")
    print(f"  Mean distance: {metrics['quaternion_mean_dist']:.6e}")
    print(f"  Max distance:  {metrics['quaternion_max_dist']:.6e}")


def generate_random_cameras(
    center: torch.Tensor,
    radius: float,
    n_views: int = 5,
    image_width: int = 512,
    image_height: int = 512,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Generate random camera poses looking at a scene center.

    Args:
        center: [3] Scene center point
        radius: Distance from center to camera
        n_views: Number of camera views to generate
        image_width: Image width in pixels
        image_height: Image height in pixels
        device: Device to use

    Returns:
        viewmats: [n_views, 4, 4] World-to-camera matrices
        Ks: [n_views, 3, 3] Intrinsic matrices
        width: Image width
        height: Image height
    """
    import math

    # Generate random camera positions on a sphere
    viewmats = []
    for i in range(n_views):
        # Random spherical coordinates
        theta = torch.rand(1).item() * 2 * math.pi  # azimuth
        phi = (torch.rand(1).item() * 0.5 + 0.25) * math.pi  # elevation (avoid poles)

        # Convert to Cartesian
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)

        cam_pos = center + torch.tensor([x, y, z], device=device)

        # Look-at matrix construction (standard computer graphics)
        forward = center - cam_pos
        forward = forward / torch.norm(forward)

        # Up vector
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = torch.linalg.cross(world_up, forward)
        if torch.norm(right) < 0.001:  # Handle degenerate case
            world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
            right = torch.linalg.cross(world_up, forward)
        right = right / torch.norm(right)
        up = torch.linalg.cross(forward, right)

        # Build view matrix (world-to-camera)
        # gsplat convention: camera looks down +Z in camera space
        w2c = torch.eye(4, device=device)
        w2c[0, :3] = right
        w2c[1, :3] = up
        w2c[2, :3] = forward  # Camera looks down +Z in camera space
        w2c[:3, 3] = -torch.mv(w2c[:3, :3], cam_pos)

        viewmats.append(w2c)

    viewmats = torch.stack(viewmats)

    # Create intrinsic matrix (simple pinhole model)
    focal = image_width * 1.2  # Reasonable FOV
    K = torch.tensor([
        [focal, 0, image_width / 2],
        [0, focal, image_height / 2],
        [0, 0, 1]
    ], device=device)
    Ks = K.unsqueeze(0).repeat(n_views, 1, 1)

    return viewmats, Ks, image_width, image_height


def render_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int
) -> torch.Tensor:
    """
    Render Gaussians from multiple viewpoints using gsplat.

    Args:
        means: [N, 3] Gaussian centers
        quats: [N, 4] Quaternions
        scales: [N, 3] Scales
        opacities: [N] Opacities
        colors: [N, C] Colors (SH coefficients)
        viewmats: [V, 4, 4] View matrices
        Ks: [V, 3, 3] Intrinsic matrices
        width: Image width
        height: Image height

    Returns:
        images: [V, H, W, 3] Rendered RGB images
    """
    import gsplat

    n_views = viewmats.shape[0]
    device = means.device

    # Scales should already be in linear space after exponentiation in test_voxelize_3dgs.py

    # Reshape colors to [N, K, 3] format expected by gsplat for SH coefficients
    # colors is [N, 48] = [N, 16 * 3] for 16 SH coefficients
    # Reshape to [N, 16, 3]
    if colors.shape[1] % 3 == 0:
        K = colors.shape[1] // 3
        colors_reshaped = colors.reshape(-1, K, 3)
        sh_degree = int(np.sqrt(K) - 1)  # Degree from number of coefficients
    else:
        # Fallback: use first 3 as RGB
        colors_reshaped = colors[:, :3].unsqueeze(1)  # [N, 1, 3]
        sh_degree = None

    images = []

    for i in range(n_views):
        # Render using gsplat with white background
        backgrounds = torch.ones((1, 3), device=device)

        renders, alphas, info = gsplat.rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),  # Ensure normalized
            scales=scales,
            opacities=opacities.squeeze(),
            colors=colors_reshaped,
            viewmats=viewmats[i:i+1],
            Ks=Ks[i:i+1],
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=False,
            backgrounds=backgrounds,
        )

        images.append(renders[0])  # [H, W, 3]

    return torch.stack(images)  # [V, H, W, 3]


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute PSNR between two images.

    Args:
        img1, img2: Images of shape [..., H, W, 3] in range [0, 1]

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def try_render_comparison(
    original_params: Dict[str, torch.Tensor],
    merged_params: Dict[str, torch.Tensor],
    n_views: int = 5,
    image_size: int = 512,
    output_dir: str = None
) -> Dict[str, float]:
    """
    Render both original and merged Gaussians from random views and compute PSNR.

    Args:
        original_params: Dict with keys: means, quats, scales, opacities, colors
        merged_params: Dict with keys: means, quats, scales, opacities, colors
        n_views: Number of random camera views to render
        image_size: Image resolution (square images)
        output_dir: Optional directory to save rendered images

    Returns:
        Dictionary with PSNR metrics per view and average
    """
    try:
        import gsplat
        print(f"\n{'=' * 80}")
        print(f"Rendering comparison with {n_views} random camera views...")
        print('=' * 80)

        device = original_params['means'].device

        # Compute scene center and bounds from original Gaussians
        center = original_params['means'].mean(dim=0)
        bbox_size = (original_params['means'].max(dim=0)[0] -
                     original_params['means'].min(dim=0)[0]).max().item()
        radius = bbox_size * 1.5  # Camera distance from center

        print(f"  Scene center: {center.cpu().numpy()}")
        print(f"  Scene size: {bbox_size:.4f}")
        print(f"  Camera radius: {radius:.4f}")

        # Generate random camera views
        viewmats, Ks, width, height = generate_random_cameras(
            center=center,
            radius=radius,
            n_views=n_views,
            image_width=image_size,
            image_height=image_size,
            device=device
        )

        print(f"\n  Rendering original Gaussians ({original_params['means'].shape[0]} points)...")
        torch.cuda.synchronize()
        start_time = time.time()

        original_images = render_gaussians(
            means=original_params['means'],
            quats=original_params['quats'],
            scales=original_params['scales'],
            opacities=original_params['opacities'],
            colors=original_params['colors'],
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height
        )

        torch.cuda.synchronize()
        original_time = time.time() - start_time
        print(f"    Time: {original_time*1000:.2f} ms ({original_time*1000/n_views:.2f} ms/view)")

        print(f"\n  Rendering merged Gaussians ({merged_params['means'].shape[0]} points)...")
        torch.cuda.synchronize()
        start_time = time.time()

        merged_images = render_gaussians(
            means=merged_params['means'],
            quats=merged_params['quats'],
            scales=merged_params['scales'],
            opacities=merged_params['opacities'],
            colors=merged_params['colors'],
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height
        )

        torch.cuda.synchronize()
        merged_time = time.time() - start_time
        print(f"    Time: {merged_time*1000:.2f} ms ({merged_time*1000/n_views:.2f} ms/view)")

        # Compute PSNR for each view
        print(f"\n  Computing PSNR metrics...")
        print(f"  Image statistics:")
        print(f"    Original images: min={original_images.min().item():.4f}, max={original_images.max().item():.4f}, mean={original_images.mean().item():.4f}")
        print(f"    Merged images: min={merged_images.min().item():.4f}, max={merged_images.max().item():.4f}, mean={merged_images.mean().item():.4f}")

        psnrs = []
        for i in range(n_views):
            mse = torch.mean((original_images[i] - merged_images[i]) ** 2).item()
            psnr = compute_psnr(original_images[i], merged_images[i])
            psnrs.append(psnr)
            print(f"    View {i+1}: MSE={mse:.6e}, PSNR={psnr:.2f} dB")

        avg_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        min_psnr = np.min(psnrs)
        max_psnr = np.max(psnrs)

        print(f"\n  Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"  Range: [{min_psnr:.2f}, {max_psnr:.2f}] dB")

        # Save rendered images if output directory is specified
        if output_dir is not None:
            import os
            from PIL import Image

            os.makedirs(output_dir, exist_ok=True)
            print(f"\n  Saving rendered images to {output_dir}/...")

            for i in range(n_views):
                # Convert to uint8 [0, 255]
                original_img = (original_images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                merged_img = (merged_images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # Save original
                Image.fromarray(original_img).save(os.path.join(output_dir, f"view_{i:03d}_original.png"))

                # Save merged
                Image.fromarray(merged_img).save(os.path.join(output_dir, f"view_{i:03d}_merged.png"))

                # Save side-by-side comparison
                comparison = np.concatenate([original_img, merged_img], axis=1)
                Image.fromarray(comparison).save(os.path.join(output_dir, f"view_{i:03d}_comparison.png"))

            print(f"    Saved {n_views} views (original, merged, and comparison)")

        metrics = {
            'psnr_avg': avg_psnr,
            'psnr_std': std_psnr,
            'psnr_min': min_psnr,
            'psnr_max': max_psnr,
            'psnr_per_view': psnrs,
            'original_render_time_ms': original_time * 1000,
            'merged_render_time_ms': merged_time * 1000,
        }

        return metrics

    except ImportError:
        print("\ngsplat not available - skipping rendering comparison")
        return {}
    except Exception as e:
        print(f"\nError during rendering: {e}")
        import traceback
        traceback.print_exc()
        return {}
