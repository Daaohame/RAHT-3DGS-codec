"""
Point Cloud Voxelization using PyTorch and CUDA

Converted from MATLAB voxelizePC.m
Original: CONFIDENTIAL (C) Mitsubishi Electric Research Labs (MERL) 2017 Eduardo Pavez

This module provides:
- get_morton_code(): Optimized Morton code computation for 3D coordinates
- voxelize_pc_batched(): Main optimized voxelization function
- voxelize_pc(): Dict-based interface for backward compatibility with legacy code
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict

# Optional: open3d for PLY file writing (only needed if writeFileOut=True)
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def get_morton_code(V: torch.Tensor, J: int) -> torch.Tensor:
    """
    Compute Morton codes (Z-order curve) for integer coordinates.

    More efficient implementation using bitwise operations.
    Based on RAHT_param.py implementation.

    Args:
        V: Tensor of shape (N, 3) with integer coordinates (x, y, z)
        J: Octree depth (number of bits per dimension)

    Returns:
        M: Tensor of shape (N,) with Morton codes
    """
    N = V.shape[0]
    device = V.device

    # Initialize Morton codes
    MC = torch.zeros(N, dtype=torch.int64, device=device)

    # Interleave bits from least significant to most significant
    for i in range(1, J + 1):
        # Extract (i-1)-th bit from each coordinate
        b = (V >> (i - 1)) & 1  # (N, 3) {0, 1}

        # Interleave bits: z (LSB), y (middle), x (MSB)
        # digit = z + 2*y + 4*x
        digit = (b[:, 2].to(torch.int64)
                 + (b[:, 1].to(torch.int64) << 1)
                 + (b[:, 0].to(torch.int64) << 2))

        # Set the 3-bit digit at position (i-1) in the Morton code
        MC |= (digit << (3 * (i - 1)))

    return MC


def voxelize_pc_batched(
    PC: torch.Tensor,
    vmin: Optional[torch.Tensor] = None,
    width: Optional[float] = None,
    J: int = 10,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Optimized batched version of voxelize_pc using scatter operations.
    Better for large point clouds on GPU.

    Returns same outputs as voxelize_pc, plus a dict with additional info:
        - Nvox: number of voxels
        - voxel_size: size of each voxel
        - vmin: minimum coordinates used
        - width: bounding box width used
    """
    PC = PC.to(device)

    N = PC.shape[0]
    has_attribute = PC.shape[1] > 3

    V = PC[:, 0:3]
    C = PC[:, 3:] if has_attribute else None

    if vmin is None:
        vmin = V.min(dim=0)[0]
    else:
        vmin = vmin.to(device)

    V0 = V - vmin.unsqueeze(0)

    if width is None:
        width = V0.max().item()

    voxel_size = width / (2 ** J)
    V0_integer = torch.floor(V0 / voxel_size).long()

    M = get_morton_code(V0_integer, J)
    M_sort, idx = torch.sort(M)

    V0 = V0[idx]
    PCsorted = V[idx]

    if has_attribute:
        C0 = C[idx]
        PCsorted = torch.cat([PCsorted, C0], dim=1)

    V0_voxelized = voxel_size * torch.floor(V0 / voxel_size)
    DeltaV = V0 - V0_voxelized

    # Find unique voxels more efficiently
    voxel_boundary = M_sort[1:] - M_sort[:-1]
    voxel_indices = torch.cat([
        torch.tensor([0], device=device),
        torch.where(voxel_boundary != 0)[0] + 1
    ])

    Nvox = len(voxel_indices)

    if has_attribute:
        # Use scatter_add for efficient averaging
        # Assign each point to its voxel - VECTORIZED (no Python loop!)
        # Compute points per voxel from voxel_indices differences
        voxel_counts_int = torch.diff(
            torch.cat([voxel_indices, torch.tensor([N], device=device)])
        )

        # Create voxel_id using repeat_interleave (fully GPU-parallelized)
        voxel_id = torch.repeat_interleave(
            torch.arange(Nvox, device=device),
            voxel_counts_int
        )

        # Convert counts to float for averaging
        voxel_counts = voxel_counts_int.float()

        # Sum attributes per voxel
        C_sum = torch.zeros(Nvox, C0.shape[1], device=device)
        C_sum.scatter_add_(0, voxel_id.unsqueeze(1).expand(-1, C0.shape[1]), C0)

        # Average attributes
        Cvox = C_sum / voxel_counts.unsqueeze(1)

        # Broadcast averaged attributes back to all points
        C0_voxelized = Cvox[voxel_id]
        DeltaC = C0 - C0_voxelized

        Vvox = V0_integer[voxel_indices]
        Cvox_output = Cvox

        PCvox = torch.cat([Vvox.float(), Cvox_output], dim=1)
        DeltaPC = torch.cat([DeltaV, DeltaC], dim=1)
    else:
        Vvox = V0_integer[voxel_indices]
        PCvox = Vvox.float()
        DeltaPC = DeltaV

    info = {
        'Nvox': Nvox,
        'voxel_size': voxel_size,
        'vmin': vmin,
        'width': width,
        'N': N,
        'sort_idx': idx  # Sorting indices from Morton code ordering
    }

    return PCvox, PCsorted, voxel_indices, DeltaPC, info


def voxelize_pc(PC: torch.Tensor, param: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dict-based interface for backward compatibility with legacy code.

    This function wraps voxelize_pc_batched() and adds optional file writing.

    Args:
        PC (torch.Tensor): An [N, 3+d] tensor where the first 3 columns are XYZ
                           coordinates and the remaining d columns are attributes (e.g., colors).
        param (dict): A dictionary containing voxelization parameters:
            'vmin' (list, tuple, or torch.Tensor, optional): The minimum coordinate [x,y,z].
            'width' (float, optional): The side length of the cubic bounding box.
            'J' (int): The maximum depth of the octree decomposition.
            'writeFileOut' (bool, optional): Flag to control whether output files are written.
            'filename' (str, optional): The base filename for output files (required if writeFileOut=True).

    Returns:
        tuple: (PCvox, PCsorted, voxel_indices, DeltaPC)
            - PCvox: Voxelized point cloud
            - PCsorted: Original PC sorted by Morton code
            - voxel_indices: Indices marking start of each voxel
            - DeltaPC: Quantization error
    """
    # Extract parameters
    vmin = param.get('vmin')
    width = param.get('width')
    J = param.get('J')
    writeFileOut = param.get('writeFileOut', False)
    filename = param.get('filename')
    device = PC.device

    # Convert vmin to tensor if provided
    if vmin is not None and not isinstance(vmin, torch.Tensor):
        vmin = torch.tensor(vmin, dtype=PC.dtype, device=device)

    # Call optimized backend
    PCvox, PCsorted, voxel_indices, DeltaPC, info = voxelize_pc_batched(
        PC=PC,
        vmin=vmin,
        width=width,
        J=J,
        device=str(device)
    )

    # Extract info for file writing
    Nvox = info['Nvox']
    voxel_size = info['voxel_size']
    N = info['N']
    vmin_used = info['vmin']
    width_used = info['width']
    hasAttribute = PC.shape[1] > 3

    # Optional file output
    if writeFileOut:
        if not HAS_OPEN3D:
            raise ImportError("open3d is required for file writing. Install it with: pip install open3d")

        if filename is None:
            raise ValueError("A filename must be provided in 'param' when writeFileOut is True.")

        # Extract voxelized coordinates for PLY output
        if hasAttribute:
            Vvox_integer = PCvox[:, :3]
            Cvox = PCvox[:, 3:]
        else:
            Vvox_integer = PCvox
            Cvox = None

        # 1. Write the voxelized point cloud to a .ply file
        # Convert integer voxel indices to world coordinates (voxel centers)
        # Use voxel center: (index + 0.5) * voxel_size + vmin
        Vvox_coords = ((Vvox_integer.float() + 0.5) * voxel_size) + vmin_used
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(Vvox_coords.cpu().numpy())

        if hasAttribute:
            Cvox_for_ply = Cvox.cpu().numpy()
            if torch.max(Cvox) > 1.0:
                 Cvox_for_ply = Cvox_for_ply / 255.0
            pc.colors = o3d.utility.Vector3dVector(Cvox_for_ply)

        filename_pcvox = f"{filename}_vox.ply"
        o3d.io.write_point_cloud(filename_pcvox, pc)
        print(f"Voxelized point cloud saved to {filename_pcvox}")

        # 1b. Also write the sorted original point cloud (full detail, Morton order)
        if hasAttribute:
            Vsorted = PCsorted[:, :3]
            Csorted = PCsorted[:, 3:]
            pc_sorted = o3d.geometry.PointCloud()
            pc_sorted.points = o3d.utility.Vector3dVector(Vsorted.cpu().numpy())
            Csorted_for_ply = Csorted.cpu().numpy()
            if torch.max(Csorted) > 1.0:
                Csorted_for_ply = Csorted_for_ply / 255.0
            pc_sorted.colors = o3d.utility.Vector3dVector(Csorted_for_ply)
            filename_sorted = f"{filename}_sorted.ply"
            o3d.io.write_point_cloud(filename_sorted, pc_sorted)
            print(f"Sorted original point cloud saved to {filename_sorted}")

        # 2. Write the metadata and deltas to a .txt file
        filename_data = f"{filename}_data.txt"
        with open(filename_data, 'w') as f:
            header = f"{vmin_used[0]} {vmin_used[1]} {vmin_used[2]} {width_used} {J} {Nvox} {N} {int(hasAttribute)}\n"
            f.write(header)
            np.savetxt(f, voxel_indices.cpu().numpy(), fmt='%d')
            np.savetxt(f, DeltaPC.cpu().numpy(), fmt='%.6f')
        print(f"Voxelization data saved to {filename_data}")

    return PCvox, PCsorted, voxel_indices, DeltaPC


# Example usage
if __name__ == "__main__":
    # Create sample point cloud
    N = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Random points in [0, 1]^3 with RGB colors
    points = torch.rand(N, 3, device=device)
    colors = torch.rand(N, 3, device=device) * 255

    PC = torch.cat([points, colors], dim=1)

    print(f"Input point cloud: {PC.shape}")
    print(f"Device: {device}")

    # Voxelize with octree depth J=8
    J = 8
    PCvox, PCsorted, voxel_indices, DeltaPC, info = voxelize_pc_batched(
        PC, J=J, device=device
    )

    print(f"\nVoxelization results:")
    print(f"  Voxel size: {info['voxel_size']:.6f}")
    print(f"  Number of voxels: {info['Nvox']}")
    print(f"  Number of points: {info['N']}")
    print(f"  Voxelized PC shape: {PCvox.shape}")
    print(f"  Sorted PC shape: {PCsorted.shape}")
    print(f"  Voxel indices shape: {voxel_indices.shape}")
    print(f"  Delta PC shape: {DeltaPC.shape}")

    # Verify that voxel indices are correct
    print(f"\nFirst 10 voxel indices: {voxel_indices[:10].cpu().numpy()}")
    print(f"Compression ratio: {N / info['Nvox']:.2f}x")
