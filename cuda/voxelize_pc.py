"""
Point Cloud Voxelization using PyTorch and CUDA

Converted from MATLAB voxelizePC.m
Original: CONFIDENTIAL (C) Mitsubishi Electric Research Labs (MERL) 2017 Eduardo Pavez
"""

import torch
from typing import Optional, Tuple, Dict


def get_morton_code(V: torch.Tensor, J: int) -> torch.Tensor:
    """
    Compute Morton codes (Z-order curve) for integer coordinates.

    Args:
        V: Tensor of shape (N, 3) with integer coordinates (x, y, z)
        J: Octree depth (number of bits per dimension)

    Returns:
        M: Tensor of shape (N,) with Morton codes
    """
    N = V.shape[0]
    device = V.device

    # Initialize Morton codes
    M = torch.zeros(N, dtype=torch.int64, device=device)

    # Bit weights for x, y, z: [1, 2, 4]
    tt = torch.tensor([1, 2, 4], dtype=torch.int64, device=device)

    # Interleave bits from least significant to most significant
    for i in range(J):
        # Extract i-th bit from each coordinate (z, y, x order for fliplr)
        bits = torch.stack([
            (V[:, 2] >> i) & 1,  # z bit
            (V[:, 1] >> i) & 1,  # y bit
            (V[:, 0] >> i) & 1   # x bit
        ], dim=1)

        # Add interleaved bits to Morton code
        M = M + (bits * tt.unsqueeze(0)).sum(dim=1) * (8 ** i)

    return M


def voxelize_pc(
    PC: torch.Tensor,
    vmin: Optional[torch.Tensor] = None,
    width: Optional[float] = None,
    J: int = 10,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Voxelize a point cloud using Morton code ordering.

    Args:
        PC: Point cloud tensor of shape (N, 3+d) where:
            - PC[:, 0:3] are xyz coordinates
            - PC[:, 3:] are attributes (e.g., colors)
        vmin: Minimum coordinates (3,) for bounding box. If None, computed from PC.
        width: Width of cubic bounding box. If None, computed from PC.
        J: Octree depth. Voxel size = width / (2^J)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        PCvox: Voxelized point cloud (Nvox, 3+d) with integer coords and averaged attributes
        PCsorted: Original PC sorted by Morton code (N, 3+d)
        voxel_indices: Indices marking start of each voxel (Nvox,)
        DeltaPC: Quantization error (N, 3+d) - difference from original to voxelized
    """
    # Move to device
    PC = PC.to(device)

    N = PC.shape[0]
    has_attribute = PC.shape[1] > 3

    # Split coordinates and attributes
    V = PC[:, 0:3]
    C = PC[:, 3:] if has_attribute else None

    # Compute bounding box
    if vmin is None:
        vmin = V.min(dim=0)[0]
    else:
        vmin = vmin.to(device)

    # Normalize coordinates
    V0 = V - vmin.unsqueeze(0)

    if width is None:
        width = V0.max().item()

    # Compute voxel size and integer coordinates
    voxel_size = width / (2 ** J)
    V0_integer = torch.floor(V0 / voxel_size).long()

    # Compute Morton codes
    M = get_morton_code(V0_integer, J)

    # Sort by Morton code
    M_sort, idx = torch.sort(M)
    V0 = V0[idx]
    PCsorted = V[idx]

    if has_attribute:
        C0 = C[idx]
        PCsorted = torch.cat([PCsorted, C0], dim=1)

    # Voxelize coordinates (quantize to voxel centers)
    V0_voxelized = voxel_size * torch.floor(V0 / voxel_size)
    DeltaV = V0 - V0_voxelized

    # Find voxel boundaries (where Morton code changes)
    voxel_boundary = M_sort[1:] - M_sort[:-1]
    # Indices where new voxel starts (include first point)
    voxel_indices = torch.cat([
        torch.tensor([0], device=device),
        torch.where(voxel_boundary != 0)[0] + 1
    ])

    Nvox = len(voxel_indices)

    # Voxelize attributes (average within each voxel)
    if has_attribute:
        C0_voxelized = torch.zeros_like(C0)

        for i in range(Nvox):
            start_ind = voxel_indices[i]
            end_ind = voxel_indices[i + 1] if i < Nvox - 1 else N

            # Average attributes in this voxel
            cmean = C0[start_ind:end_ind].mean(dim=0, keepdim=True)
            C0_voxelized[start_ind:end_ind] = cmean

        DeltaC = C0 - C0_voxelized

        # Create voxelized point cloud (one point per voxel)
        Vvox = V0_integer[voxel_indices]
        Cvox = C0_voxelized[voxel_indices]

        PCvox = torch.cat([Vvox.float(), Cvox], dim=1)
        DeltaPC = torch.cat([DeltaV, DeltaC], dim=1)
    else:
        # No attributes
        Vvox = V0_integer[voxel_indices]
        PCvox = Vvox.float()
        DeltaPC = DeltaV

    return PCvox, PCsorted, voxel_indices, DeltaPC


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
        # Assign each point to its voxel
        voxel_id = torch.zeros(N, dtype=torch.long, device=device)
        for i in range(Nvox):
            start_ind = voxel_indices[i]
            end_ind = voxel_indices[i + 1] if i < Nvox - 1 else N
            voxel_id[start_ind:end_ind] = i

        # Count points per voxel
        voxel_counts = torch.zeros(Nvox, dtype=torch.float32, device=device)
        voxel_counts.scatter_add_(0, voxel_id, torch.ones(N, device=device))

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
        'N': N
    }

    return PCvox, PCsorted, voxel_indices, DeltaPC, info


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
