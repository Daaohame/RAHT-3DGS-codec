import torch
import numpy as np
# The 'open3d' library is used for PLY file writing.
import open3d as o3d

def _spread_bits(x: torch.Tensor) -> torch.Tensor:
    """
    Spreads the bits of a 16-bit integer tensor for 3D Morton codes.
    This is a vectorized implementation of bit interleaving.
    Example: 1111 -> 001001001001
    """
    x = x.long() # Promote to 64-bit to have space for bit manipulation
    x = (x | (x << 16)) & 0x0000ffff0000ffff
    x = (x | (x << 8))  & 0x00ff00ff00ff00ff
    x = (x | (x << 4))  & 0x0f0f0f0f0f0f0f0f
    x = (x | (x << 2))  & 0x3333333333333333
    x = (x | (x << 1))  & 0x5555555555555555
    return x

def get_morton_code(v_integer: torch.Tensor) -> torch.Tensor:
    """
    Calculates the 3D Morton code for a tensor of integer coordinates.

    Args:
        v_integer (torch.Tensor): An [N, 3] integer tensor of coordinates.
                                  Values should ideally be within [0, 2^16-1].

    Returns:
        torch.Tensor: A 1D tensor of shape [N] containing the Morton codes.
    """
    if v_integer.shape[1] != 3:
        raise ValueError("Input tensor must have 3 columns for x, y, z coordinates.")

    # Clamp coordinates to ensure they fit within 16 bits for this implementation
    max_coord = (1 << 16) - 1
    x = torch.clamp(v_integer[:, 0], 0, max_coord)
    y = torch.clamp(v_integer[:, 1], 0, max_coord)
    z = torch.clamp(v_integer[:, 2], 0, max_coord)

    return _spread_bits(x) | (_spread_bits(y) << 1) | (_spread_bits(z) << 2)


def voxelize_pc(PC: torch.Tensor, param: dict):
    """
    Voxelizes a point cloud by grouping points into voxels using an octree-like
    structure defined by Morton codes.

    Args:
        PC (torch.Tensor): An [N, 3+d] tensor where the first 3 columns are XYZ
                           coordinates and the remaining d columns are attributes (e.g., colors).
        param (dict): A dictionary containing voxelization parameters:
            'vmin' (list, tuple, or torch.Tensor, optional): The minimum coordinate [x,y,z].
            'width' (float, optional): The side length of the cubic bounding box.
            'J' (int): The maximum depth of the octree decomposition.
            'writeFileOut' (bool): Flag to control whether output files are written.
            'filename' (str): The base filename for output files.

    Returns:
        tuple: A tuple containing the voxelized point cloud, sorted original cloud,
               voxel indices, and the quantization error.
    """
    # --- Parameter Extraction ---
    vmin = param.get('vmin')
    width = param.get('width')
    J = param.get('J')
    writeFileOut = param.get('writeFileOut', False)
    filename = param.get('filename')
    device = PC.device

    # --- Data Splitting (Vertices and Attributes) ---
    hasAttribute = PC.shape[1] > 3
    V = PC[:, 0:3]
    C = PC[:, 3:] if hasAttribute else None

    # --- Bounding Box Calculation ---
    if vmin is None:
        vmin, _ = torch.min(V, dim=0)
    else:
        # FIX: Ensure vmin is a tensor with the correct type and device.
        if not isinstance(vmin, torch.Tensor):
            vmin = torch.tensor(vmin, dtype=V.dtype, device=device)

    N = PC.shape[0]
    # This subtraction now works correctly.
    V0 = V - vmin

    if width is None:
        width = torch.max(V0)

    # --- Voxelization and Morton Code Sorting ---
    voxel_size = width / (2**J)
    V0_integer = torch.floor(V0 / voxel_size)
    M = get_morton_code(V0_integer.int())

    M_sort, idx = torch.sort(M)

    V0 = V0[idx, :]
    PCsorted = V[idx, :]
    C0 = C[idx, :] if hasAttribute else None
    if hasAttribute:
        PCsorted = torch.cat((PCsorted, C0), dim=1)

    # --- Quantization and Error Calculation (DeltaV) ---
    V0voxelized = voxel_size * (torch.floor(V0 / voxel_size))
    DeltaV = V0 - V0voxelized

    # --- Find Voxel Boundaries ---
    voxel_boundary_flags = M_sort[1:] - M_sort[:-1]
    boundary_indices = (torch.nonzero(voxel_boundary_flags).squeeze(1) + 1)
    voxel_indices = torch.cat((torch.tensor([0], device=device), boundary_indices))

    # --- Voxel Attribute Averaging (DeltaC) ---
    Nvox = voxel_indices.shape[0]
    DeltaC = None
    C0voxelized = None
    
    if hasAttribute:
        C0voxelized = torch.zeros_like(C0)
        for i in range(Nvox):
            start_ind = voxel_indices[i]
            end_ind = N if (i == Nvox - 1) else voxel_indices[i+1]
            cmean = torch.mean(C0[start_ind:end_ind, :], dim=0)
            C0voxelized[start_ind:end_ind, :] = cmean
        DeltaC = C0 - C0voxelized

    # --- Assemble Final Voxelized Point Cloud and Deltas ---
    Vvox = V0_integer[voxel_indices, :]
    PCvox = Vvox.float()
    DeltaPC = DeltaV

    if hasAttribute:
        Cvox = C0voxelized[voxel_indices, :]
        PCvox = torch.cat((Vvox.float(), Cvox), dim=1)
        DeltaPC = torch.cat((DeltaV, DeltaC), dim=1)

    # --- File Output ---
    if writeFileOut:
        if filename is None:
            raise ValueError("A filename must be provided in 'param' when writeFileOut is True.")

        # 1. Write the voxelized point cloud to a .ply file
        Vvox_coords = (Vvox.float() * voxel_size) + vmin
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

        # 2. Write the metadata and deltas to a .txt file
        filename_data = f"{filename}_data.txt"
        with open(filename_data, 'w') as f:
            header = f"{vmin[0]} {vmin[1]} {vmin[2]} {width} {J} {Nvox} {N} {int(hasAttribute)}\n"
            f.write(header)
            np.savetxt(f, voxel_indices.cpu().numpy(), fmt='%d')
            np.savetxt(f, DeltaPC.cpu().numpy(), fmt='%.6f')
        print(f"Voxelization data saved to {filename_data}")

    return PCvox, PCsorted, voxel_indices, DeltaPC
