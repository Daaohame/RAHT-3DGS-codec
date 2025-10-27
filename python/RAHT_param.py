import torch
from typing import List, Tuple
from utils import block_indices

@torch.no_grad()
def RAHT_param(
    V: torch.Tensor,
    minV: torch.Tensor,
    width: float,
    depth: int,
    return_one_based: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    RAHT prelude with optional 0/1-based output.
    Defaults to 1-based to match MATLAB exactly.
    """
    device = V.device
    N = V.shape[0]

    # --- quantization (float64 -> int64), bounds check ---
    Q = width / (2 ** depth)
    V = V.to(torch.float64, copy=False)
    minV = minV.to(torch.float64).to(device)
    Vint = torch.floor((V - minV) / Q).to(torch.int64)

    if (Vint < 0).any() or (Vint > (2 ** depth - 1)).any():
        raise ValueError("RAHT_param:OutOfBounds: indices must be within [0, 2^depth-1] per axis.")

    # --- Morton code MC (int64) ---
    MC = torch.zeros(N, dtype=torch.int64, device=device)
    for i in range(1, depth + 1):
        b = (Vint >> (i - 1)) & 1                     # (N,3) {0,1}
        digit = (b[:, 2].to(torch.int64)
                 + (b[:, 1].to(torch.int64) << 1)
                 + (b[:, 0].to(torch.int64) << 2))
        MC |= (digit << (3 * (i - 1)))

    # --- Build List / Flags / weights ---
    Nbits = 3 * depth
    List:    List[torch.Tensor] = []
    Flags:   List[torch.Tensor] = []
    weights: List[torch.Tensor] = []

    if return_one_based:
        curr_list = torch.arange(1, N + 1, device=device, dtype=torch.int64)
        take = lambda idx: MC[idx - 1]
        end_sentinel = N + 1
    else:
        curr_list = torch.arange(0, N, device=device, dtype=torch.int64)
        take = lambda idx: MC[idx]
        end_sentinel = N

    List.append(curr_list)

    for j in range(1, 65):
        next_starts = torch.cat([curr_list[1:], torch.tensor([end_sentinel], device=device, dtype=torch.int64)])
        w = (next_starts - curr_list).to(torch.int64)
        weights.append(w)

        Mj = take(curr_list)
        if Mj.numel() == 1:
            Flags.append(torch.tensor([False], dtype=torch.bool, device=device))
            break

        diff = (Mj[:-1] ^ Mj[1:])
        mask = ((torch.tensor(1, dtype=torch.int64, device=device) << Nbits)
                - (torch.tensor(1, dtype=torch.int64, device=device) << j))
        masked = diff & mask
        flag_j = torch.cat([masked.eq(0), torch.tensor([False], dtype=torch.bool, device=device)])
        Flags.append(flag_j)

        prev_flags = torch.cat([torch.tensor([False], dtype=torch.bool, device=device), flag_j[:-1]])
        tmp_list = curr_list[~prev_flags]
        if tmp_list.numel() == 1:
            break

        if j >= Nbits:
            break

        curr_list = tmp_list
        List.append(curr_list)

    return List, Flags, weights

@torch.no_grad()
def RAHT_param_reorder(
    V: torch.Tensor,
    minV: torch.Tensor,
    width: float,
    depth: int,
    return_one_based: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    RAHT prelude with optional 0/1-based output.
    Defaults to 1-based to match MATLAB exactly.
    """
    device = V.device
    N = V.shape[0]

    # --- quantization (float64 -> int64), bounds check ---
    Q = width / (2 ** depth)
    V = V.to(torch.float64, copy=False)
    minV = minV.to(torch.float64).to(device)
    Vint = torch.floor((V - minV) / Q).to(torch.int64)

    if (Vint < 0).any() or (Vint > (2 ** depth - 1)).any():
        raise ValueError("RAHT_param:OutOfBounds: indices must be within [0, 2^depth-1] per axis.")

    # --- Morton code MC (int64) ---
    MC = torch.zeros(N, dtype=torch.int64, device=device)
    for i in range(1, depth + 1):
        b = (Vint >> (i - 1)) & 1                     # (N,3) {0,1}
        digit = (b[:, 2].to(torch.int64)
                 + (b[:, 1].to(torch.int64) << 1)
                 + (b[:, 0].to(torch.int64) << 2))
        MC |= (digit << (3 * (i - 1)))

    # --- Build List / Flags / weights ---
    Nbits = 3 * depth
    List:    List[torch.Tensor] = []
    Flags:   List[torch.Tensor] = []
    weights: List[torch.Tensor] = []

    if return_one_based:
        curr_list = torch.arange(1, N + 1, device=device, dtype=torch.int64)
        take = lambda idx: MC[idx - 1]
        end_sentinel = N + 1
    else:
        curr_list = torch.arange(0, N, device=device, dtype=torch.int64)
        take = lambda idx: MC[idx]
        end_sentinel = N

    List.append(curr_list)
    ac_list = []
    indices = torch.zeros(N)
    pre_indices = torch.zeros(N)

    for j in range(1, 65):
        next_starts = torch.cat([curr_list[1:], torch.tensor([end_sentinel], device=device, dtype=torch.int64)])
        w = (next_starts - curr_list).to(torch.int64)
        weights.append(w)

        Mj = take(curr_list)
        if Mj.numel() == 1:
            Flags.append(torch.tensor([False], dtype=torch.bool, device=device))
            break

        diff = (Mj[:-1] ^ Mj[1:])
        mask = ((torch.tensor(1, dtype=torch.int64, device=device) << Nbits)
                - (torch.tensor(1, dtype=torch.int64, device=device) << j))
        masked = diff & mask
        flag_j = torch.cat([masked.eq(0), torch.tensor([False], dtype=torch.bool, device=device)])
        Flags.append(flag_j)

        prev_flags = torch.cat([torch.tensor([False], dtype=torch.bool, device=device), flag_j[:-1]])
        tmp_list = curr_list[~prev_flags]
        if tmp_list.numel() == 1 or j >= Nbits:
            indices.zero_()
            indices[tmp_list] = 1
            indices_remain = indices.bool() ^ pre_indices.bool()
            indices_remain = torch.nonzero(indices_remain, as_tuple=True)[0]
            ac_list.append(indices_remain)
            ac_list.append(tmp_list)
            ac_list = ac_list[::-1]
            order_RAGFT = torch.cat(ac_list)
            break

        curr_list = tmp_list
        List.append(curr_list)


        if torch.fmod(torch.tensor(j), 3) == 0 and torch.tensor(j > 2):
            if j == 3:
                indices[curr_list] = 1
                indices_remain =  torch.nonzero(indices == 0, as_tuple=True)[0]
                ac_list.append(indices_remain)
                pre_indices.copy_(indices)
            else:
                indices.zero_()
                indices[curr_list] = 1
                indices_remain = indices.bool() ^ pre_indices.bool()
                indices_remain = torch.nonzero(indices_remain, as_tuple=True)[0]
                ac_list.append(indices_remain)
                pre_indices.copy_(indices)


    return List, Flags, weights, order_RAGFT


@torch.no_grad()
def RAHT_param_reorder_fast(
    V: torch.Tensor,
    minV: torch.Tensor,
    width: float,
    depth: int,
    return_one_based: bool = False
):
    """
    GPU-friendly RAHT parameter reorder.
    Produces List / Flags / weights / order_RAGFT.
    """
    device = V.device
    N = V.shape[0]

    Q = width / (2 ** depth)
    Vint = torch.floor((V - minV) / Q).to(torch.int64)

    shifts = torch.arange(depth, device=device, dtype=torch.int64)
    bits = (Vint.unsqueeze(2) >> shifts) & 1   # (N,3,depth)
    digits = bits[:, 2] + (bits[:, 1] << 1) + (bits[:, 0] << 2)  # (N, depth)
    powers = (3 * torch.arange(depth, device=device, dtype=torch.int64)).unsqueeze(0)
    MC = torch.sum(digits << powers, dim=1)

    curr_list = torch.arange(N, device=device, dtype=torch.int64)
    end_sentinel = N
    Nbits = 3 * depth
    List, Flags, weights = [curr_list], [], []

    ac_list = []
    indices = torch.zeros(N, dtype=torch.bool, device=device)
    pre_indices = torch.zeros(N, dtype=torch.bool, device=device)
    order_RAGFT = None

    mask_table = ((1 << Nbits) - (1 << torch.arange(1, 65, device=device, dtype=torch.int64)))

    for j in range(1, 65):
        # compute weights
        next_starts = torch.cat([curr_list[1:], torch.tensor([end_sentinel], device=device)])
        w = (next_starts - curr_list)
        weights.append(w)

        # flags
        Mj = MC[curr_list]
        if Mj.numel() == 1:
            Flags.append(torch.zeros(1, dtype=torch.bool, device=device))
            break

        diff = (Mj[:-1] ^ Mj[1:])
        masked = diff & mask_table[j - 1]
        eq = masked.eq(0)
        flag_j = torch.zeros_like(curr_list, dtype=torch.bool)
        flag_j[:-1] = eq
        Flags.append(flag_j)

        # next list
        prev_flags = torch.zeros_like(flag_j)
        prev_flags[1:] = flag_j[:-1]
        tmp_list = curr_list[~prev_flags]

        # === RAGFT tracking ===
        if j % 3 == 0 and j > 2:
            indices.zero_()
            indices[tmp_list] = True
            if j == 3:
                remain = (~indices).nonzero(as_tuple=True)[0]
                ac_list.append(remain)
                pre_indices.copy_(indices)
            else:
                diff_mask = indices ^ pre_indices
                remain = diff_mask.nonzero(as_tuple=True)[0]
                ac_list.append(remain)
                pre_indices.copy_(indices)

        # === Termination ===
        if tmp_list.numel() == 1 or j >= Nbits:
            indices.zero_()
            indices[tmp_list] = True
            diff_mask = indices ^ pre_indices
            remain = diff_mask.nonzero(as_tuple=True)[0]
            ac_list.append(remain)
            ac_list.append(tmp_list)
            ac_list = ac_list[::-1]
            order_RAGFT = torch.cat(ac_list)
            break

        curr_list = tmp_list
        List.append(curr_list)

    return List, Flags, weights, order_RAGFT



@torch.no_grad()
def morton_encode_vectorized(coords: torch.Tensor, depth: int) -> torch.Tensor:
    """
    Vectorized Morton code encoding using bit manipulation.
    coords: (N, 3) int64 tensor with values in [0, 2^depth-1]
    Returns: (N,) int64 Morton codes
    """
    device = coords.device
    N = coords.shape[0]
    
    # Separate coordinates
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Spread bits for each coordinate (dilate by factor of 3)
    # For each bit position, shift it to its final position in Morton code
    morton = torch.zeros(N, dtype=torch.int64, device=device)
    
    for i in range(depth):
        # Extract bit i from each coordinate
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        z_bit = (z >> i) & 1
        
        # Place in Morton code: bit i of x goes to position 3*i+2
        #                        bit i of y goes to position 3*i+1
        #                        bit i of z goes to position 3*i
        morton |= (z_bit << (3 * i)) | (y_bit << (3 * i + 1)) | (x_bit << (3 * i + 2))
    
    return morton


@torch.no_grad()
def RAHT_param_optimized(
    V: torch.Tensor,
    minV: torch.Tensor,
    width: float,
    depth: int,
    return_one_based: bool = False,
    max_levels: int = 64
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    GPU-optimized RAHT parameterization with parallel Morton encoding
    and optimized hierarchical clustering.
    
    Args:
        V: (N, 3) point coordinates
        minV: (3,) minimum bounds for quantization
        width: quantization width
        depth: octree depth (bits per axis)
        return_one_based: MATLAB compatibility (1-indexed)
        max_levels: maximum hierarchy levels to compute
    
    Returns:
        List, Flags, weights for hierarchical transform
    """
    device = V.device
    N = V.shape[0]
    
    # --- Quantization (fully vectorized) ---
    Q = width / (2 ** depth)
    V = V.to(torch.float64, copy=False)
    minV = minV.to(torch.float64).to(device)
    Vint = torch.floor((V - minV) / Q).to(torch.int64)
    
    # Bounds check (single kernel launch)
    max_val = (2 ** depth - 1)
    if (Vint < 0).any() or (Vint > max_val).any():
        raise ValueError(f"Indices must be within [0, {max_val}] per axis.")
    
    # --- Vectorized Morton encoding ---
    MC = morton_encode_vectorized(Vint, depth)
    
    # --- Sort by Morton code for better memory access patterns ---
    sorted_mc, sort_indices = torch.sort(MC)
    
    # --- Pre-allocate outputs ---
    Nbits = 3 * depth
    max_levels = min(max_levels, Nbits + 1)
    
    List: List[torch.Tensor] = []
    Flags: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    
    # Index adjustment for 0/1-based
    if return_one_based:
        curr_list = sort_indices + 1
        mc_lookup = lambda idx: sorted_mc[idx - 1] if idx.numel() > 0 else sorted_mc[idx]
        end_sentinel = N + 1
    else:
        curr_list = sort_indices
        mc_lookup = lambda idx: sorted_mc[idx] if idx.numel() > 0 else sorted_mc[idx]
        end_sentinel = N
    
    List.append(curr_list)
    
    # --- Hierarchical clustering (optimized) ---
    for j in range(1, max_levels):
        curr_size = curr_list.shape[0]
        
        if curr_size == 1:
            # Terminal case: single node
            weights.append(torch.tensor([end_sentinel - curr_list[0]], 
                                       dtype=torch.int64, device=device))
            Flags.append(torch.tensor([False], dtype=torch.bool, device=device))
            break
        
        # Compute weights (vectorized)
        next_starts = torch.cat([
            curr_list[1:], 
            torch.tensor([end_sentinel], device=device, dtype=torch.int64)
        ])
        w = next_starts - curr_list
        weights.append(w)
        
        # Get Morton codes for current level
        Mj = mc_lookup(curr_list)
        
        # Compute flags using vectorized XOR and masking
        mask = ((1 << Nbits) - (1 << j))
        
        # XOR consecutive Morton codes and mask
        diff = Mj[:-1] ^ Mj[1:]
        masked_diff = diff & mask
        
        # flag_j[i] = True means position i and i+1 should merge
        flag_j = torch.cat([
            masked_diff.eq(0),
            torch.tensor([False], dtype=torch.bool, device=device)
        ])
        
        Flags.append(flag_j)
        
        # Build next level list (vectorized filtering)
        # Keep nodes that don't have a flag in the PREVIOUS position
        prev_flags = torch.cat([
            torch.tensor([False], dtype=torch.bool, device=device),
            flag_j[:-1]
        ])
        
        curr_list = curr_list[~prev_flags]
        
        if curr_list.numel() <= 1 or j >= Nbits:
            break
        
        List.append(curr_list)
    
    return List, Flags, weights