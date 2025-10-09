import torch
from typing import List, Tuple

def RAHT_param(
    V: torch.Tensor, 
    minV: torch.Tensor, 
    width: float, 
    depth: int,
    device: str='cpu'
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Computes index lists, flags, and weights for RAHT (Recursive Aggregated
    Hierarchical Transform) based on Morton-ordered voxel data.

    Args:
        V (torch.Tensor): A tensor of shape (N, 3) containing quantized and
                          voxelized 3D points in Morton order.
        minV (torch.Tensor): A tensor of shape (1, 3) with the minimum
                             coordinate values for quantization.
        width (float): The width of the entire voxel grid.
        depth (int): The depth of the octree.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            - List_ (List[torch.Tensor]): A list where each element is a tensor of
              indices representing the start of a group at a specific hierarchy level.
            - Flags (List[torch.Tensor]): A list of boolean tensors. A True flag at
              index i means the groups i and i+1 at that level can be merged.
            - weights (List[torch.Tensor]): A list of tensors containing the number
              of points in each group at each level.
    """
    V = V.to(device)
    minV = minV.to(device)

    # Compute Morton codes from voxel coordinates
    Q = width / (2**depth)
    sizeV = V.size(0)
    Vint = torch.floor((V - minV.expand_as(V)) / Q).to(torch.long)

    MC = torch.zeros((sizeV, 1), dtype=torch.long, device=device)
    tri = torch.tensor([[1], [2], [4]], dtype=torch.long, device=device)

    for i in range(1, depth + 1):
        bits = (Vint >> (i - 1)) & 1
        flipped_bits = torch.fliplr(bits)
        MC += torch.matmul(flipped_bits, tri)
        tri *= 8

    # Create list, flag, and weight arrays
    Nbits = 3 * depth
    List_ = []
    Flags = []
    weights = []
    
    # initialize List_ with all point indices
    current_list = torch.arange(sizeV, device=device).unsqueeze(1)
    List_.append(current_list)

    for j in range(Nbits):
        current_list_j = List_[j]

        # a. Compute weights
        next_indices = torch.cat((
            current_list_j[1:], 
            torch.tensor([[sizeV]], dtype=torch.long, device=device)
        ))
        weights.append(next_indices - current_list_j)

        # b. Get Morton codes for the group-starting points
        Mj = MC[current_list_j.squeeze()]
        
        # Break if we don't have at least two groups to compare
        if Mj.numel() < 2:
            break
            
        # c. Find differences between consecutive Morton codes
        diff = torch.bitwise_xor(Mj[:-1], Mj[1:])

        # d. Check for matching prefixes
        mask = (2**Nbits) - (2**(j + 1))
        masked = torch.bitwise_and(diff, mask)
        current_flags = (masked == 0).view(-1)
        Flags.append(torch.cat((
            current_flags, 
            torch.tensor([False], device=device)
        )))

        # e. Create the list for the next level
        prev_flags = Flags[j][:-1]
        keep_mask = ~torch.cat((
            torch.tensor([False], device=device),
            prev_flags
        ))
        next_list = current_list_j[keep_mask]
        
        # f. Stop if the hierarchy is complete
        if next_list.numel() <= 1:
            break

        List_.append(next_list)
        
    return List_, Flags, weights
