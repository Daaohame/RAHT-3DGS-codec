import torch
import torch.nn.functional as F
from typing import List, Tuple, Union

def inverse_RAHT(Coeff: torch.Tensor, List, Flags, weights):
    """
    GPU/CPU compatible version of inverse RAHT transform.
    """
    device = Coeff.device

    T = Coeff.clone().to(device)
    Nlevels = len(Flags)

    for j in range(Nlevels - 1, -1, -1):
        left_sibling_index = Flags[j].to(device)
        right_sibling_index = torch.cat([
            torch.zeros(1, dtype=left_sibling_index.dtype, device=device),
            left_sibling_index[:-1]
        ])

        i0 = List[j][left_sibling_index == 1].to(device)
        i1 = List[j][right_sibling_index == 1].to(device)

        x0 = T[i0, :]
        x1 = T[i1, :]
        signal_dimension = T.shape[1]

        w0 = weights[j][left_sibling_index == 1].to(device)
        w1 = weights[j][right_sibling_index == 1].to(device)

        a = torch.sqrt(w0 / (w0 + w1)).unsqueeze(1).expand(-1, signal_dimension)
        b = torch.sqrt(w1 / (w0 + w1)).unsqueeze(1).expand(-1, signal_dimension)

        T[i0, :] = a * x0 - b * x1
        T[i1, :] = b * x0 + a * x1

    return T


def inverse_RAHT_optimized(T: torch.Tensor, 
                           List: List[torch.Tensor], 
                           Flags: List[torch.Tensor], 
                           weights: List[torch.Tensor],
                           one_based: bool = False) -> torch.Tensor:
    """
    PyTorch version of the inverse RAHT.

    Parameters
    ----------
    T : (N, D) float tensor
        Input transformed coefficients (from RAHT2_optimized).
    List : list[LongTensor]
        From RAHT_param_torch; List[j] are group start indices at level j.
    Flags : list[BoolTensor]
        Flags[j][k]==True iff k and k+1 share the same MSB prefix at level j.
        Last element padded False (same as MATLAB).
    weights : list[LongTensor]
        Run-length weights per level (length of each group).
    one_based : bool
        If True, `List` entries are 1-based and will be converted to 0-based for tensor indexing.

    Returns
    -------
    C : (N, D) float tensor
        Reconstructed original attributes.
    """
    device = T.device
    N, D = T.shape
    C = T.clone().to(torch.float64).to(device)

    def to0(idx: torch.Tensor) -> torch.Tensor:
        return idx - 1 if one_based else idx

    Nlevels = len(Flags)
    # Iterate top-down (reverse of the forward pass)
    for j in reversed(range(Nlevels)):
        # sibling masks at this level
        left_mask  = Flags[j]
        right_mask = torch.cat([torch.tensor([False], device=device),
                                Flags[j][:-1]])

        # indices of left and right siblings (in the global order)
        i0 = List[j][left_mask]
        i1 = List[j][right_mask]

        if i0.numel() == 0:
            continue

        i0_ = to0(i0).long()
        i1_ = to0(i1).long()

        # pick coefficients (from our working tensor 'C')
        T_i0 = C.index_select(0, i0_) # (M,D)
        T_i1 = C.index_select(0, i1_) # (M,D)

        # pick transform weights for this level (run-lengths)
        w0 = weights[j][left_mask].to(torch.float64)
        w1 = weights[j][right_mask].to(torch.float64)
        denom = w0 + w1
        # denom = torch.clamp(denom, min=1e-12)

        a = torch.sqrt(w0 / denom).unsqueeze(1)  # (M,1)
        b = torch.sqrt(w1 / denom).unsqueeze(1)  # (M,1)

        # We do NOT update 'w' as this is the inverse operation.

        # Inverse 2x2 RAHT butterfly
        x0 = a * T_i0 - b * T_i1
        x1 = b * T_i0 + a * T_i1
        
        C.scatter_(0, i0_.unsqueeze(1).expand(-1, D), x0)
        C.scatter_(0, i1_.unsqueeze(1).expand(-1, D), x1)

    return C