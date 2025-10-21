import torch
import torch.nn.functional as F
from typing import List, Tuple, Union

@torch.no_grad()
def RAHT(C, List, Flags, weights, device: Union[str, torch.device] = 'cuda'):
    """
    Converts the MATLAB RAHT function to Python using PyTorch.

    Args:
        C (torch.Tensor): Input tensor of shape [N, number_of_attributes].
        List (list of torch.Tensor): List of index tensors for each level.
        Flags (list of torch.Tensor): List of binary flag tensors for each level.
        weights (list of torch.Tensor): List of weight tensors for each level.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed
                                           tensor T and the updated weights w.
    """
    # Initialize T and w
    T = C.clone().to(device)
    w = torch.ones(C.size(0), 1, device=device, dtype=C.dtype)

    # MATLAB: Nlevels=length(Flags); % of the octree
    Nlevels = len(Flags)

    # MATLAB: for j=1:Nlevels % bottom up
    for j in range(Nlevels):

        # MATLAB: left_sibling_index=Flags{j};
        left_sibling_index = Flags[j]

        # MATLAB: right_sibling_index=[0;Flags{j}(1:end-1)];
        zero_tensor = torch.tensor([0], device=C.device, dtype=left_sibling_index.dtype)
        right_sibling_index = torch.cat((zero_tensor, left_sibling_index[:-1]))

        # Create boolean masks for logical indexing
        i0_mask = (left_sibling_index == 1)
        i1_mask = (right_sibling_index == 1)

        # MATLAB: i0=List{j}(left_sibling_index==1);
        i0 = List[j][i0_mask]

        # MATLAB: i1=List{j}(right_sibling_index==1);
        i1 = List[j][i1_mask]

        # MATLAB: if(~isempty(i0) && ~isempty(i1))
        if i0.numel() > 0 and i1.numel() > 0:
            # MATLAB: x0=T(i0,:);
            x0 = T[i0, :]
            # MATLAB: x1=T(i1,:);
            x1 = T[i1, :]

            # MATLAB: w0=weights{j}(left_sibling_index==1);
            w0 = weights[j][i0_mask]
            # MATLAB: w1=weights{j}(right_sibling_index==1);
            w1 = weights[j][i1_mask]

            # MATLAB: a=sqrt(w0./(w0+w1));
            # MATLAB: b=sqrt(w1./(w0+w1));
            w_sum = w0 + w1
            a = torch.sqrt(w0 / w_sum)
            b = torch.sqrt(w1 / w_sum)

            # MATLAB: w(i0)=w(i0)+w(i1);
            # MATLAB: w(i1)=w(i0);
            # Note: The second assignment in MATLAB uses the *updated* value of w(i0).
            w_i0_updated = w[i0] + w[i1]
            w[i0] = w_i0_updated
            w[i1] = w_i0_updated

            # MATLAB: T(i0,:)=repmat(a,1,signal_dimension).*x0+repmat(b,1,signal_dimension).*x1;
            # MATLAB: T(i1,:)=-repmat(b,1,signal_dimension).*x0+repmat(a,1,signal_dimension).*x1;
            # PyTorch's broadcasting automatically handles the expansion of a and b.
            a_reshaped = a.view(-1, *([1] * (x0.ndim - 1)))
            b_reshaped = b.view(-1, *([1] * (x0.ndim - 1)))
            T[i0, :] = a_reshaped * x0 + b_reshaped * x1
            T[i1, :] = -b_reshaped * x0 + a_reshaped * x1

    return T, w


@torch.no_grad()
def RAHT2(C,
               List,
               Flags,
               weights,
               one_based: bool = False):
    """
    PyTorch version of:
        function [T,w] = RAHT(C,List,Flags,weights)

    Parameters
    ----------
    C : (N, D) float tensor
        Input attributes (rows correspond to points in Morton order).
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
    T : (N, D) float tensor
        Transformed coefficients.
    w : (N, 1)  long tensor
        Aggregated node weights after combining siblings.
    """
    device = C.device
    N, D = C.shape
    T = C.clone().to(torch.float64)
    w = torch.ones((N, 1), dtype=torch.float64, device=device)

    def to0(idx: torch.Tensor) -> torch.Tensor:
        return idx - 1 if one_based else idx

    Nlevels = len(Flags)
    # MATLAB: for j = 1:Nlevels-3  (1-based). Python 0-based: range(Nlevels-3)
    for j in range(Nlevels):
    # for j in range(1):
        # sibling masks at this level
        left_mask  = Flags[j]                              # (len(List[j]),)
        right_mask = torch.cat([torch.tensor([False], device=device),
                                Flags[j][:-1]])            # [0; Flags(1:end-1)]

        # indices of left and right siblings (in the global order)
        i0 = List[j][left_mask]                            # starts that HAVE right siblings
        i1 = List[j][right_mask]                           # their right siblings

        if i0.numel() == 0:                                # nothing to do at this level
            continue

        i0_ = to0(i0).long()
        i1_ = to0(i1).long()

        # pick coefficients
        x0 = T.index_select(0, i0_)                        # (M,D)
        x1 = T.index_select(0, i1_)                        # (M,D)

        # pick transform weights for this level (run-lengths)
        w0 = weights[j][left_mask].to(torch.float64)       # (M,)
        w1 = weights[j][right_mask].to(torch.float64)      # (M,)
        denom = w0 + w1
        # numerical safety（不期望出现0，但以防万一）
        # denom = torch.clamp(denom, min=1e-12)

        a = torch.sqrt(w0 / denom).unsqueeze(1)  # (M,1)
        b = torch.sqrt(w1 / denom).unsqueeze(1)  # (M,1)

        # update node weights (MATLAB: w(i0)=w(i0)+w(i1); w(i1)=w(i0))
        new_w0 = w.index_select(0, i0_) + w.index_select(0, i1_)  # (M,1)
        new_w1 = new_w0.clone()
        w.scatter_(0, i0_.unsqueeze(1), new_w0)
        w.scatter_(0, i1_.unsqueeze(1), new_w1)

        # 2x2 RAHT butterfly
        T_i0 = a * x0 + b * x1
        T_i1 = -b * x0 + a * x1
        T.scatter_(0, i0_.unsqueeze(1).expand(-1, D), T_i0)
        T.scatter_(0, i1_.unsqueeze(1).expand(-1, D), T_i1)

    return T, w


@torch.no_grad()
def RAHT_optimized(C: torch.Tensor, 
                   List: List[torch.Tensor], 
                   Flags: List[torch.Tensor], 
                   weights: List[torch.Tensor],
                   device: Union[str, torch.device] = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fully GPU-optimized Region Adaptive Hierarchical Transform (RAHT)

    Args:
        C: Input coefficients tensor [N, number_of_attributes]
        List: List of node indices for each level
        Flags: List of binary flags indicating left siblings for each level
        weights: List of transform weights for each level
        device: Device to run computations on ('cuda' or 'cpu')

    Returns:
        Tuple of (T, w) where:
        T: Transformed coefficients [N, number_of_attributes]
        w: Updated weights [N, 1]
    """
    # Initialize T and w
    T = C.clone().to(device)
    w = torch.ones(C.size(0), 1, device=device, dtype=C.dtype)

    Nlevels = len(Flags)

    for j in range(Nlevels):
        # Move all data to device at once
        left_sibling_mask = Flags[j].to(device, non_blocking=True)
        list_j = List[j].to(device, non_blocking=True)
        weights_j = weights[j].to(device, non_blocking=True)

        # Create right sibling mask using GPU operations
        right_sibling_mask = torch.cat([
            torch.zeros(1, dtype=left_sibling_mask.dtype, device=device),
            left_sibling_mask[:-1]
        ])

        # Get sibling pairs using vectorized operations
        left_indices = list_j[left_sibling_mask.bool()]
        right_indices = list_j[right_sibling_mask.bool()]

        if left_indices.numel() > 0 and right_indices.numel() > 0:
            # Vectorized coefficient extraction
            x0 = T[left_indices]   # [num_pairs, signal_dimension]
            x1 = T[right_indices]  # [num_pairs, signal_dimension]

            # Vectorized weight extraction
            w0 = weights_j[left_sibling_mask.bool()]
            w1 = weights_j[right_sibling_mask.bool()]

            # Compute transform coefficients in parallel
            w_sum = w0 + w1
            # Use rsqrt for better GPU performance
            inv_sqrt_w_sum = torch.rsqrt(w_sum)
            a = torch.sqrt(w0) * inv_sqrt_w_sum
            b = torch.sqrt(w1) * inv_sqrt_w_sum

            # Expand for broadcasting (more efficient than unsqueeze + expand)
            a_broad = a[:, None]  # [num_pairs, 1]
            b_broad = b[:, None]  # [num_pairs, 1]

            # Parallel RAHT transform computation
            new_x0 = a_broad * x0 + b_broad * x1
            new_x1 = -b_broad * x0 + a_broad * x1

            # Update T using advanced indexing (parallel writes)
            T[left_indices] = new_x0
            T[right_indices] = new_x1

            # Update weights in parallel
            w_left_old = w[left_indices, 0]
            w_right_old = w[right_indices, 0]
            w_new = w_left_old + w_right_old

            w[left_indices, 0] = w_new
            w[right_indices, 0] = w_new

    return T, w


@torch.no_grad()
def RAHT2_optimized(C,
               List,
               Flags,
               weights,
               one_based: bool = False):
    """
    PyTorch version of:
        function [T,w] = RAHT(C,List,Flags,weights)

    Parameters
    ----------
    C : (N, D) float tensor
        Input attributes (rows correspond to points in Morton order).
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
    T : (N, D) float tensor
        Transformed coefficients.
    w : (N, 1)  long tensor
        Aggregated node weights after combining siblings.
    """
    device = C.device
    N, D = C.shape
    T = C.clone().to(torch.float64).to(device)
    w = torch.ones((N, 1), dtype=torch.float64, device=device)

    def to0(idx: torch.Tensor) -> torch.Tensor:
        return idx - 1 if one_based else idx

    Nlevels = len(Flags)
    # MATLAB: for j = 1:Nlevels-3  (1-based). Python 0-based: range(Nlevels-3)
    for j in range(Nlevels):
    # for j in range(1):
        # sibling masks at this level
        left_mask  = Flags[j]                              # (len(List[j]),)
        right_mask = torch.cat([torch.tensor([False], device=device),
                                Flags[j][:-1]])            # [0; Flags(1:end-1)]

        # indices of left and right siblings (in the global order)
        i0 = List[j][left_mask]                            # starts that HAVE right siblings
        i1 = List[j][right_mask]                           # their right siblings

        if i0.numel() == 0:                                # nothing to do at this level
            continue

        i0_ = to0(i0).long()
        i1_ = to0(i1).long()

        # pick coefficients
        x0 = T.index_select(0, i0_)                        # (M,D)
        x1 = T.index_select(0, i1_)                        # (M,D)

        # pick transform weights for this level (run-lengths)
        w0 = weights[j][left_mask].to(torch.float64)       # (M,)
        w1 = weights[j][right_mask].to(torch.float64)      # (M,)
        denom = w0 + w1
        # numerical safety（不期望出现0，但以防万一）
        # denom = torch.clamp(denom, min=1e-12)

        a = torch.sqrt(w0 / denom).unsqueeze(1)  # (M,1)
        b = torch.sqrt(w1 / denom).unsqueeze(1)  # (M,1)

        # update node weights (MATLAB: w(i0)=w(i0)+w(i1); w(i1)=w(i0))
        new_w0 = w.index_select(0, i0_) + w.index_select(0, i1_)  # (M,1)
        new_w1 = new_w0.clone()
        w.scatter_(0, i0_.unsqueeze(1), new_w0)
        w.scatter_(0, i1_.unsqueeze(1), new_w1)

        # 2x2 RAHT butterfly
        T_i0 = a * x0 + b * x1
        T_i1 = -b * x0 + a * x1
        T.scatter_(0, i0_.unsqueeze(1).expand(-1, D), T_i0)
        T.scatter_(0, i1_.unsqueeze(1).expand(-1, D), T_i1)

    return T, w

