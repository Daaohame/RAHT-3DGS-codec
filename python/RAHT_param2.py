import torch

@torch.no_grad()
def RAHT_param2(V: torch.Tensor,
                     minV: torch.Tensor,
                     width: float,
                     depth: int,
                     return_one_based: bool = False):
    """
    PyTorch implementation (no NumPy) of RAHT_param.
    Inputs can be on CPU or CUDA; outputs stay on the same device.

    Parameters
    ----------
    V : (N,3) float/long tensor
        Quantized & voxelized coords in Morton order.
    minV : (3,) float/long tensor
        Bounding-box min used for voxelization.
    width : float
    depth : int
    return_one_based : bool
        If True, List entries are 1-based (MATLAB style).

    Returns
    -------
    List   : list[1D LongTensor]
    Flags  : list[1D BoolTensor]
    weights: list[1D LongTensor]
    """
    device = V.device
    N = V.shape[0]

    # ---- integer voxels (uint behavior emulated via int64) ----
    Q = width / (2 ** depth)
    V = V.to(dtype=torch.float64)
    minV = minV.to(dtype=torch.float64, device=device)
    Vint = torch.floor((V - minV) / Q).to(torch.int64)        # (N,3)

    # ---- Morton code MC (int64 is sufficient for practical depths) ----
    MC = torch.zeros(N, dtype=torch.int64, device=device)
    tri = torch.tensor([1, 2, 4], dtype=torch.int64, device=device)
    for i in range(1, depth + 1):
        bits = ((Vint >> (i - 1)) & 1)                        # (N,3) in {0,1}
        bits_rev = bits[:, [2, 1, 0]].to(torch.int64)         # fliplr
        MC = MC + (bits_rev * tri).sum(dim=1)
        tri = tri * 8

    # ---- Build List / Flags / weights ----
    Nbits = 3 * depth
    List, Flags, weights = [], [], []

    curr_list = (torch.arange(1, N + 1, device=device, dtype=torch.int64)
                 if return_one_based else
                 torch.arange(N, device=device, dtype=torch.int64))

    def take_MC(idx: torch.Tensor) -> torch.Tensor:
        return MC[idx - 1] if return_one_based else MC[idx]

    j = 1
    while True:
        List.append(curr_list)

        # weights{j} = [List{j}(2:end); end_sentinel] - List{j}
        end_sentinel = (N + 1) if return_one_based else N
        next_starts = torch.cat([
            curr_list[1:],
            torch.tensor([end_sentinel], dtype=curr_list.dtype, device=device)
        ])
        w = (next_starts - curr_list).to(torch.int64)
        weights.append(w)

        Mj = take_MC(curr_list).to(torch.int64)
        if Mj.numel() <= 1:
            Flags.append(torch.tensor([False], dtype=torch.bool, device=device))
            break

        # diff of adjacent MC; mask keeps the top (Nbits - j) bits
        diff  = torch.bitwise_xor(Mj[:-1], Mj[1:])
        mask  = (torch.tensor(1, dtype=torch.int64, device=device) << Nbits) - \
                (torch.tensor(1, dtype=torch.int64, device=device) << j)
        masked = torch.bitwise_and(diff, mask)
        flag_j = torch.cat([masked.eq(0), torch.tensor([False], device=device, dtype=torch.bool)])
        Flags.append(flag_j)

        # tmpList = curr_list(~[0; Flags(1:end-1)])
        prev_flags = torch.cat([torch.tensor([False], device=device, dtype=torch.bool),
                                flag_j[:-1]])
        tmp_list = curr_list[~prev_flags]

        if tmp_list.numel() == 1:
            curr_list = tmp_list
            j += 1
            if j > 64:   # 保持与原实现一致的保护
                break
            continue

        curr_list = tmp_list
        j += 1
        if j > 64:
            break

    return List, Flags, weights
