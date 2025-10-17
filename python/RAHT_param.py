import torch
from typing import List, Tuple

@torch.no_grad()
def RAHT_param(
    V: torch.Tensor,
    minV: torch.Tensor,
    width: float,
    depth: int,
    return_one_based: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    MATLAB-equivalent RAHT_param with optional 0/1-based output.
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
        raise ValueError("RAHT_param2:OutOfBounds: indices must be within [0, 2^depth-1] per axis.")

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