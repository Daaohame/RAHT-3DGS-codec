import torch
from scipy.io import savemat, loadmat

def rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
    """
    RGB -> YUV (full range with 128/255 Cb/Cr offsets)
    Input:  Nx3 torch.Tensor, range [0,255], dtype=torch.float32 or torch.uint8
    Output: Nx3 torch.float64, range [0,255], clipped
    """
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("Expected Nx3 tensor")

    device = rgb.device
    rgb = rgb.to(torch.float64)
    rgb1 = torch.hstack((rgb / 255.0, torch.ones((rgb.size(0), 1), dtype=torch.float64, device=device)))

    Q = torch.tensor([                          # BT.709 full-range
        [0.21260000, -0.114572,   0.5      ],
        [0.71520000, -0.385428,  -0.454153 ],
        [0.07220000,  0.5,       -0.045847 ],
        [0.0,         0.50196078, 0.50196078]
    ], dtype=torch.float64, device=device)
    
    # Q = torch.tensor([                          # BT.601 full-range
    #     [0.29899999, -0.1687,    0.5      ],
    #     [0.58700000, -0.3313,   -0.4187   ],
    #     [0.11400000,  0.5,      -0.0813   ],
    #     [0.0,         0.50196078, 0.50196078]
    # ], dtype=torch.float64, device=device)

    yuv = rgb1 @ Q
    yuv = torch.clamp(yuv, 0.0, 1.0) * 255.0
    return yuv


def save_mat(tensor: torch.Tensor, filename: str) -> None:
    savemat(filename, {"data": tensor.detach().cpu().numpy()})

def save_lists(filename, **kwargs):
    out = {}
    for key, tensor_list in kwargs.items():
        out[key] = [t.detach().cpu().numpy() for t in tensor_list]
    savemat(filename, out)


def sanity_check_vector(T: torch.Tensor, C: torch.Tensor, rtol=1e-5, atol=1e-8) -> bool:
    """
    Sanity check: max(T) == sqrt(N) * mean(C)
    T, C: 1D tensors of shape [N]
    """
    assert T.dim() == 1 and C.dim() == 1 and T.size(0) == C.size(0), "T and C must be 1D with same length"
    N = T.size(0)

    lhs = T.max()
    rhs = torch.sqrt(torch.tensor(float(N), dtype=C.dtype, device=C.device)) * C.mean()

    return torch.allclose(lhs, rhs, rtol=rtol, atol=atol)


def is_frame_morton_ordered(Vin: torch.Tensor, J: int):
    """
    Equivalent to MATLAB function:
        [error, out, index] = is_frame_morton_ordered(Vin, J)

    Args:
        Vin (torch.Tensor): Nx3 tensor, each row = (x, y, z)
        J (int): number of bits per coordinate

    Returns:
        error (float): L2 norm between original and sorted coordinates
        out (torch.Tensor): Vin reordered by Morton order
        index (torch.Tensor): sorting indices
    """

    # Ensure type
    if not torch.is_tensor(Vin):
        Vin = torch.tensor(Vin, dtype=torch.float64)
    else:
        Vin = Vin.to(torch.float64)

    N = Vin.shape[0]

    # floor (same as MATLAB floor(double(Vin)))
    V = torch.floor(Vin).to(torch.int64)

    # Initialize morton code
    M = torch.zeros(N, dtype=torch.int64)

    # tt = [1, 2, 4]^T
    tt = torch.tensor([1, 2, 4], dtype=torch.int64)

    # Compute Morton code
    for i in range(1, J + 1):
        bits = torch.bitwise_and(torch.bitwise_right_shift(V, i - 1), 1)  # bitget(V, i)
        bits_flipped = torch.flip(bits, dims=[1])                         # fliplr
        M = M + torch.matmul(bits_flipped, tt)
        tt = tt * 8

    # Sort Morton code
    M_sorted, index = torch.sort(M)

    # Reorder points
    V_sorted = V[index, :]
    out = Vin[index, :]

    # Compute error (norm of difference)
    error = torch.norm(V.to(torch.float64) - V_sorted.to(torch.float64)).item()

    return error, out, index


def block_indices(V: torch.Tensor, bsize: int):
    """

    Args:
        V (torch.Tensor): Nx3 point cloud tensor (integer or float)
        bsize (int): block size

    Returns:
        indices (torch.Tensor): indices (1-based, like MATLAB)
    """
    # ensure tensor
    if not torch.is_tensor(V):
        V = torch.tensor(V, dtype=torch.float64)

    # Coarsen coordinates by block size
    V_coarse = torch.floor(V / bsize) * bsize  # Nx3

    # Compute absolute variation between consecutive points
    diff = torch.abs(V_coarse[1:] - V_coarse[:-1])  # (N-1)x3
    variation = torch.sum(diff, dim=1)  # (N-1,)

    # Prepend a leading 1 to mimic MATLAB behavior
    variation = torch.cat([torch.ones(1, dtype=variation.dtype), variation])

    # Find indices where variation ≠ 0
    indices = torch.nonzero(variation, as_tuple=True)[0]
    indices_remain = torch.nonzero(variation == 0, as_tuple=True)[0]

    return indices,indices_remain
