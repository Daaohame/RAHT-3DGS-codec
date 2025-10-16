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

def rgb_to_yuv_2(rgb_tensor):
    """Converts a PyTorch tensor of RGB colors [0,255] to YUV using BT.709 full-range."""
    r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]
    Y = torch.clamp(torch.floor(0.212600 * r + 0.715200 * g + 0.072200 * b + 0.5), 0.0, 255.0)
    U = torch.clamp(torch.floor(-0.114572 * r - 0.385428 * g + 0.5 * b + 128.0 + 0.5), 0.0, 255.0)
    V = torch.clamp(torch.floor(0.5 * r - 0.454153 * g - 0.045847 * b + 128.0 + 0.5), 0.0, 255.0)
    return torch.stack([Y, U, V], dim=1)

# def rgb_to_yuv_2(rgb_tensor: torch.Tensor):
#     """Converts an Nx3 PyTorch tensor of RGB colors [0,255] to YUV using BT.601 full-range."""
#     r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]
#     Y = torch.clamp(torch.floor(0.299000 * r + 0.587000 * g + 0.114000 * b + 0.5), 0.0, 255.0)
#     U = torch.clamp(torch.floor(-0.168736 * r - 0.331264 * g + 0.500000 * b + 128.0 + 0.5), 0.0, 255.0)
#     V = torch.clamp(torch.floor( 0.500000 * r - 0.418688 * g - 0.081312 * b + 128.0 + 0.5), 0.0, 255.0)
#     return torch.stack([Y, U, V], dim=1)


def save_mat(tensor: torch.Tensor, filename: str) -> None:
    savemat(filename, {"data": tensor.detach().cpu().numpy()})

def save_lists(filename, **kwargs):
    out = {}
    for key, tensor_list in kwargs.items():
        out[key] = [t.detach().cpu().numpy() for t in tensor_list]
    savemat(filename, out)