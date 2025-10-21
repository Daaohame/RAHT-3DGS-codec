import torch
from scipy.io import savemat, loadmat

def rgb_to_yuv_torch(rgb_tensor):
    """Converts a PyTorch tensor of RGB colors [0,255] to YUV."""
    rgb_tensor = rgb_tensor.float()
    conversion_matrix = torch.torch.tensor([
        [0.2126, 0.7152, 0.0722],
        [-0.1146, -0.3854, 0.5000],
        [0.5000, -0.4542, -0.0458]
    ]).to(rgb_tensor.device)
    
    yuv = torch.matmul(rgb_tensor, conversion_matrix.T)
    yuv[:, 1:] += 128.0 # Add offset to U and V
    return yuv

def rgb_to_yuv_torch2(rgb_tensor):
    """Converts a PyTorch tensor of RGB colors [0,255] to YUV."""
    r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]
    Y = torch.clamp(torch.floor(0.212600 * r + 0.715200 * g + 0.072200 * b + 0.5), 0.0, 255.0)
    U = torch.clamp(torch.floor(-0.114572 * r - 0.385428 * g + 0.5 * b + 128.0 + 0.5), 0.0, 255.0)
    V = torch.clamp(torch.floor(0.5 * r - 0.454153 * g - 0.045847 * b + 128.0 + 0.5), 0.0, 255.0)
    return torch.stack([Y, U, V], dim=1)


def save_mat(tensor: torch.Tensor, filename: str) -> None:
    savemat(filename, {"data": tensor.detach().cpu().numpy()})

def save_lists(filename, **kwargs):
    out = {}
    for key, tensor_list in kwargs.items():
        out[key] = [t.detach().cpu().numpy() for t in tensor_list]
    savemat(filename, out)

def signed_to_unsigned(v):
    return torch.where(v >= 0, 2 * v, -2 * v - 1)

def unsigned_to_signed(u):
    torch.where(u % 2 == 0,
                torch.div(u, 2, rounding_mode='floor'),
                -torch.div(u, 2, rounding_mode='floor') - 1)