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
    Y = torch.clamp(torch.round(0.212600 * r + 0.715200 * g + 0.072200 * b), 0.0, 255.0)
    U = torch.clamp(torch.round(-0.114572 * r - 0.385428 * g + 0.5 * b + 128.0), 0.0, 255.0)
    V = torch.clamp(torch.round(0.5 * r - 0.454153 * g - 0.045847 * b + 128.0), 0.0, 255.0)
    return torch.stack([Y, U, V], dim=1)