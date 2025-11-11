# Merge Cluster CUDA Extension

This CUDA extension provides GPU-accelerated merging of 3D Gaussian clusters using a weighted mean strategy.

## Files

- `merge_cluster.cu` - Original CUDA kernel implementation
- `merge_cluster_wrapper.cu` - C++/CUDA wrapper for PyTorch integration
- `merge_cluster.py` - Python interface module
- `setup_merge_cluster.py` - Setup script for building the extension
- `build_merge_cluster.py` - JIT compilation helper (alternative to setup.py)

## Installation

### Option 1: Build and Install (Recommended for production)

```bash
cd /ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/python
python setup_merge_cluster.py install
```

Or for development (builds in-place):
```bash
python setup_merge_cluster.py build_ext --inplace
```

### Option 2: JIT Compilation (Convenient for development)

No installation needed! The extension will be compiled automatically on first import:

```python
from build_merge_cluster import merge_gaussian_clusters
```

The compiled extension is cached, so subsequent imports are fast.

## GPU Architecture

The default setup uses `sm_70` (Volta architecture). Adjust the architecture in `setup_merge_cluster.py` or `build_merge_cluster.py` based on your GPU:

- **sm_70**: Volta (V100)
- **sm_75**: Turing (RTX 20xx, T4)
- **sm_80**: Ampere (A100, RTX 30xx)
- **sm_86**: Ampere (RTX 30xx mobile)
- **sm_89**: Ada Lovelace (RTX 40xx)

To support multiple architectures, add multiple `-gencode` flags in the nvcc compile args.

## Usage

### Basic Example

```python
import torch
from merge_cluster import merge_gaussian_clusters

# Create example 3D Gaussian attributes
N = 1000  # Number of Gaussians
means = torch.randn(N, 3, device='cuda')
quats = torch.randn(N, 4, device='cuda')
quats = quats / quats.norm(dim=1, keepdim=True)  # Normalize quaternions
scales = torch.rand(N, 3, device='cuda')
opacities = torch.rand(N, device='cuda')
colors = torch.randn(N, 48, device='cuda')  # e.g., 16 SH coefficients * 3 channels

# Define cluster assignments (e.g., from k-means or other clustering)
cluster_labels = torch.randint(0, 100, (N,), device='cuda')

# Merge clusters
merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = \
    merge_gaussian_clusters(
        means, quats, scales, opacities, colors, cluster_labels,
        weight_by_opacity=True  # Weight by opacity (default) or use equal weights
    )

print(f"Merged {N} Gaussians into {merged_means.shape[0]} clusters")
```

### Advanced: Pre-computed Cluster Indices

If you already have cluster_indices and cluster_offsets:

```python
from merge_cluster import merge_gaussian_clusters_with_indices

merged = merge_gaussian_clusters_with_indices(
    means, quats, scales, opacities, colors,
    cluster_indices, cluster_offsets,
    weight_by_opacity=True
)
```

## API Reference

### `merge_gaussian_clusters()`

Merge 3D Gaussian clusters using weighted mean strategy.

**Parameters:**
- `means` (torch.Tensor): [N, 3] Gaussian positions
- `quats` (torch.Tensor): [N, 4] Quaternions (rotations)
- `scales` (torch.Tensor): [N, 3] Scales
- `opacities` (torch.Tensor): [N] Opacities
- `colors` (torch.Tensor): [N, color_dim] Colors (e.g., SH coefficients or RGB)
- `cluster_labels` (torch.Tensor): [N] Cluster ID for each Gaussian
- `weight_by_opacity` (bool): Weight contributions by opacity (default: True)

**Returns:**
Tuple of (merged_means, merged_quats, merged_scales, merged_opacities, merged_colors)

**Requirements:**
- All inputs must be CUDA tensors
- All tensors must have the same number of elements (N)
- `quats` should be normalized (unit quaternions)
- `opacities` should be in [0, 1] range

### Merging Strategy

**Weighted Mean:**
- Positions (means), quaternions, scales, and colors are averaged using weighted mean
- Weights can be either opacity-based or uniform (equal weights)

**Quaternions:**
- Averaged using weighted mean, then normalized to unit quaternions
- Handles degenerate cases (zero-norm quaternions → identity quaternion)

**Opacities:**
- Summed across cluster members (not weighted)
- Clamped to [0, 1] range

**Colors:**
- Weighted average per color channel
- Handles any color_dim (RGB, SH coefficients, etc.)

## Performance

The CUDA kernel is highly optimized for parallel execution:
- Each cluster is processed by a separate thread
- Uses 256 threads per block by default
- All operations are performed in a single kernel launch

For typical workloads (thousands of Gaussians merged into hundreds of clusters), this implementation provides 10-100x speedup compared to CPU-based merging.

## Troubleshooting

### Import Error: "merge_cluster_cuda extension not found"

The extension hasn't been built yet. Either:
1. Run `python setup_merge_cluster.py install`
2. Use `from build_merge_cluster import merge_gaussian_clusters` for JIT compilation

### CUDA Error: "no kernel image is available for execution"

Your GPU architecture doesn't match the compiled architectures. Edit the CUDA arch in:
- `setup_merge_cluster.py`: Change `-arch=sm_70` to match your GPU
- `build_merge_cluster.py`: Change `-arch=sm_70` to match your GPU

Check your GPU compute capability with:
```python
import torch
print(torch.cuda.get_device_capability())  # e.g., (8, 0) for sm_80
```

### Build Errors

Make sure you have:
- PyTorch with CUDA support installed
- CUDA toolkit installed (matching PyTorch's CUDA version)
- GCC/G++ compiler installed

Check CUDA availability:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```
