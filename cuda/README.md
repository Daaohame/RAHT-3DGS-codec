# Merge Cluster CUDA Extension

This CUDA extension provides GPU-accelerated merging of 3D Gaussian clusters using a weighted mean strategy.

## Project Structure

```
cuda/
├── merge_cluster.cu              # CUDA kernel implementation
├── merge_cluster_wrapper.cu      # PyTorch C++ wrapper
├── merge_cluster_cuda/           # Python package
│   └── __init__.py              # High-level API
├── setup.py                      # Build configuration
├── pyproject.toml               # Modern packaging config
└── README.md                    # This file
```

## Installation

### Build the CUDA Extension

Choose one of the following methods based on your needs:

#### Method 1: Install with pip (Recommended)
```bash
pip install .
```
- **What it does**: Builds and installs the package to Python's site-packages
- **When to use**: Production use, or when you want the package globally available
- **Location**: Installed in site-packages (e.g., `~/.local/lib/python3.x/site-packages/`)
- **Requirement**: Can import from anywhere after installation
- **Note**: Use `pip install -e .` for editable/development install (changes to source reflected without reinstall)

#### Method 2: Build In-Place (Development Only)
```bash
python setup.py build_ext --inplace
```
- **What it does**: Compiles the CUDA extension and places the `.so` file in the current directory
- **When to use**: Quick testing, when you want to work in this directory only
- **Location**: Extension is in the current directory only
- **Requirement**: You must run your code from this directory or add this directory to Python path

### Verify Installation

After installation, verify the package works:
```python
import merge_cluster_cuda
print("Successfully installed!")
print(f"Available functions: {[f for f in dir(merge_cluster_cuda) if not f.startswith('_')]}")
```

### What Gets Installed

After installation, you can import everything from the `merge_cluster_cuda` package:
- High-level functions: `merge_gaussian_clusters()`, `merge_gaussian_clusters_with_indices()`, `prepare_cluster_data()`
- Low-level CUDA extension: `merge_cluster_cuda._C` for direct kernel access

All functionality is accessible from the single `merge_cluster_cuda` import.

## GPU Architecture

The default setup uses `sm_86` (RTX A6000). Adjust the architecture in `setup.py` based on your GPU:

- **sm_70**: Volta (V100)
- **sm_75**: Turing (RTX 20xx, T4)
- **sm_80**: Ampere (A100, RTX 30xx)
- **sm_86**: Ampere (RTX 30xx mobile, RTX A6000)
- **sm_89**: Ada Lovelace (RTX 40xx)

To support multiple architectures, add multiple `-gencode` flags in the nvcc compile args.

## Usage

### Option 1: High-Level API (Recommended)

After installation, you can use the high-level API from anywhere:

```python
import torch
from merge_cluster_cuda import merge_gaussian_clusters

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

### Option 2: Low-Level CUDA API (Advanced)

For advanced use cases, you can access the CUDA extension directly from the same package:

```python
import torch
import merge_cluster_cuda

# Create example data
N = 1000
means = torch.randn(N, 3, device='cuda')
quats = torch.randn(N, 4, device='cuda')
quats = quats / quats.norm(dim=1, keepdim=True)
scales = torch.rand(N, 3, device='cuda')
opacities = torch.rand(N, device='cuda')
colors = torch.randn(N, 48, device='cuda')
cluster_labels = torch.randint(0, 100, (N,), device='cuda', dtype=torch.long)

# Prepare cluster data (sort and create offsets)
cluster_indices, cluster_offsets = merge_cluster_cuda.prepare_cluster_data(cluster_labels)

# Call CUDA kernel directly via merge_cluster_cuda._C
merged = merge_cluster_cuda._C.merge_clusters_cuda(
    cluster_indices,
    cluster_offsets,
    means.contiguous(),
    quats.contiguous(),
    scales.contiguous(),
    opacities.contiguous(),
    colors.contiguous(),
    True  # weight_by_opacity
)

merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = merged
print(f"Merged {N} Gaussians into {merged_means.shape[0]} clusters")
```

## API Reference

### High-Level API

All functions are accessible via `import merge_cluster_cuda`.

#### `merge_cluster_cuda.merge_gaussian_clusters()`

Merge 3D Gaussian clusters using weighted mean strategy. Handles cluster data preparation automatically.

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

### Low-Level API

#### `merge_cluster_cuda._C.merge_clusters_cuda()`

Direct CUDA kernel interface. Requires pre-sorted cluster data.

**Parameters:**
- `cluster_indices` (torch.Tensor): [total_clustered] int32 tensor of sorted Gaussian indices
- `cluster_offsets` (torch.Tensor): [num_clusters + 1] int32 tensor marking cluster boundaries
- `means` (torch.Tensor): [N, 3] float32 tensor of positions
- `quats` (torch.Tensor): [N, 4] float32 tensor of quaternions
- `scales` (torch.Tensor): [N, 3] float32 tensor of scales
- `opacities` (torch.Tensor): [N] float32 tensor of opacities
- `colors` (torch.Tensor): [N, color_dim] float32 tensor of colors
- `weight_by_opacity` (bool): Weight by opacity or use equal weights

**Returns:**
List of [merged_means, merged_quats, merged_scales, merged_opacities, merged_colors]

#### `merge_cluster_cuda.prepare_cluster_data()`

Helper function to prepare cluster data for the CUDA kernel.

**Parameters:**
- `cluster_labels` (torch.Tensor): [N] tensor of cluster IDs for each Gaussian

**Returns:**
Tuple of (cluster_indices, cluster_offsets) ready for CUDA kernel

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

### Import Error: "No module named 'merge_cluster_cuda'"

The package hasn't been installed yet. Install it:
```bash
# Recommended: use pip
pip install .

# Or: build in-place (for development only, requires running from this directory)
python setup.py build_ext --inplace
```

**Note:** If you used `build_ext --inplace`, the module is only available when running from this directory. Use `pip install .` for global availability.

### Import Error: "merge_cluster_cuda._C extension not found"

This warning appears if the CUDA extension wasn't built properly. Make sure you:
1. Have PyTorch with CUDA support installed
2. Have CUDA toolkit installed (matching PyTorch's CUDA version)
3. Have a C++ compiler (GCC/G++) installed

Then rebuild:
```bash
pip install . --force-reinstall --no-cache-dir
```

### CUDA Error: "no kernel image is available for execution"

Your GPU architecture doesn't match the compiled architectures. Edit the CUDA arch in `setup.py`:
- Change `-arch=sm_86` to match your GPU (e.g., `-arch=sm_80` for A100)

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
