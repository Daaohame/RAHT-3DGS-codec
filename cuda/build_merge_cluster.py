"""
Just-in-time compilation helper for merge_cluster CUDA extension.

This provides an alternative to setup.py that compiles the extension on-the-fly
the first time it's imported. This is convenient for development.

Usage:
    from build_merge_cluster import merge_gaussian_clusters
    # The extension will be compiled automatically on first import
"""

import os
import torch
from torch.utils.cpp_extension import load

# Get the directory of this file
_current_dir = os.path.dirname(os.path.abspath(__file__))

# JIT compile the extension
merge_cluster_cuda = load(
    name='merge_cluster_cuda',
    sources=[
        os.path.join(_current_dir, 'merge_cluster_wrapper.cu'),
    ],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-arch=sm_70',  # Adjust based on your GPU architecture
    ],
    verbose=True
)

# Import the high-level interface
from merge_cluster import (
    merge_gaussian_clusters,
    merge_gaussian_clusters_with_indices,
    prepare_cluster_data
)

__all__ = [
    'merge_gaussian_clusters',
    'merge_gaussian_clusters_with_indices',
    'prepare_cluster_data',
    'merge_cluster_cuda'
]
