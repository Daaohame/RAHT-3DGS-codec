"""
Setup script for building the merge_cluster CUDA extension.

Usage:
    pip install .                    # Regular installation
    pip install -e .                 # Editable/development installation
    python setup_merge_cluster.py build_ext --inplace  # Build in-place only
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory of this setup file
setup_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='merge_cluster_cuda',
    version='1.0.0',
    author='Your Name',
    description='CUDA extension for merging 3D Gaussian clusters',
    ext_modules=[
        CUDAExtension(
            name='merge_cluster_cuda',
            sources=[
                os.path.join(setup_dir, 'merge_cluster_wrapper.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_86',  # Adjust based on your GPU architecture
                    # '-gencode=arch=compute_70,code=sm_70',
                    # '-gencode=arch=compute_75,code=sm_75',
                    # '-gencode=arch=compute_80,code=sm_80',
                    # '-gencode=arch=compute_86,code=sm_86',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
