"""
Setup script for building the merge_cluster CUDA extension.

Usage:
    python setup_merge_cluster.py install
    or
    python setup_merge_cluster.py build_ext --inplace
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
                    '-arch=sm_70',  # Adjust based on your GPU architecture
                    # Add more architectures if needed:
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
