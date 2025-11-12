"""
Setup script for building the merge_cluster_cuda package.

Usage:
    pip install .                           # Regular installation
    pip install -e .                        # Editable/development installation
    python setup.py build_ext --inplace     # Build in-place only
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='merge_cluster_cuda',
    version='1.0.0',
    author='Your Name',
    description='CUDA extension for merging 3D Gaussian clusters',
    packages=find_packages(),  # Find the merge_cluster_cuda package
    ext_modules=[
        CUDAExtension(
            name='merge_cluster_cuda._C',
            sources=[
                'merge_cluster_wrapper.cu',
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
