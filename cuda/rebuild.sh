#!/bin/bash
# Clean rebuild script for merge_cluster_cuda CUDA extension
# Use this after modifying .cu files to ensure the extension is rebuilt

set -e

echo "=================================================="
echo "Cleaning old build artifacts..."
echo "=================================================="

# Remove old compiled extensions
rm -f merge_cluster_cuda/_C.*.so

# Remove build directories
rm -rf build dist *.egg-info

echo ""
echo "=================================================="
echo "Rebuilding CUDA extension..."
echo "=================================================="

# Rebuild the extension in-place
python setup.py build_ext --inplace

echo ""
echo "=================================================="
echo "Testing import..."
echo "=================================================="

# Test that the extension loads
python -c "from merge_cluster_cuda import _C; print('✓ Extension loaded successfully')"

echo ""
echo "=================================================="
echo "Done! Extension rebuilt and ready to use."
echo "=================================================="
