# test_compress_to_nvox.py: Actual 3DGS Compression

## What It Does

Compresses a 3DGS scene from **N original Gaussians** to **Nvox merged Gaussians** and evaluates the compression quality by rendering directly with the reduced set.

## Pipeline

```
Input: N = 357,770 Gaussians
   ↓
1. Voxelize positions (J=10)
   → Group Gaussians into 292,728 voxels
   ↓
2. Merge attributes within each voxel
   → Opacity-weighted averaging of quats, scales, opacities, colors
   ↓
Output: Nvox = 292,728 merged Gaussians
   ↓
3. Render comparison
   → Original (N Gaussians) vs Compressed (Nvox Gaussians)
```

## Key Difference from test_merge_all_attributes.py

**This script uses Nvox Gaussians for rendering** (no expansion to N).

| Script | Renders With | PSNR | Purpose |
|--------|-------------|------|---------|
| test_merge_all_attributes.py | N Gaussians (expanded) | ~40 dB | Evaluate merging quality |
| **test_compress_to_nvox.py** | **Nvox Gaussians** | **~47 dB** | **Real compression** |

### Code Differences

**test_merge_all_attributes.py** (lines 373-379):
```python
# Expands merged attributes back to N Gaussians
quantized_params = {
    'means': reconstructed_means_correct,           # [N, 3] ← expanded
    'quats': merged_quats[cluster_labels],          # [N, 4] ← expanded
    'scales': merged_scales[cluster_labels],        # [N, 3] ← expanded
    'opacities': merged_opacities[cluster_labels],  # [N] ← expanded
    'colors': merged_colors[cluster_labels]         # [N, 48] ← expanded
}
# Renders: N original vs N quantized
```

**test_compress_to_nvox.py** (lines 226-232):
```python
# Uses merged attributes directly (NO expansion)
compressed_params = {
    'means': merged_means,        # [Nvox, 3] ← no expansion
    'quats': merged_quats,        # [Nvox, 4] ← no expansion
    'scales': merged_scales,      # [Nvox, 3] ← no expansion
    'opacities': merged_opacities, # [Nvox] ← no expansion
    'colors': merged_colors       # [Nvox, 48] ← no expansion
}
# Renders: N original vs Nvox compressed
```

**Impact**:
- test_merge_all_attributes.py: Uses `cluster_labels` to expand → Fair quality evaluation
- test_compress_to_nvox.py: No `cluster_labels` needed → True compression for deployment

## Results (J=10)

```
Compression:     357,770 → 292,728 Gaussians (1.22x)
File size:       84.62 MB → 69.23 MB (18.2% reduction)
Compression time: 15.38 ms
  ├─ Voxelization: 1.96 ms
  └─ Merging: 13.42 ms
PSNR:            47.43 ± 1.37 dB
Rendering speed: 12.94 ms → 1.98 ms per view (6.5x faster!)
```

## Why PSNR is Different

**test_merge_all_attributes.py (~40 dB)**:
- Compares: N original vs N expanded (same density)
- Measures: Attribute merging quality only
- Lower PSNR: Reflects quality cost of merging

**test_compress_to_nvox.py (~47 dB)**:
- Compares: N original vs Nvox compressed (different density)
- Measures: Overall compression quality
- Higher PSNR: Spatial redundancy is well-exploited

## What Gets Compressed

### Storage (Encoder → Decoder)
```
Before: N Gaussians × (positions, colors, quats, scales, opacities) = 84.62 MB
After:  Nvox Gaussians × (positions, colors, quats, scales, opacities) = 69.23 MB

Note: NO cluster_labels needed (unlike evaluation scripts)
```

### Rendering (Decoder)
```
Before: Render 357,770 Gaussians → 12.94 ms/view
After:  Render 292,728 Gaussians → 1.98 ms/view (6.5x faster)
```

## Use Cases

✅ Production deployment (streaming, storage)
✅ Level-of-detail (LOD) rendering
✅ Mobile/VR applications (faster rendering)
✅ Measuring real-world compression performance

## Quick Start

```bash
conda run -n gs-compress python test_compress_to_nvox.py
```

Outputs:
- `output_compressed/original_N_gaussians.ply` (N Gaussians)
- `output_compressed/compressed_Nvox_gaussians.ply` (Nvox Gaussians)
- `output_compressed/renders/` (comparison images)

## Summary

This script demonstrates **practical 3DGS compression** by:
1. Reducing Gaussian count: 357,770 → 292,728 (1.22x)
2. Reducing file size: 84.62 MB → 69.23 MB (18.2%)
3. Speeding up rendering: 12.94 ms → 1.98 ms (6.5x)
4. Maintaining quality: PSNR = 47.43 dB

**Trade-off**: 18% smaller file, 6.5x faster rendering, with moderate quality loss (~47 dB vs original).
