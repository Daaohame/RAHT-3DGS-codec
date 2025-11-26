# Point Cloud Voxelization Guide

## 1. Function

Voxelization converts a **continuous point cloud** into a **discrete voxel grid**:

- **Input**: N points with float coordinates `[x, y, z]`
- **Output**: Nvox voxels with integer coordinates `[i, j, k]` where Nvox ≤ N

Multiple points falling into the same voxel are **merged** (attributes averaged). This enables:
- **Compression**: Fewer voxels than points
- **RAHT encoding**: Integer coords for octree-based transform
- **Morton ordering**: Required for RAHT octree processing and efficient voxel grouping

---

## 2. Example (Positions Only)

**Input**: 8 points in 3D space, `J=2` (grid is `2²=4` divisions per axis)

```
Points (continuous):
  P0: [0.1, 0.1, 0.1]
  P1: [0.9, 0.9, 0.9]
  P2: [0.15, 0.12, 0.08]  ← same voxel as P0
  P3: [0.5, 0.5, 0.5]
  P4: [0.52, 0.48, 0.51]
  P5: [0.3, 0.7, 0.2]
  P6: [0.85, 0.92, 0.88]  ← same voxel as P1
  P7: [0.0, 0.0, 0.0]     ← same voxel as P0
```

**Step 1: Quantize to Integer Voxel Coordinates**

```
voxel_size = width / 2^J = 1.0 / 4 = 0.25

V_integer = floor(V / voxel_size)

  P0: floor([0.1, 0.1, 0.1] / 0.25)   = [0, 0, 0]
  P1: floor([0.9, 0.9, 0.9] / 0.25)   = [3, 3, 3]
  P2: floor([0.15, 0.12, 0.08] / 0.25)= [0, 0, 0]  ← same as P0
  P3: floor([0.5, 0.5, 0.5] / 0.25)   = [2, 2, 2]
  P4: floor([0.52, 0.48, 0.51] / 0.25)= [2, 1, 2]
  P5: floor([0.3, 0.7, 0.2] / 0.25)   = [1, 2, 0]
  P6: floor([0.85, 0.92, 0.88] / 0.25)= [3, 3, 3]  ← same as P1
  P7: floor([0.0, 0.0, 0.0] / 0.25)   = [0, 0, 0]  ← same as P0
```

**Step 2: Compute Morton Codes & Sort**

Morton codes map 3D voxel coordinates to 1D integers via bit interleaving (Z-order curve).
This serves three purposes:
1. **Unique voxel IDs**: Points with same Morton code → same voxel
2. **RAHT requirement**: Downstream compression needs Morton-ordered octree traversal
3. **Efficient grouping**: Sorting groups identical voxels consecutively for O(N) boundary detection

```
Morton codes (interleave x,y,z bits → single integer):
  [0,0,0] → 0   (P0, P2, P7)
  [1,2,0] → 9   (P5)
  [2,1,2] → 22  (P4)
  [2,2,2] → 42  (P3)
  [3,3,3] → 63  (P1, P6)

Sorted order by Morton code:
  sort_idx = [0, 2, 7, 5, 4, 3, 1, 6]
              ───────  ─  ─  ─  ────
              voxel 0  1  2  3    4
```

**Step 3: Find Voxel Boundaries**

```
voxel_boundary = M_sort[1:] - M_sort[:-1]  # Detect consecutive Morton code changes
                = [0, 0, 9, 13, 20, 21, 0]
                         ^  ^   ^   ^ 
                        NEW NEW NEW NEW

voxel_indices = [0, 3, 4, 5, 6]  ← start index of each voxel in sorted array
Nvox = 5 unique voxels
```

**Output**

```python
PCvox = [[0, 0, 0],   # Voxel 0
         [1, 2, 0],   # Voxel 1
         [2, 1, 2],   # Voxel 2
         [2, 2, 2],   # Voxel 3
         [3, 3, 3]]   # Voxel 4

voxel_indices = [0, 3, 4, 5, 6]
sort_idx = [0, 2, 7, 5, 4, 3, 1, 6]
```

**Result**: 8 points → 5 voxels (1.6x compression)

---

## 3. Detailed Pipeline

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `PC` | `[N, 3+d]` tensor | Point cloud: 3 coords + d attributes (e.g., RGB) |
| `J` | int | Octree depth (grid = 2^J per axis) |
| `vmin` | `[3]` tensor | (optional) Bounding box minimum |
| `width` | float | (optional) Bounding box size |

### Processing Steps

```
INPUT: PC [N, 3+d]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 1: NORMALIZE                                            │
│   V0 = V - vmin                    # Shift to origin         │
│   voxel_size = width / 2^J         # Compute voxel size      │
│   V0_integer = floor(V0 / voxel_size)  # Quantize to int     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   V0_integer [N, 3] (int)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 2: MORTON CODE                                          │
│   M = interleave_bits(x, y, z)     # Z-order curve           │
│   M_sort, idx = sort(M)            # Sort by Morton code     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   idx [N] (sort permutation), M_sort [N] (sorted Morton codes)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 3: FIND VOXEL BOUNDARIES                                │
│   voxel_boundary = M_sort[1:] - M_sort[:-1]                  │
│   voxel_indices = where(voxel_boundary != 0)                 │
│   Nvox = len(voxel_indices)                                  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   voxel_indices [Nvox] (start of each voxel in sorted array)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 4: AVERAGE ATTRIBUTES (if present)                      │
│   voxel_id = repeat_interleave(arange(Nvox), counts)         │
│   C_sum = scatter_add(C_sorted, voxel_id)                    │
│   Cvox = C_sum / counts                                      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   Cvox [Nvox, d] (averaged attributes per voxel)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 5: BUILD OUTPUTS                                        │
│   Vvox = V0_integer[idx][voxel_indices]  # Voxel int coords  │
│   PCvox = concat(Vvox, Cvox)             # Voxelized PC      │
│   DeltaPC = PC_sorted - voxelized_values # Quantization err  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
OUTPUT: PCvox, PCsorted, voxel_indices, DeltaPC, info
```

### Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `PCvox` | `[Nvox, 3+d]` | Voxelized point cloud (int coords + avg attrs) |
| `PCsorted` | `[N, 3+d]` | Original PC sorted by Morton code |
| `voxel_indices` | `[Nvox]` | Start index of each voxel in sorted array |
| `DeltaPC` | `[N, 3+d]` | Quantization error for lossless reconstruction |
| `info` | dict | Metadata: `Nvox`, `voxel_size`, `vmin`, `width`, `sort_idx` |

### Key Relationships

```
# Points in voxel v (in sorted order):
start = voxel_indices[v]
end = voxel_indices[v+1] if v < Nvox-1 else N
points_in_voxel = PCsorted[start:end]

# Map sorted index → original index:
original_idx = sort_idx[sorted_idx]

# Reconstruct original point i:
original_pos = voxel_center + DeltaPC[i, :3]
original_attr = voxel_avg_attr + DeltaPC[i, 3:]
```

---

## 4. Optimizations

### 4.1 Efficient Morton Code (Bitwise Operations)

**Before** (slow tensor operations):
```python
for i in range(J):
    bits = torch.stack([...])
    M = M + (bits * tt.unsqueeze(0)).sum(dim=1) * (8 ** i)
```

**After** (fast bitwise):
```python
for i in range(1, J + 1):
    b = (V >> (i - 1)) & 1
    digit = b[:, 2] + (b[:, 1] << 1) + (b[:, 0] << 2)
    MC |= (digit << (3 * (i - 1)))
```

**Speedup**: 1.6x

### 4.2 Vectorized Voxel ID Assignment

**Before** (Python loop - extremely slow):
```python
voxel_id = torch.zeros(N, device=device)
for i in range(Nvox):  # 50k iterations in Python!
    start = voxel_indices[i]
    end = voxel_indices[i + 1] if i < Nvox - 1 else N
    voxel_id[start:end] = i
```

**After** (GPU-parallelized):
```python
voxel_counts = torch.diff(torch.cat([voxel_indices, torch.tensor([N])]))
voxel_id = torch.repeat_interleave(torch.arange(Nvox), voxel_counts)
```

**Speedup**: 100x+

### 4.3 Vectorized Attribute Averaging

**Before** (Python loop):
```python
for i in range(Nvox):
    start, end = voxel_indices[i], voxel_indices[i + 1]
    C0_voxelized[start:end] = C0[start:end].mean(dim=0)
```

**After** (GPU scatter_add):
```python
C_sum = torch.zeros(Nvox, d, device=device)
C_sum.scatter_add_(0, voxel_id.unsqueeze(1).expand(-1, d), C0)
Cvox = C_sum / voxel_counts.unsqueeze(1)
C0_voxelized = Cvox[voxel_id]  # Broadcast back
```

**Speedup**: Fully GPU-parallelized

### Performance Summary

| Points | Time (with warmup) | Throughput |
|--------|-------------------|------------|
| 50,000 | ~1.5 ms | 33 Mpts/s |
| 500,000 | ~3.3 ms | 150 Mpts/s |

---

## Usage Example

```python
import torch
from voxelize_pc import voxelize_pc_batched

# Create point cloud [N, 6] = [x, y, z, R, G, B]
pc_data = torch.rand(50000, 6, device='cuda')

# Voxelize
PCvox, PCsorted, voxel_indices, DeltaPC, info = voxelize_pc_batched(
    pc_data, J=8, device='cuda'
)

print(f"Input: {info['N']} points → Output: {info['Nvox']} voxels")
print(f"Compression: {info['N'] / info['Nvox']:.2f}x")
```
