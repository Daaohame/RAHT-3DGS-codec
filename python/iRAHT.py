import torch
import torch.nn.functional as F
from typing import List, Tuple, Union

def inverse_RAHT(Coeff: torch.Tensor, List, Flags, weights):
    """
    GPU/CPU compatible version of inverse RAHT transform.
    """
    device = Coeff.device

    T = Coeff.clone().to(device)
    Nlevels = len(Flags)

    for j in range(Nlevels - 1, -1, -1):
        left_sibling_index = Flags[j].to(device)
        right_sibling_index = torch.cat([
            torch.zeros(1, dtype=left_sibling_index.dtype, device=device),
            left_sibling_index[:-1]
        ])

        i0 = List[j][left_sibling_index == 1].to(device)
        i1 = List[j][right_sibling_index == 1].to(device)

        x0 = T[i0, :]
        x1 = T[i1, :]
        signal_dimension = T.shape[1]

        w0 = weights[j][left_sibling_index == 1].to(device)
        w1 = weights[j][right_sibling_index == 1].to(device)

        a = torch.sqrt(w0 / (w0 + w1)).unsqueeze(1).expand(-1, signal_dimension)
        b = torch.sqrt(w1 / (w0 + w1)).unsqueeze(1).expand(-1, signal_dimension)

        T[i0, :] = a * x0 - b * x1
        T[i1, :] = b * x0 + a * x1

    return T


def inverse_RAHT_optimized(T: torch.Tensor, 
                           List: List[torch.Tensor], 
                           Flags: List[torch.Tensor], 
                           weights: List[torch.Tensor],
                           device: Union[str, torch.device] = 'cuda') -> torch.Tensor:
    """
    Fully GPU-optimized Inverse Region Adaptive Hierarchical Transform (inverse-RAHT)
    
    Args:
        T: Transformed coefficients tensor [N, number_of_attributes]
        w: Weight tensor [N, 1] (usually ignored in reconstruction)
        List: List of node indices for each level
        Flags: List of binary flags indicating left siblings for each level
        weights: List of transform weights for each level (original weights)
        device: Device to run computations on ('cuda' or 'cpu')
    
    Returns:
        C: Reconstructed coefficients [N, number_of_attributes]
    """
    # Initialize reconstruction with transformed coefficients
    C = T.clone().to(device)
    Nlevels = len(Flags)
    
    # Top-down reconstruction (reverse order of forward transform)
    for j in reversed(range(Nlevels)):
        # Move all data to device at once
        left_sibling_mask = Flags[j].to(device, non_blocking=True)
        list_j = List[j].to(device, non_blocking=True)
        weights_j = weights[j].to(device, non_blocking=True)
        
        # Create right sibling mask using GPU operations
        right_sibling_mask = torch.cat([
            torch.zeros(1, dtype=left_sibling_mask.dtype, device=device),
            left_sibling_mask[:-1]
        ])
        
        # Get sibling pairs using vectorized operations
        left_indices = list_j[left_sibling_mask.bool()]
        right_indices = list_j[right_sibling_mask.bool()]
        
        if left_indices.numel() > 0 and right_indices.numel() > 0:
            # Vectorized coefficient extraction
            y0 = C[left_indices]   # Low-frequency components
            y1 = C[right_indices]  # High-frequency components
            
            # Vectorized weight extraction (original weights)
            w0 = weights_j[left_sibling_mask.bool()]
            w1 = weights_j[right_sibling_mask.bool()]
            
            # Compute inverse transform coefficients in parallel
            w_sum = w0 + w1
            # Use rsqrt for better GPU performance
            inv_sqrt_w_sum = torch.rsqrt(w_sum)
            a = torch.sqrt(w0) * inv_sqrt_w_sum
            b = torch.sqrt(w1) * inv_sqrt_w_sum
            
            # Expand for broadcasting
            a_broad = a[:, None]  # [num_pairs, 1]
            b_broad = b[:, None]  # [num_pairs, 1]
            
            # Inverse RAHT transform computation
            # Original: x0 = a*y0 - b*y1, x1 = b*y0 + a*y1
            new_x0 = a_broad * y0 - b_broad * y1
            new_x1 = b_broad * y0 + a_broad * y1
            
            # Update C using advanced indexing (parallel writes)
            C[left_indices] = new_x0
            C[right_indices] = new_x1
    
    return C


def inverse_RAHT_batched(T: torch.Tensor,
                         w: torch.Tensor,
                         List: List[torch.Tensor], 
                         Flags: List[torch.Tensor], 
                         weights: List[torch.Tensor],
                         device: Union[str, torch.device] = 'cuda',
                         batch_size: int = 10000) -> torch.Tensor:
    """
    Memory-efficient batched inverse RAHT for very large datasets
    """
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    C = T.clone().to(device)
    Nlevels = len(Flags)
    
    # Top-down reconstruction (reverse order)
    for j in reversed(range(Nlevels)):
        left_sibling_mask = Flags[j].to(device, non_blocking=True)
        list_j = List[j].to(device, non_blocking=True)
        weights_j = weights[j].to(device, non_blocking=True)
        
        right_sibling_mask = torch.cat([
            torch.zeros(1, dtype=left_sibling_mask.dtype, device=device),
            left_sibling_mask[:-1]
        ])
        
        left_indices = list_j[left_sibling_mask.bool()]
        right_indices = list_j[right_sibling_mask.bool()]
        
        if left_indices.numel() > 0:
            # Process in batches to manage memory
            num_pairs = left_indices.size(0)
            
            for batch_start in range(0, num_pairs, batch_size):
                batch_end = min(batch_start + batch_size, num_pairs)
                
                # Batch indices
                left_batch = left_indices[batch_start:batch_end]
                right_batch = right_indices[batch_start:batch_end]
                
                # Batch processing
                y0_batch = C[left_batch]
                y1_batch = C[right_batch]
                
                w0_batch = weights_j[left_sibling_mask.bool()][batch_start:batch_end]
                w1_batch = weights_j[right_sibling_mask.bool()][batch_start:batch_end]
                
                # Inverse transform computation
                w_sum_batch = w0_batch + w1_batch
                inv_sqrt_w_sum = torch.rsqrt(w_sum_batch)
                a_batch = torch.sqrt(w0_batch) * inv_sqrt_w_sum
                b_batch = torch.sqrt(w1_batch) * inv_sqrt_w_sum
                
                a_broad = a_batch[:, None]
                b_broad = b_batch[:, None]
                
                # Inverse RAHT transform
                C[left_batch] = a_broad * y0_batch - b_broad * y1_batch
                C[right_batch] = b_broad * y0_batch + a_broad * y1_batch
    
    return C


def inverse_RAHT_fused_kernel(T: torch.Tensor,
                              w: torch.Tensor,
                              List: List[torch.Tensor], 
                              Flags: List[torch.Tensor], 
                              weights: List[torch.Tensor],
                              device: Union[str, torch.device] = 'cuda') -> torch.Tensor:
    """
    Version optimized for compilation with torch.compile for maximum performance
    """
    
    @torch.compile(dynamic=True)
    def _inverse_raht_level(C: torch.Tensor, 
                           left_mask: torch.Tensor, 
                           list_nodes: torch.Tensor, 
                           weights_level: torch.Tensor) -> torch.Tensor:
        
        right_mask = torch.cat([torch.zeros(1, dtype=left_mask.dtype, device=C.device),
                               left_mask[:-1]])
        
        left_indices = list_nodes[left_mask.bool()]
        right_indices = list_nodes[right_mask.bool()]
        
        if left_indices.numel() == 0:
            return C
        
        y0 = C[left_indices]
        y1 = C[right_indices]
        
        w0 = weights_level[left_mask.bool()]
        w1 = weights_level[right_mask.bool()]
        
        w_sum = w0 + w1
        inv_sqrt_w_sum = torch.rsqrt(w_sum)
        a = torch.sqrt(w0) * inv_sqrt_w_sum
        b = torch.sqrt(w1) * inv_sqrt_w_sum
        
        a_broad = a[:, None]
        b_broad = b[:, None]
        
        # Inverse transform with fused operations
        new_x0 = torch.addcmul(a_broad * y0, -b_broad, y1)
        new_x1 = torch.addcmul(b_broad * y0, a_broad, y1)
        
        C = C.clone()  # Needed for torch.compile
        C[left_indices] = new_x0
        C[right_indices] = new_x1
        
        return C
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    C = T.to(device)
    Nlevels = len(Flags)
    
    # Top-down reconstruction
    for j in reversed(range(Nlevels)):
        left_mask = Flags[j].to(device, non_blocking=True)
        list_nodes = List[j].to(device, non_blocking=True)
        weights_level = weights[j].to(device, non_blocking=True)
        
        C = _inverse_raht_level(C, left_mask, list_nodes, weights_level)
    
    return C


def RAHT_round_trip_test(C_original: torch.Tensor,
                         List: List[torch.Tensor], 
                         Flags: List[torch.Tensor], 
                         weights: List[torch.Tensor],
                         device: Union[str, torch.device] = 'cuda',
                         tolerance: float = 1e-6) -> dict:
    """
    Test round-trip accuracy: C -> RAHT -> inverse-RAHT -> C'
    Returns reconstruction error statistics
    """
    from raht_pytorch import RAHT_optimized  # Import forward RAHT
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Forward transform
    T, w = RAHT_optimized(C_original, List, Flags, weights, device)
    
    # Inverse transform
    C_reconstructed = inverse_RAHT_optimized(T, w, List, Flags, weights, device)
    
    # Compute reconstruction error
    error = torch.abs(C_original - C_reconstructed)
    max_error = torch.max(error).item()
    mean_error = torch.mean(error).item()
    mse = torch.mean(error ** 2).item()
    
    # Check if reconstruction is within tolerance
    is_perfect = max_error < tolerance
    
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'mse': mse,
        'is_perfect_reconstruction': is_perfect,
        'tolerance': tolerance,
        'original_shape': C_original.shape,
        'reconstructed_shape': C_reconstructed.shape
    }


def benchmark_inverse_raht_versions(T, w, List, Flags, weights, device='cuda', 
                                   warmup_runs=5, benchmark_runs=10):
    """
    Benchmark different inverse RAHT implementations
    """
    import time
    
    versions = {
        'optimized': inverse_RAHT_optimized,
        'batched': inverse_RAHT_batched,
        'fused': inverse_RAHT_fused_kernel
    }
    
    results = {}
    
    for name, func in versions.items():
        # Warmup
        for _ in range(warmup_runs):
            _ = func(T, w, List, Flags, weights, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = func(T, w, List, Flags, weights, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        results[name] = {
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times)
        }
    
    return results


# Example usage and testing
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test with moderately sized example
    N = 16384
    num_attrs = 8
    
    # Generate test data
    C_original = torch.randn(N, num_attrs, device=device)
    
    # Generate hierarchical structure
    levels = 6
    List = []
    Flags = []
    weights = []
    
    current_size = N
    for level in range(levels):
        List.append(torch.arange(current_size, device=device))
        # Alternate pattern for siblings
        flag = torch.zeros(current_size, device=device)
        flag[::2] = 1
        Flags.append(flag)
        weights.append(torch.rand(current_size, device=device) + 0.1)
        current_size = current_size // 2
        if current_size < 2:
            break
    
    print(f"Testing with {N} coefficients, {num_attrs} attributes, {len(List)} levels")
    
    # Test round-trip reconstruction
    print("\n=== Round-trip Reconstruction Test ===")
    try:
        # You'll need to have the forward RAHT available for this test
        # For now, let's create dummy transformed data
        T_dummy = torch.randn_like(C_original)
        w_dummy = torch.ones(C_original.size(0), 1, device=device)
        
        print("Running inverse RAHT (optimized version)...")
        C_reconstructed = inverse_RAHT_optimized(T_dummy, w_dummy, List, Flags, weights, device)
        print(f"Reconstruction completed. Output shape: {C_reconstructed.shape}")
        
        # Test all versions
        print("\n=== Testing All Versions ===")
        versions = {
            'optimized': inverse_RAHT_optimized,
            'batched': inverse_RAHT_batched,
            'fused': inverse_RAHT_fused_kernel
        }
        
        for name, func in versions.items():
            try:
                result = func(T_dummy, w_dummy, List, Flags, weights, device)
                print(f"{name:>10}: Success - Shape {result.shape}")
            except Exception as e:
                print(f"{name:>10}: Error - {str(e)}")
        
        # Benchmark if all versions work
        print("\n=== Performance Benchmark ===")
        benchmark_results = benchmark_inverse_raht_versions(
            T_dummy, w_dummy, List, Flags, weights, device
        )
        
        for name, stats in benchmark_results.items():
            print(f"{name:>10}: {stats['mean_time']*1000:.2f}ms "
                  f"(min: {stats['min_time']*1000:.2f}ms, "
                  f"max: {stats['max_time']*1000:.2f}ms)")
        
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        print("Make sure the forward RAHT module is available for round-trip testing")
    
    print("\n=== Memory Usage ===")
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Peak GPU memory usage: {memory_used:.2f} MB")
        torch.cuda.reset_peak_memory_stats(device)