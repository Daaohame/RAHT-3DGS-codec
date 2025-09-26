import torch
import torch.nn.functional as F
from typing import List, Tuple, Union

def RAHT_optimized(C: torch.Tensor, 
                   List: List[torch.Tensor], 
                   Flags: List[torch.Tensor], 
                   weights: List[torch.Tensor],
                   device: Union[str, torch.device] = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fully GPU-optimized Region Adaptive Hierarchical Transform (RAHT)
    
    Args:
        C: Input coefficients tensor [N, number_of_attributes]
        List: List of node indices for each level
        Flags: List of binary flags indicating left siblings for each level
        weights: List of transform weights for each level
        device: Device to run computations on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (T, w) where:
        T: Transformed coefficients [N, number_of_attributes]
        w: Updated weights [N, 1]
    """
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize T and w
    T = C.clone().to(device)
    w = torch.ones(C.size(0), 1, device=device, dtype=C.dtype)
    
    Nlevels = len(Flags)
    
    for j in range(Nlevels):
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
            x0 = T[left_indices]   # [num_pairs, signal_dimension]
            x1 = T[right_indices]  # [num_pairs, signal_dimension]
            
            # Vectorized weight extraction
            w0 = weights_j[left_sibling_mask.bool()]
            w1 = weights_j[right_sibling_mask.bool()]
            
            # Compute transform coefficients in parallel
            w_sum = w0 + w1
            # Use rsqrt for better GPU performance
            inv_sqrt_w_sum = torch.rsqrt(w_sum)
            a = torch.sqrt(w0) * inv_sqrt_w_sum
            b = torch.sqrt(w1) * inv_sqrt_w_sum
            
            # Expand for broadcasting (more efficient than unsqueeze + expand)
            a_broad = a[:, None]  # [num_pairs, 1]
            b_broad = b[:, None]  # [num_pairs, 1]
            
            # Parallel RAHT transform computation
            new_x0 = a_broad * x0 + b_broad * x1
            new_x1 = -b_broad * x0 + a_broad * x1
            
            # Update T using advanced indexing (parallel writes)
            T[left_indices] = new_x0
            T[right_indices] = new_x1
            
            # Update weights in parallel
            w_left_old = w[left_indices, 0]
            w_right_old = w[right_indices, 0]
            w_new = w_left_old + w_right_old
            
            w[left_indices, 0] = w_new
            w[right_indices, 0] = w_new
    
    return T, w


def RAHT_batched(C: torch.Tensor, 
                 List: List[torch.Tensor], 
                 Flags: List[torch.Tensor], 
                 weights: List[torch.Tensor],
                 device: Union[str, torch.device] = 'cuda',
                 batch_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient batched version for very large datasets
    """
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    T = C.clone().to(device)
    w = torch.ones(C.size(0), 1, device=device, dtype=C.dtype)
    
    Nlevels = len(Flags)
    
    for j in range(Nlevels):
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
                x0_batch = T[left_batch]
                x1_batch = T[right_batch]
                
                w0_batch = weights_j[left_sibling_mask.bool()][batch_start:batch_end]
                w1_batch = weights_j[right_sibling_mask.bool()][batch_start:batch_end]
                
                # Transform computation
                w_sum_batch = w0_batch + w1_batch
                inv_sqrt_w_sum = torch.rsqrt(w_sum_batch)
                a_batch = torch.sqrt(w0_batch) * inv_sqrt_w_sum
                b_batch = torch.sqrt(w1_batch) * inv_sqrt_w_sum
                
                a_broad = a_batch[:, None]
                b_broad = b_batch[:, None]
                
                # Update T
                T[left_batch] = a_broad * x0_batch + b_broad * x1_batch
                T[right_batch] = -b_broad * x0_batch + a_broad * x1_batch
                
                # Update weights
                w_new_batch = w[left_batch, 0] + w[right_batch, 0]
                w[left_batch, 0] = w_new_batch
                w[right_batch, 0] = w_new_batch
    
    return T, w


def RAHT_fused_kernel(C: torch.Tensor, 
                      List: List[torch.Tensor], 
                      Flags: List[torch.Tensor], 
                      weights: List[torch.Tensor],
                      device: Union[str, torch.device] = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Version optimized for compilation with torch.compile for maximum performance
    """
    
    @torch.compile(dynamic=True)
    def _raht_level(T: torch.Tensor, w: torch.Tensor, 
                    left_mask: torch.Tensor, list_nodes: torch.Tensor, 
                    weights_level: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        right_mask = torch.cat([torch.zeros(1, dtype=left_mask.dtype, device=T.device),
                               left_mask[:-1]])
        
        left_indices = list_nodes[left_mask.bool()]
        right_indices = list_nodes[right_mask.bool()]
        
        if left_indices.numel() == 0:
            return T, w
        
        x0 = T[left_indices]
        x1 = T[right_indices]
        
        w0 = weights_level[left_mask.bool()]
        w1 = weights_level[right_mask.bool()]
        
        w_sum = w0 + w1
        inv_sqrt_w_sum = torch.rsqrt(w_sum)
        a = torch.sqrt(w0) * inv_sqrt_w_sum
        b = torch.sqrt(w1) * inv_sqrt_w_sum
        
        a_broad = a[:, None]
        b_broad = b[:, None]
        
        new_x0 = torch.addcmul(a_broad * x0, b_broad, x1)
        new_x1 = torch.addcmul(-b_broad * x0, a_broad, x1)
        
        T = T.clone()  # Needed for torch.compile
        T[left_indices] = new_x0
        T[right_indices] = new_x1
        
        w_new = w[left_indices, 0] + w[right_indices, 0]
        w = w.clone()  # Needed for torch.compile
        w[left_indices, 0] = w_new
        w[right_indices, 0] = w_new
        
        return T, w
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    T = C.to(device)
    w = torch.ones(C.size(0), 1, device=device, dtype=C.dtype)
    
    for j in range(len(Flags)):
        left_mask = Flags[j].to(device, non_blocking=True)
        list_nodes = List[j].to(device, non_blocking=True)
        weights_level = weights[j].to(device, non_blocking=True)
        
        T, w = _raht_level(T, w, left_mask, list_nodes, weights_level)
    
    return T, w


# Benchmarking and profiling utilities
def profile_raht_versions(C, List, Flags, weights, device='cuda', warmup_runs=5, benchmark_runs=10):
    """
    Benchmark different RAHT implementations
    """
    import time
    
    versions = {
        'optimized': RAHT_optimized,
        'batched': RAHT_batched,
        'fused': RAHT_fused_kernel
    }
    
    results = {}
    
    for name, func in versions.items():
        # Warmup
        for _ in range(warmup_runs):
            _ = func(C, List, Flags, weights, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = func(C, List, Flags, weights, device)
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


# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Large example for performance testing
    N = 65536  # Large number of coefficients
    num_attrs = 16
    
    C = torch.randn(N, num_attrs, device=device)
    
    # Generate hierarchical structure
    levels = 8
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
    
    # Run optimized version
    print("Running optimized RAHT...")
    breakpoint()
    T_opt, w_opt = RAHT_optimized(C, List, Flags, weights, device)
    print(f"Output shape: {T_opt.shape}, Memory usage: {T_opt.element_size() * T_opt.numel() / 1024**2:.2f} MB")
    
    # Benchmark if requested
    if len(List) > 0:
        print("\nBenchmarking different implementations...")
        benchmark_results = profile_raht_versions(C, List, Flags, weights, device)
        
        for name, stats in benchmark_results.items():
            print(f"{name:>10}: {stats['mean_time']*1000:.2f}ms (min: {stats['min_time']*1000:.2f}ms)")