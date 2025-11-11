// CUDA kernel: Merge clusters using weighted mean strategy
__global__ void merge_weighted_mean_kernel(
    const int* cluster_indices,     // [total_clustered] - flat indices
    const int* cluster_offsets,     // [num_clusters + 1] - boundaries
    int num_clusters,
    
    const float* means,             // [N, 3] - input Gaussians
    const float* quats,             // [N, 4]
    const float* scales,            // [N, 3] 
    const float* opacities,         // [N]
    const float* colors,            // [N, color_dim]
    int color_dim,
    bool weight_by_opacity,
    
    float* merged_means,            // [num_clusters, 3] - output
    float* merged_quats,            // [num_clusters, 4]
    float* merged_scales,           // [num_clusters, 3]
    float* merged_opacities,        // [num_clusters]
    float* merged_colors            // [num_clusters, color_dim]
) {
    int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id >= num_clusters) return;
    
    int start = cluster_offsets[cluster_id];
    int end = cluster_offsets[cluster_id + 1];
    int cluster_size = end - start;
    
    if (cluster_size == 0) return;
    
    // Initialize accumulators
    float3 mean_acc = {0.0f, 0.0f, 0.0f};
    float4 quat_acc = {0.0f, 0.0f, 0.0f, 0.0f};
    float3 scale_acc = {0.0f, 0.0f, 0.0f};
    float opacity_sum = 0.0f;
    float total_weight = 0.0f;
    
    // First pass: compute weights and accumulate weighted sums
    for (int i = start; i < end; i++) {
        int idx = cluster_indices[i];
        
        // Determine weight
        float weight = weight_by_opacity ? opacities[idx] : 1.0f;
        total_weight += weight;
        
        // Accumulate weighted sums
        mean_acc.x += means[idx * 3 + 0] * weight;
        mean_acc.y += means[idx * 3 + 1] * weight;
        mean_acc.z += means[idx * 3 + 2] * weight;
        
        quat_acc.x += quats[idx * 4 + 0] * weight;
        quat_acc.y += quats[idx * 4 + 1] * weight;
        quat_acc.z += quats[idx * 4 + 2] * weight;
        quat_acc.w += quats[idx * 4 + 3] * weight;
        
        scale_acc.x += scales[idx * 3 + 0] * weight;
        scale_acc.y += scales[idx * 3 + 1] * weight;
        scale_acc.z += scales[idx * 3 + 2] * weight;
        
        // Accumulate opacity (sum, not weighted)
        opacity_sum += opacities[idx];
    }
    
    // Avoid division by zero
    if (total_weight == 0.0f) {
        total_weight = 1.0f;
    }
    
    // Compute final weighted averages
    merged_means[cluster_id * 3 + 0] = mean_acc.x / total_weight;
    merged_means[cluster_id * 3 + 1] = mean_acc.y / total_weight;
    merged_means[cluster_id * 3 + 2] = mean_acc.z / total_weight;
    
    // Normalize quaternion
    float quat_norm = sqrtf(quat_acc.x * quat_acc.x + quat_acc.y * quat_acc.y + 
                           quat_acc.z * quat_acc.z + quat_acc.w * quat_acc.w);
    if (quat_norm > 0.0f) {
        merged_quats[cluster_id * 4 + 0] = quat_acc.x / quat_norm;
        merged_quats[cluster_id * 4 + 1] = quat_acc.y / quat_norm;
        merged_quats[cluster_id * 4 + 2] = quat_acc.z / quat_norm;
        merged_quats[cluster_id * 4 + 3] = quat_acc.w / quat_norm;
    } else {
        // Fallback to identity quaternion
        merged_quats[cluster_id * 4 + 0] = 0.0f;
        merged_quats[cluster_id * 4 + 1] = 0.0f;
        merged_quats[cluster_id * 4 + 2] = 0.0f;
        merged_quats[cluster_id * 4 + 3] = 1.0f;
    }
    
    merged_scales[cluster_id * 3 + 0] = scale_acc.x / total_weight;
    merged_scales[cluster_id * 3 + 1] = scale_acc.y / total_weight;
    merged_scales[cluster_id * 3 + 2] = scale_acc.z / total_weight;
    
    // Clamp opacity to [0, 1] (sum in linear space)
    merged_opacities[cluster_id] = fminf(opacity_sum, 1.0f);
    
    // Merge colors (weighted average)
    for (int c = 0; c < color_dim; c++) {
        float color_acc = 0.0f;
        float color_weight_sum = 0.0f;
        
        for (int i = start; i < end; i++) {
            int idx = cluster_indices[i];
            float weight = weight_by_opacity ? opacities[idx] : 1.0f;
            color_acc += colors[idx * color_dim + c] * weight;
            color_weight_sum += weight;
        }
        
        merged_colors[cluster_id * color_dim + c] = (color_weight_sum > 0.0f) ? 
            color_acc / color_weight_sum : 0.0f;
    }
}