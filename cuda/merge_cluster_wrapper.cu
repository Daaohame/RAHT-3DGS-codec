#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Include the kernel from merge_cluster.cu
#include "merge_cluster.cu"

// Host function to launch the kernel
std::vector<torch::Tensor> merge_clusters_cuda(
    torch::Tensor cluster_indices,      // [total_clustered] int32
    torch::Tensor cluster_offsets,      // [num_clusters + 1] int32
    torch::Tensor means,                // [N, 3] float32
    torch::Tensor quats,                // [N, 4] float32
    torch::Tensor scales,               // [N, 3] float32
    torch::Tensor opacities,            // [N] float32
    torch::Tensor colors,               // [N, color_dim] float32
    bool weight_by_opacity
) {
    // Input validation
    TORCH_CHECK(cluster_indices.is_cuda(), "cluster_indices must be a CUDA tensor");
    TORCH_CHECK(cluster_offsets.is_cuda(), "cluster_offsets must be a CUDA tensor");
    TORCH_CHECK(means.is_cuda(), "means must be a CUDA tensor");
    TORCH_CHECK(quats.is_cuda(), "quats must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(opacities.is_cuda(), "opacities must be a CUDA tensor");
    TORCH_CHECK(colors.is_cuda(), "colors must be a CUDA tensor");

    TORCH_CHECK(cluster_indices.dtype() == torch::kInt32, "cluster_indices must be int32");
    TORCH_CHECK(cluster_offsets.dtype() == torch::kInt32, "cluster_offsets must be int32");
    TORCH_CHECK(means.dtype() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(quats.dtype() == torch::kFloat32, "quats must be float32");
    TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(opacities.dtype() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(colors.dtype() == torch::kFloat32, "colors must be float32");

    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be [N, 3]");
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, "quats must be [N, 4]");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must be [N, 3]");
    TORCH_CHECK(opacities.dim() == 1, "opacities must be [N]");
    TORCH_CHECK(colors.dim() == 2, "colors must be [N, color_dim]");

    const int N = means.size(0);
    TORCH_CHECK(quats.size(0) == N, "quats must have same length as means");
    TORCH_CHECK(scales.size(0) == N, "scales must have same length as means");
    TORCH_CHECK(opacities.size(0) == N, "opacities must have same length as means");
    TORCH_CHECK(colors.size(0) == N, "colors must have same length as means");

    const int num_clusters = cluster_offsets.size(0) - 1;
    const int color_dim = colors.size(1);

    // Allocate output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(means.device());
    torch::Tensor merged_means = torch::zeros({num_clusters, 3}, options);
    torch::Tensor merged_quats = torch::zeros({num_clusters, 4}, options);
    torch::Tensor merged_scales = torch::zeros({num_clusters, 3}, options);
    torch::Tensor merged_opacities = torch::zeros({num_clusters}, options);
    torch::Tensor merged_colors = torch::zeros({num_clusters, color_dim}, options);

    // Get raw pointers
    const int* cluster_indices_ptr = cluster_indices.data_ptr<int>();
    const int* cluster_offsets_ptr = cluster_offsets.data_ptr<int>();
    const float* means_ptr = means.data_ptr<float>();
    const float* quats_ptr = quats.data_ptr<float>();
    const float* scales_ptr = scales.data_ptr<float>();
    const float* opacities_ptr = opacities.data_ptr<float>();
    const float* colors_ptr = colors.data_ptr<float>();

    float* merged_means_ptr = merged_means.data_ptr<float>();
    float* merged_quats_ptr = merged_quats.data_ptr<float>();
    float* merged_scales_ptr = merged_scales.data_ptr<float>();
    float* merged_opacities_ptr = merged_opacities.data_ptr<float>();
    float* merged_colors_ptr = merged_colors.data_ptr<float>();

    // Launch kernel
    const int threads = 256;
    const int blocks = (num_clusters + threads - 1) / threads;

    merge_weighted_mean_kernel<<<blocks, threads>>>(
        cluster_indices_ptr,
        cluster_offsets_ptr,
        num_clusters,
        means_ptr,
        quats_ptr,
        scales_ptr,
        opacities_ptr,
        colors_ptr,
        color_dim,
        weight_by_opacity,
        merged_means_ptr,
        merged_quats_ptr,
        merged_scales_ptr,
        merged_opacities_ptr,
        merged_colors_ptr
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    // Synchronize (optional, depending on usage)
    cudaDeviceSynchronize();

    return {merged_means, merged_quats, merged_scales, merged_opacities, merged_colors};
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("merge_clusters_cuda", &merge_clusters_cuda, "Merge 3D Gaussian clusters (CUDA)",
          py::arg("cluster_indices"),
          py::arg("cluster_offsets"),
          py::arg("means"),
          py::arg("quats"),
          py::arg("scales"),
          py::arg("opacities"),
          py::arg("colors"),
          py::arg("weight_by_opacity") = true);
}
