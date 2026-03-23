#include <torch/extension.h>

torch::Tensor grid_subsampling(const torch::Tensor &pc_, const float grid_size_, torch::Tensor &hash_table, torch::Tensor &hash_storage);
torch::Tensor grid_subsampling_test(const torch::Tensor &pc_, const float grid_size_, torch::Tensor &hash_table, torch::Tensor &hash_storage, uint32_t ra);

void knn_edge_maxpooling_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
);

void aligned_knn_edge_maxpooling_forward(
    torch::Tensor &output,
    torch::Tensor &indices,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void aligned_knn_edge_maxpooling_infer(
    torch::Tensor &output,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void half_aligned_knn_edge_maxpooling_forward(
    torch::Tensor &output,
    torch::Tensor &indices,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void half_aligned_knn_edge_maxpooling_infer(
    torch::Tensor &output,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void half_knn_edge_maxpooling_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("grid_subsampling", &grid_subsampling);
    m.def("grid_subsampling_test", &grid_subsampling_test);
    m.def("knn_edge_maxpooling_backward", &knn_edge_maxpooling_backward);
    m.def("aligned_knn_edge_maxpooling_forward", &aligned_knn_edge_maxpooling_forward);
    m.def("aligned_knn_edge_maxpooling_infer", &aligned_knn_edge_maxpooling_infer);
    m.def("half_aligned_knn_edge_maxpooling_forward", &half_aligned_knn_edge_maxpooling_forward);
    m.def("half_aligned_knn_edge_maxpooling_infer", &half_aligned_knn_edge_maxpooling_infer);
    m.def("half_knn_edge_maxpooling_backward", &half_knn_edge_maxpooling_backward);
}

