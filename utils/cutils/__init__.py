from pathlib import Path

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils.cpp_extension import load


# cutils contains cpp/cuda functions (neighbor edge-max-pooling, grid subsampling, knn search).


path = Path(__file__).parent
build_dir = path / "build"
build_dir.mkdir(exist_ok=True)
sources = [str(p) for p in path.glob("srcs/*.*") if p.suffix in [".cpp", ".cu"]]

cutils = load("cutils_", sources=sources, extra_cflags=["-O3", "-mavx2", "-funroll-loops"],
              extra_cuda_cflags=["-Xptxas", "-v"],
              verbose=True, build_directory=build_dir)


def next_prime(x) -> int:
    r"""
    Finds the next prime, x included.
    x should be >= 3 for a correct result.
    """
    x = int(x) | 1
    for i in range(x, 2 * x, 2):
        prime = True
        for j in range(3, int(i ** 0.5) + 1, 2):
            if i % j == 0:
                prime = False
                break
        if prime:
            return i


def grid_subsampling(xyz: torch.Tensor, grid_size: float, hash_size: float = 1.) -> torch.Tensor:
    r"""
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 3,), dtype=torch.int64)
    indices = cutils.grid_subsampling(xyz, grid_size, table, storage)
    return indices


def grid_subsampling_test(xyz: torch.Tensor, grid_size: float, hash_size: float = 1., pick=0) -> torch.Tensor:
    r"""
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    pick:  the nth point in the same grid to pick, random picked if actual resident points < pick
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 4,), dtype=torch.int64)
    indices = cutils.grid_subsampling_test(xyz, grid_size, table, storage, pick)
    return indices


class KEMP(Function):
    r"""
    f_i = max{f_j | j in knn_i} - f_i
    output = knn_edge_maxpooling(feature, knn, training=True)

    Only cuda version supported.

    feature: BNC, float / half
    knn:     BNk, int64
    output:  BNC, float / half

    While not training and gradient is not required,
    backward indices are not saved. Consumed time and space reduced slightly.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, feature: torch.Tensor, knn: torch.Tensor, training: bool = True) -> torch.Tensor:
        assert feature.is_cuda and knn.is_cuda
        assert feature.is_contiguous() and knn.is_contiguous() and feature.shape[:2] == knn.shape[:2]
        assert knn.dtype == torch.int64
        if feature.dtype == torch.half:
            assert feature.shape[-1] % 8 == 0, "KEMP half precision impl only supports multiples of 8 as feature dim"
        elif feature.dtype == torch.float32:
            assert feature.shape[-1] % 4 == 0, "KEMP single precision impl only supports multiples of 4 as feature dim"
        else:
            raise NotImplementedError

        output = torch.empty_like(feature)
        if training or feature.requires_grad:
            indices = torch.empty_like(feature, dtype=torch.int32)
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)
            else:
                cutils.aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)
            ctx.save_for_backward(indices)
        else:
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_infer(output, feature, knn)
            else:
                cutils.aligned_knn_edge_maxpooling_infer(output, feature, knn)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        output = -grad
        indices, = ctx.saved_tensors
        if grad.dtype == torch.half:
            cutils.half_knn_edge_maxpooling_backward(output, indices, grad)
        else:
            cutils.knn_edge_maxpooling_backward(output, indices, grad)
        return output, None, None


knn_edge_maxpooling = KEMP.apply

