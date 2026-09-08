#pragma once

#include <cstddef>

void matmul_cuda_naive(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
);

void matmul_cuda_tiled(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
);

void matmul_cuda_register_tiled(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
);
