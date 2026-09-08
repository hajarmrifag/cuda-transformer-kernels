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
