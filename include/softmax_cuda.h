#pragma once

#include <cstddef>

void softmax_cuda_naive(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
);
