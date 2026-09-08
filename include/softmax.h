#pragma once

#include <cstddef>
#include <vector>

std::vector<float> softmax_cpu(
    const std::vector<float>& input,
    std::size_t rows,
    std::size_t cols
);
