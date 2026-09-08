#pragma once

#include <vector>
#include <cstddef>

std::vector<float> matmul_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::size_t M,
    std::size_t K,
    std::size_t N
);
