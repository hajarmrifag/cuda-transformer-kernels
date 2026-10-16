#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace validation {
inline std::size_t elements(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0 || rows > std::numeric_limits<std::size_t>::max() / cols
        || rows * cols > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("positive, representable float-array dimensions required");
    return rows * cols;
}
inline void softmax_shape(std::size_t rows, std::size_t cols, bool cached) {
    elements(rows, cols);
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("rows exceed CUDA grid.x limit");
    if (cached && cols > 2048)
        throw std::invalid_argument("cached softmax supports at most 2048 columns");
}
// Finite-input contract: NaN and infinity never compare equal, even to themselves.
inline bool close(const std::vector<float>& reference, const std::vector<float>& candidate,
                  double absolute_tolerance, double relative_tolerance = 0.0) {
    if (reference.size() != candidate.size() || !std::isfinite(absolute_tolerance)
        || !std::isfinite(relative_tolerance) || absolute_tolerance < 0 || relative_tolerance < 0)
        return false;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double a = reference[i], b = candidate[i];
        if (!std::isfinite(a) || !std::isfinite(b)
            || std::abs(a - b) > absolute_tolerance + relative_tolerance * std::abs(a)) return false;
    }
    return true;
}
}
