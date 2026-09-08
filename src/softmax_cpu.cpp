#include "softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

std::vector<float> softmax_cpu(
    const std::vector<float>& input,
    std::size_t rows,
    std::size_t cols
) {
    if (input.size() != rows * cols) {
        throw std::invalid_argument("Input has incorrect dimensions");
    }

    std::vector<float> output(input.size());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * cols;

        float max_value =
            -std::numeric_limits<float>::infinity();

        for (std::size_t col = 0; col < cols; ++col) {
            max_value = std::max(
                max_value,
                input[offset + col]
            );
        }

        float sum = 0.0f;

        for (std::size_t col = 0; col < cols; ++col) {
            const float value =
                std::exp(input[offset + col] - max_value);

            output[offset + col] = value;
            sum += value;
        }

        for (std::size_t col = 0; col < cols; ++col) {
            output[offset + col] /= sum;
        }
    }

    return output;
}
