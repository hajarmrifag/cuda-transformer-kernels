#include "softmax.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

std::vector<float> random_input(
    std::size_t rows,
    std::size_t cols,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> data(rows * cols);

    for (auto& value : data) {
        value = dist(rng);
    }

    return data;
}

bool validate_softmax(
    const std::vector<float>& output,
    std::size_t rows,
    std::size_t cols
) {
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;

        for (std::size_t col = 0; col < cols; ++col) {
            const float value =
                output[row * cols + col];

            if (!std::isfinite(value) || value < 0.0f) {
                return false;
            }

            sum += value;
        }

        if (std::fabs(sum - 1.0f) > 1e-4f) {
            return false;
        }
    }

    return true;
}

int main() {
    constexpr std::size_t rows = 1024;
    constexpr int runs = 7;

    const std::vector<std::size_t> widths = {
        128,
        256,
        512,
        1024,
        2048
    };

    std::cout << std::fixed << std::setprecision(3);

    std::cout
        << "Shape\tMedian (ms)\tMElements/s\tCorrect\n"
        << "------------------------------------------------\n";

    for (const auto cols : widths) {
        const auto input =
            random_input(rows, cols, 42);

        const auto warmup =
            softmax_cpu(input, rows, cols);

        if (!validate_softmax(warmup, rows, cols)) {
            std::cerr << "Warm-up validation failed\n";
            return 1;
        }

        std::vector<double> timings;
        timings.reserve(runs);

        std::vector<float> output;

        for (int run = 0; run < runs; ++run) {
            const auto start =
                std::chrono::high_resolution_clock::now();

            output =
                softmax_cpu(input, rows, cols);

            const auto end =
                std::chrono::high_resolution_clock::now();

            const std::chrono::duration<double, std::milli>
                elapsed = end - start;

            timings.push_back(elapsed.count());
        }

        std::sort(timings.begin(), timings.end());

        const double median_ms =
            timings[timings.size() / 2];

        const double million_elements_per_second =
            (
                static_cast<double>(rows) *
                static_cast<double>(cols)
            ) /
            (median_ms / 1000.0) /
            1e6;

        const bool correct =
            validate_softmax(output, rows, cols);

        std::cout
            << rows << "x" << cols
            << '\t'
            << median_ms
            << '\t'
            << million_elements_per_second
            << '\t'
            << (correct ? "PASS" : "FAIL")
            << '\n';

        if (!correct) {
            return 1;
        }
    }

    return 0;
}
