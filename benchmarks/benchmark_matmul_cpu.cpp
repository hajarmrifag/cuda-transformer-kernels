#include "matmul.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

std::vector<float> random_matrix(
    std::size_t rows,
    std::size_t cols,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> matrix(rows * cols);

    for (auto& value : matrix) {
        value = dist(rng);
    }

    return matrix;
}

int main() {
    constexpr int runs = 7;

    const std::vector<std::size_t> sizes = {
        128,
        256,
        512,
        1024
    };

    std::cout << std::fixed << std::setprecision(3);

    std::cout
        << "Size\tMedian (ms)\tGFLOPS\n"
        << "----------------------------------------\n";

    for (const auto N : sizes) {
        const auto A = random_matrix(N, N, 42);
        const auto B = random_matrix(N, N, 1337);

        // Warm-up run
        auto warmup = matmul_cpu(A, B, N, N, N);

        if (warmup.empty()) {
            std::cerr << "Warm-up failed\n";
            return 1;
        }

        std::vector<double> timings;
        timings.reserve(runs);

        for (int run = 0; run < runs; ++run) {
            const auto start =
                std::chrono::high_resolution_clock::now();

            const auto C = matmul_cpu(
                A,
                B,
                N,
                N,
                N
            );

            const auto end =
                std::chrono::high_resolution_clock::now();

            if (C.empty()) {
                std::cerr << "Unexpected empty result\n";
                return 1;
            }

            const std::chrono::duration<double, std::milli>
                elapsed = end - start;

            timings.push_back(elapsed.count());
        }

        std::sort(timings.begin(), timings.end());

        const double median_ms =
            timings[timings.size() / 2];

        const double seconds =
            median_ms / 1000.0;

        const double operations =
            2.0 *
            static_cast<double>(N) *
            static_cast<double>(N) *
            static_cast<double>(N);

        const double gflops =
            operations / seconds / 1e9;

        std::cout
            << N << "x" << N
            << '\t'
            << median_ms
            << '\t'
            << gflops
            << '\n';
    }

    return 0;
}
