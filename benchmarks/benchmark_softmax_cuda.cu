#include "softmax.h"
#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t error = (call);                                   \
        if (error != cudaSuccess) {                                   \
            std::cerr                                                 \
                << "CUDA error at "                                   \
                << __FILE__ << ":" << __LINE__ << ": "                \
                << cudaGetErrorString(error) << '\n';                  \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

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

bool outputs_close(
    const std::vector<float>& reference,
    const std::vector<float>& candidate,
    float tolerance = 1e-4f
) {
    if (reference.size() != candidate.size()) {
        return false;
    }

    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float diff =
            std::fabs(reference[i] - candidate[i]);

        if (diff > tolerance) {
            std::cerr
                << "Mismatch at index " << i
                << ": CPU=" << reference[i]
                << ", GPU=" << candidate[i]
                << ", diff=" << diff
                << '\n';

            return false;
        }
    }

    return true;
}

float benchmark_softmax(
    const float* d_input,
    float* d_output,
    std::size_t rows,
    std::size_t cols,
    int runs
) {
    softmax_cuda_naive(
        d_input,
        d_output,
        rows,
        cols
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> timings;
    timings.reserve(runs);

    for (int run = 0; run < runs; ++run) {
        CUDA_CHECK(cudaEventRecord(start));

        softmax_cuda_naive(
            d_input,
            d_output,
            rows,
            cols
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        ));

        timings.push_back(milliseconds);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(timings.begin(), timings.end());

    return timings[timings.size() / 2];
}

int main() {
    constexpr std::size_t rows = 1024;
    constexpr int runs = 20;

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

        const auto reference =
            softmax_cpu(input, rows, cols);

        std::vector<float> output(
            rows * cols,
            0.0f
        );

        const std::size_t bytes =
            rows * cols * sizeof(float);

        float* d_input = nullptr;
        float* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));

        CUDA_CHECK(cudaMemcpy(
            d_input,
            input.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        const float median_ms =
            benchmark_softmax(
                d_input,
                d_output,
                rows,
                cols,
                runs
            );

        CUDA_CHECK(cudaMemcpy(
            output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        const bool correct =
            outputs_close(reference, output);

        const double million_elements_per_second =
            (
                static_cast<double>(rows) *
                static_cast<double>(cols)
            ) /
            (
                static_cast<double>(median_ms) /
                1000.0
            ) /
            1e6;

        std::cout
            << rows << "x" << cols
            << '\t'
            << median_ms
            << '\t'
            << million_elements_per_second
            << '\t'
            << (correct ? "PASS" : "FAIL")
            << '\n';

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        if (!correct) {
            return 1;
        }
    }

    return 0;
}
