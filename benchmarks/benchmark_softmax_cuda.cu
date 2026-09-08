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

using SoftmaxFunction = void (*)(
    const float*,
    float*,
    std::size_t,
    std::size_t
);

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
    SoftmaxFunction function,
    const float* d_input,
    float* d_output,
    std::size_t rows,
    std::size_t cols,
    int runs
) {
    function(
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

        function(
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

bool run_and_check(
    SoftmaxFunction function,
    const std::vector<float>& reference,
    std::vector<float>& output,
    const float* d_input,
    float* d_output,
    std::size_t bytes,
    std::size_t rows,
    std::size_t cols
) {
    function(
        d_input,
        d_output,
        rows,
        cols
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    return outputs_close(reference, output);
}

double throughput_millions(
    std::size_t rows,
    std::size_t cols,
    float milliseconds
) {
    return (
        static_cast<double>(rows) *
        static_cast<double>(cols)
    ) /
    (
        static_cast<double>(milliseconds) /
        1000.0
    ) /
    1e6;
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
        << "Shape\tNaive MEl/s\tRegister MEl/s\t"
        << "Speedup\tCorrect\n"
        << "------------------------------------------------------------\n";

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

        const bool naive_correct =
            run_and_check(
                softmax_cuda_naive,
                reference,
                output,
                d_input,
                d_output,
                bytes,
                rows,
                cols
            );

        const bool register_correct =
            run_and_check(
                softmax_cuda_register_cached,
                reference,
                output,
                d_input,
                d_output,
                bytes,
                rows,
                cols
            );

        const float naive_ms =
            benchmark_softmax(
                softmax_cuda_naive,
                d_input,
                d_output,
                rows,
                cols,
                runs
            );

        const float register_ms =
            benchmark_softmax(
                softmax_cuda_register_cached,
                d_input,
                d_output,
                rows,
                cols,
                runs
            );

        const double naive_throughput =
            throughput_millions(
                rows,
                cols,
                naive_ms
            );

        const double register_throughput =
            throughput_millions(
                rows,
                cols,
                register_ms
            );

        const double speedup =
            static_cast<double>(naive_ms) /
            static_cast<double>(register_ms);

        const bool correct =
            naive_correct &&
            register_correct;

        std::cout
            << rows << "x" << cols
            << '\t'
            << naive_throughput
            << '\t'
            << register_throughput
            << '\t'
            << speedup << "x"
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
