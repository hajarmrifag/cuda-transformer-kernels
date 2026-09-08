#include "matmul.h"
#include "matmul_cuda.h"

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

using MatmulFunction = void (*)(
    const float*,
    const float*,
    float*,
    std::size_t,
    std::size_t,
    std::size_t
);

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

bool matrices_close(
    const std::vector<float>& reference,
    const std::vector<float>& candidate,
    float tolerance = 1e-3f
) {
    if (reference.size() != candidate.size()) {
        return false;
    }

    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float difference =
            std::fabs(reference[i] - candidate[i]);

        if (difference > tolerance) {
            std::cerr
                << "Mismatch at index " << i
                << ": CPU=" << reference[i]
                << ", GPU=" << candidate[i]
                << ", diff=" << difference
                << '\n';

            return false;
        }
    }

    return true;
}

float benchmark_kernel(
    MatmulFunction function,
    const float* d_A,
    const float* d_B,
    float* d_C,
    std::size_t N,
    int runs
) {
    function(d_A, d_B, d_C, N, N, N);
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
            d_A,
            d_B,
            d_C,
            N,
            N,
            N
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

double calculate_gflops(
    std::size_t N,
    float milliseconds
) {
    const double operations =
        2.0 *
        static_cast<double>(N) *
        static_cast<double>(N) *
        static_cast<double>(N);

    const double seconds =
        static_cast<double>(milliseconds) / 1000.0;

    return operations / seconds / 1e9;
}

int main() {
    constexpr int runs = 20;

    const std::vector<std::size_t> sizes = {
        128,
        256,
        512,
        1024
    };

    std::cout << std::fixed << std::setprecision(3);

    std::cout
        << "Size\tNaive ms\tTiled ms\tNaive GFLOPS\t"
        << "Tiled GFLOPS\tSpeedup\tCorrect\n"
        << "-------------------------------------------------------------"
        << "-------------------\n";

    for (const auto N : sizes) {
        const auto A = random_matrix(N, N, 42);
        const auto B = random_matrix(N, N, 1337);

        const auto reference =
            matmul_cpu(A, B, N, N, N);

        std::vector<float> naive_result(N * N, 0.0f);
        std::vector<float> tiled_result(N * N, 0.0f);

        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_C = nullptr;

        const std::size_t bytes =
            N * N * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(
            d_A,
            A.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_B,
            B.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        const float naive_ms =
            benchmark_kernel(
                matmul_cuda_naive,
                d_A,
                d_B,
                d_C,
                N,
                runs
            );

        CUDA_CHECK(cudaMemcpy(
            naive_result.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        const bool naive_correct =
            matrices_close(reference, naive_result);

        const float tiled_ms =
            benchmark_kernel(
                matmul_cuda_tiled,
                d_A,
                d_B,
                d_C,
                N,
                runs
            );

        CUDA_CHECK(cudaMemcpy(
            tiled_result.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        const bool tiled_correct =
            matrices_close(reference, tiled_result);

        const double naive_gflops =
            calculate_gflops(N, naive_ms);

        const double tiled_gflops =
            calculate_gflops(N, tiled_ms);

        const double speedup =
            static_cast<double>(naive_ms) /
            static_cast<double>(tiled_ms);

        std::cout
            << N << "x" << N
            << '\t'
            << naive_ms
            << '\t'
            << tiled_ms
            << '\t'
            << naive_gflops
            << '\t'
            << tiled_gflops
            << '\t'
            << speedup << "x"
            << '\t'
            << (
                naive_correct && tiled_correct
                    ? "PASS"
                    : "FAIL"
            )
            << '\n';

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        if (!naive_correct || !tiled_correct) {
            return 1;
        }
    }

    return 0;
}
