#include "matmul.h"
#include "matmul_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

#define CUBLAS_CHECK(call)                                            \
    do {                                                              \
        cublasStatus_t status = (call);                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                        \
            std::cerr                                                 \
                << "cuBLAS error at "                                 \
                << __FILE__ << ":" << __LINE__                        \
                << ", status=" << static_cast<int>(status)            \
                << '\n';                                              \
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

float benchmark_cublas(
    cublasHandle_t handle,
    const float* d_A,
    const float* d_B,
    float* d_C,
    std::size_t N,
    int runs
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Our matrices are row-major.
    // cuBLAS expects column-major, so compute:
    // C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(N),
        static_cast<int>(N),
        &alpha,
        d_B,
        static_cast<int>(N),
        d_A,
        static_cast<int>(N),
        &beta,
        d_C,
        static_cast<int>(N)
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> timings;
    timings.reserve(runs);

    for (int run = 0; run < runs; ++run) {
        CUDA_CHECK(cudaEventRecord(start));

        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(N),
            static_cast<int>(N),
            &alpha,
            d_B,
            static_cast<int>(N),
            d_A,
            static_cast<int>(N),
            &beta,
            d_C,
            static_cast<int>(N)
        ));

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

bool run_and_check(
    MatmulFunction function,
    const std::vector<float>& reference,
    std::vector<float>& result,
    const float* d_A,
    const float* d_B,
    float* d_C,
    std::size_t bytes,
    std::size_t N
) {
    function(d_A, d_B, d_C, N, N, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        result.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    return matrices_close(reference, result);
}

bool run_and_check_cublas(
    cublasHandle_t handle,
    const std::vector<float>& reference,
    std::vector<float>& result,
    const float* d_A,
    const float* d_B,
    float* d_C,
    std::size_t bytes,
    std::size_t N
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(N),
        static_cast<int>(N),
        &alpha,
        d_B,
        static_cast<int>(N),
        d_A,
        static_cast<int>(N),
        &beta,
        d_C,
        static_cast<int>(N)
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        result.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    return matrices_close(reference, result);
}

int main() {
    constexpr int runs = 20;

    const std::vector<std::size_t> sizes = {
        128,
        256,
        512,
        1024
    };

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::cout << std::fixed << std::setprecision(3);

    std::cout
        << "Size\tNaive GFLOPS\tTiled GFLOPS\t"
        << "Register GFLOPS\tcuBLAS GFLOPS\t"
        << "Register/cuBLAS\tCorrect\n"
        << "--------------------------------------------------------------------------------\n";

    for (const auto N : sizes) {
        const auto A = random_matrix(N, N, 42);
        const auto B = random_matrix(N, N, 1337);

        const auto reference =
            matmul_cpu(A, B, N, N, N);

        std::vector<float> result(N * N, 0.0f);

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

        const bool naive_correct =
            run_and_check(
                matmul_cuda_naive,
                reference,
                result,
                d_A,
                d_B,
                d_C,
                bytes,
                N
            );

        const bool tiled_correct =
            run_and_check(
                matmul_cuda_tiled,
                reference,
                result,
                d_A,
                d_B,
                d_C,
                bytes,
                N
            );

        const bool register_correct =
            run_and_check(
                matmul_cuda_register_tiled,
                reference,
                result,
                d_A,
                d_B,
                d_C,
                bytes,
                N
            );

        const bool cublas_correct =
            run_and_check_cublas(
                handle,
                reference,
                result,
                d_A,
                d_B,
                d_C,
                bytes,
                N
            );

        const float naive_ms =
            benchmark_kernel(
                matmul_cuda_naive,
                d_A,
                d_B,
                d_C,
                N,
                runs
            );

        const float tiled_ms =
            benchmark_kernel(
                matmul_cuda_tiled,
                d_A,
                d_B,
                d_C,
                N,
                runs
            );

        const float register_ms =
            benchmark_kernel(
                matmul_cuda_register_tiled,
                d_A,
                d_B,
                d_C,
                N,
                runs
            );

        const float cublas_ms =
            benchmark_cublas(
                handle,
                d_A,
                d_B,
                d_C,
                N,
                runs
            );

        const double naive_gflops =
            calculate_gflops(N, naive_ms);

        const double tiled_gflops =
            calculate_gflops(N, tiled_ms);

        const double register_gflops =
            calculate_gflops(N, register_ms);

        const double cublas_gflops =
            calculate_gflops(N, cublas_ms);

        const double register_vs_cublas =
            register_gflops / cublas_gflops;

        const bool correct =
            naive_correct &&
            tiled_correct &&
            register_correct &&
            cublas_correct;

        std::cout
            << N << "x" << N
            << '\t'
            << naive_gflops
            << '\t'
            << tiled_gflops
            << '\t'
            << register_gflops
            << '\t'
            << cublas_gflops
            << '\t'
            << register_vs_cublas * 100.0 << "%"
            << '\t'
            << (correct ? "PASS" : "FAIL")
            << '\n';

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        if (!correct) {
            CUBLAS_CHECK(cublasDestroy(handle));
            return 1;
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
