#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t error = (call);                                   \
        if (error != cudaSuccess) {                                   \
            std::fprintf(                                             \
                stderr,                                               \
                "CUDA error at %s:%d: %s\n",                          \
                __FILE__,                                             \
                __LINE__,                                             \
                cudaGetErrorString(error)                             \
            );                                                        \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

__global__ void matmul_naive_kernel(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
) {
    const std::size_t row =
        blockIdx.y * blockDim.y + threadIdx.y;

    const std::size_t col =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (std::size_t k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

void matmul_cuda_naive(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
) {
    const dim3 block(16, 16);

    const dim3 grid(
        static_cast<unsigned int>((N + block.x - 1) / block.x),
        static_cast<unsigned int>((M + block.y - 1) / block.y)
    );

    matmul_naive_kernel<<<grid, block>>>(
        A,
        B,
        C,
        M,
        K,
        N
    );

    CUDA_CHECK(cudaGetLastError());
}
