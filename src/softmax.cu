#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

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

constexpr int SOFTMAX_BLOCK_SIZE = 256;

__global__ void softmax_naive_kernel(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const std::size_t offset = row * cols;

    __shared__ float shared[SOFTMAX_BLOCK_SIZE];

    // -------------------------
    // Row maximum
    // -------------------------

    float local_max =
        -CUDART_INF_F;

    for (
        std::size_t col = tid;
        col < cols;
        col += blockDim.x
    ) {
        local_max =
            fmaxf(local_max, input[offset + col]);
    }

    shared[tid] = local_max;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] =
                fmaxf(
                    shared[tid],
                    shared[tid + stride]
                );
        }

        __syncthreads();
    }

    const float row_max = shared[0];

    // -------------------------
    // Exponential + row sum
    // -------------------------

    float local_sum = 0.0f;

    for (
        std::size_t col = tid;
        col < cols;
        col += blockDim.x
    ) {
        const float value =
            expf(input[offset + col] - row_max);

        output[offset + col] = value;

        local_sum += value;
    }

    shared[tid] = local_sum;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }

    const float row_sum = shared[0];

    // -------------------------
    // Normalize
    // -------------------------

    for (
        std::size_t col = tid;
        col < cols;
        col += blockDim.x
    ) {
        output[offset + col] /= row_sum;
    }
}

void softmax_cuda_naive(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
) {
    const dim3 block(SOFTMAX_BLOCK_SIZE);
    const dim3 grid(
        static_cast<unsigned int>(rows)
    );

    softmax_naive_kernel<<<grid, block>>>(
        input,
        output,
        rows,
        cols
    );

    CUDA_CHECK(cudaGetLastError());
}

constexpr int SOFTMAX_VALUES_PER_THREAD = 8;

__global__ void softmax_register_cached_kernel(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const std::size_t offset = row * cols;

    __shared__ float shared[SOFTMAX_BLOCK_SIZE];

    float values[SOFTMAX_VALUES_PER_THREAD];
    float exponentials[SOFTMAX_VALUES_PER_THREAD];

    float local_max = -CUDART_INF_F;

    #pragma unroll
    for (int i = 0; i < SOFTMAX_VALUES_PER_THREAD; ++i) {
        const std::size_t col =
            tid + static_cast<std::size_t>(i) * blockDim.x;

        float value = -CUDART_INF_F;

        if (col < cols) {
            value = input[offset + col];
        }

        values[i] = value;
        local_max = fmaxf(local_max, value);
    }

    shared[tid] = local_max;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] =
                fmaxf(
                    shared[tid],
                    shared[tid + stride]
                );
        }

        __syncthreads();
    }

    const float row_max = shared[0];

    float local_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < SOFTMAX_VALUES_PER_THREAD; ++i) {
        const std::size_t col =
            tid + static_cast<std::size_t>(i) * blockDim.x;

        float exp_value = 0.0f;

        if (col < cols) {
            exp_value =
                expf(values[i] - row_max);
        }

        exponentials[i] = exp_value;
        local_sum += exp_value;
    }

    shared[tid] = local_sum;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }

    const float row_sum = shared[0];

    #pragma unroll
    for (int i = 0; i < SOFTMAX_VALUES_PER_THREAD; ++i) {
        const std::size_t col =
            tid + static_cast<std::size_t>(i) * blockDim.x;

        if (col < cols) {
            output[offset + col] =
                exponentials[i] / row_sum;
        }
    }
}

void softmax_cuda_register_cached(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols
) {
    const dim3 block(SOFTMAX_BLOCK_SIZE);

    const dim3 grid(
        static_cast<unsigned int>(rows)
    );

    softmax_register_cached_kernel<<<grid, block>>>(
        input,
        output,
        rows,
        cols
    );

    CUDA_CHECK(cudaGetLastError());
}
