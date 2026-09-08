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

constexpr int TILE_SIZE = 16;

__global__ void matmul_tiled_kernel(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    const std::size_t row =
        blockIdx.y * TILE_SIZE + threadIdx.y;

    const std::size_t col =
        blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    const std::size_t num_tiles =
        (K + TILE_SIZE - 1) / TILE_SIZE;

    for (std::size_t tile = 0; tile < num_tiles; ++tile) {
        const std::size_t a_col =
            tile * TILE_SIZE + threadIdx.x;

        const std::size_t b_row =
            tile * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] =
                A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] =
                B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum +=
                tile_A[threadIdx.y][k] *
                tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_cuda_tiled(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
) {
    const dim3 block(TILE_SIZE, TILE_SIZE);

    const dim3 grid(
        static_cast<unsigned int>(
            (N + TILE_SIZE - 1) / TILE_SIZE
        ),
        static_cast<unsigned int>(
            (M + TILE_SIZE - 1) / TILE_SIZE
        )
    );

    matmul_tiled_kernel<<<grid, block>>>(
        A,
        B,
        C,
        M,
        K,
        N
    );

    CUDA_CHECK(cudaGetLastError());
}
