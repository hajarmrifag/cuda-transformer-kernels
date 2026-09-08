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

constexpr int REGISTER_BLOCK_TILE = 32;
constexpr int REGISTER_K_TILE = 16;
constexpr int THREAD_TILE = 2;

__global__ void matmul_register_tiled_kernel(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
) {
    __shared__ float tile_A[REGISTER_BLOCK_TILE][REGISTER_K_TILE];
    __shared__ float tile_B[REGISTER_K_TILE][REGISTER_BLOCK_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int linear_thread =
        ty * blockDim.x + tx;

    const std::size_t block_row =
        blockIdx.y * REGISTER_BLOCK_TILE;

    const std::size_t block_col =
        blockIdx.x * REGISTER_BLOCK_TILE;

    const std::size_t row0 =
        block_row + ty * THREAD_TILE;

    const std::size_t row1 =
        row0 + 1;

    const std::size_t col0 =
        block_col + tx * THREAD_TILE;

    const std::size_t col1 =
        col0 + 1;

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;

    const std::size_t num_tiles =
        (K + REGISTER_K_TILE - 1) /
        REGISTER_K_TILE;

    for (
        std::size_t tile = 0;
        tile < num_tiles;
        ++tile
    ) {
        const std::size_t k_base =
            tile * REGISTER_K_TILE;

        // 256 threads cooperatively load
        // 512 A values and 512 B values.
        for (int load = 0; load < 2; ++load) {
            const int index =
                linear_thread + load * 256;

            const int a_row =
                index / REGISTER_K_TILE;

            const int a_col =
                index % REGISTER_K_TILE;

            const std::size_t global_a_row =
                block_row + a_row;

            const std::size_t global_a_col =
                k_base + a_col;

            if (
                global_a_row < M &&
                global_a_col < K
            ) {
                tile_A[a_row][a_col] =
                    A[
                        global_a_row * K +
                        global_a_col
                    ];
            } else {
                tile_A[a_row][a_col] = 0.0f;
            }

            const int b_row =
                index / REGISTER_BLOCK_TILE;

            const int b_col =
                index % REGISTER_BLOCK_TILE;

            const std::size_t global_b_row =
                k_base + b_row;

            const std::size_t global_b_col =
                block_col + b_col;

            if (
                global_b_row < K &&
                global_b_col < N
            ) {
                tile_B[b_row][b_col] =
                    B[
                        global_b_row * N +
                        global_b_col
                    ];
            } else {
                tile_B[b_row][b_col] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < REGISTER_K_TILE; ++k) {
            const float a0 =
                tile_A[ty * THREAD_TILE][k];

            const float a1 =
                tile_A[ty * THREAD_TILE + 1][k];

            const float b0 =
                tile_B[k][tx * THREAD_TILE];

            const float b1 =
                tile_B[k][tx * THREAD_TILE + 1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 < M && col0 < N) {
        C[row0 * N + col0] = c00;
    }

    if (row0 < M && col1 < N) {
        C[row0 * N + col1] = c01;
    }

    if (row1 < M && col0 < N) {
        C[row1 * N + col0] = c10;
    }

    if (row1 < M && col1 < N) {
        C[row1 * N + col1] = c11;
    }
}

void matmul_cuda_register_tiled(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t K,
    std::size_t N
) {
    const dim3 block(16, 16);

    const dim3 grid(
        static_cast<unsigned int>(
            (N + REGISTER_BLOCK_TILE - 1) /
            REGISTER_BLOCK_TILE
        ),
        static_cast<unsigned int>(
            (M + REGISTER_BLOCK_TILE - 1) /
            REGISTER_BLOCK_TILE
        )
    );

    matmul_register_tiled_kernel<<<grid, block>>>(
        A,
        B,
        C,
        M,
        K,
        N
    );

    CUDA_CHECK(cudaGetLastError());
}
