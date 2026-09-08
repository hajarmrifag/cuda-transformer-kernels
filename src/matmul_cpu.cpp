#include "matmul.h"

#include <stdexcept>
#include <vector>

std::vector<float> matmul_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::size_t M,
    std::size_t K,
    std::size_t N
) {
    if (A.size() != M * K) {
        throw std::invalid_argument("A has incorrect dimensions");
    }

    if (B.size() != K * N) {
        throw std::invalid_argument("B has incorrect dimensions");
    }

    std::vector<float> C(M * N, 0.0f);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float a = A[i * K + k];

            for (std::size_t j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }

    return C;
}
