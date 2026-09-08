#include "matmul.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t M = 2;
    constexpr std::size_t K = 3;
    constexpr std::size_t N = 2;

    const std::vector<float> A = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    const std::vector<float> B = {
        7.0f,  8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };

    const std::vector<float> expected = {
        58.0f, 64.0f,
        139.0f, 154.0f
    };

    const auto start = std::chrono::high_resolution_clock::now();

    const auto C = matmul_cpu(A, B, M, K, N);

    const auto end = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < C.size(); ++i) {
        if (std::fabs(C[i] - expected[i]) > 1e-5f) {
            std::cerr << "FAILED: incorrect result at index "
                      << i << '\n';
            return 1;
        }
    }

    const std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "CPU matrix multiplication: PASS\n";
    std::cout << "Execution time: "
              << elapsed.count()
              << " microseconds\n";

    return 0;
}
