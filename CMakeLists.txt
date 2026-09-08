cmake_minimum_required(VERSION 3.20)

project(cuda_transformer_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------
# CPU implementation
# -------------------------

add_library(matmul_cpu
    src/matmul_cpu.cpp
)

target_include_directories(matmul_cpu
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(benchmark_matmul_cpu
    benchmarks/benchmark_matmul_cpu.cpp
)

target_link_libraries(benchmark_matmul_cpu
    PRIVATE
    matmul_cpu
)

# -------------------------
# Optional CUDA support
# -------------------------

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")

    add_library(matmul_cuda
        src/matmul.cu
    )

    target_include_directories(matmul_cuda
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    )

    set_target_properties(matmul_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(benchmark_matmul_cuda
        benchmarks/benchmark_matmul_cuda.cu
    )

    target_include_directories(benchmark_matmul_cuda
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(benchmark_matmul_cuda
        PRIVATE
        matmul_cpu
        matmul_cuda
    )

else()
    message(STATUS "CUDA compiler not found; CUDA targets disabled")
endif()
