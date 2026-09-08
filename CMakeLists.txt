cmake_minimum_required(VERSION 3.20)

project(cuda_transformer_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
