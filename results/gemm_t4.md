# GEMM Benchmark Results

## Environment

- GPU: NVIDIA Tesla T4
- Compute capability: 7.5
- CUDA compiler: CUDA 12.8
- Build type: Release
- CUDA architecture: 75
- Timing: CUDA events
- Runs per implementation: 20
- Reported timing statistic: median
- Data type: FP32

## Results

| Matrix size | Naive CUDA (GFLOPS) | Shared-memory tiled (GFLOPS) | Register-tiled (GFLOPS) | cuBLAS SGEMM (GFLOPS) | Custom / cuBLAS |
|---|---:|---:|---:|---:|---:|
| 128x128 | 207.721 | 231.168 | 233.640 | 251.096 | 93.048% |
| 256x256 | 327.680 | 400.526 | 645.675 | 837.521 | 77.094% |
| 512x512 | 376.644 | 473.157 | 872.722 | 3030.567 | 28.797% |
| 1024x1024 | 385.648 | 485.283 | 1278.508 | 5322.721 | 24.020% |

All implementations passed comparison against the CPU reference.

## 1024x1024 Optimization Progression

- Naive CUDA: 385.648 GFLOPS
- Shared-memory tiled: 485.283 GFLOPS
- Register-tiled + padded shared memory: 1278.508 GFLOPS
- cuBLAS SGEMM: 5322.721 GFLOPS

The final custom kernel achieved approximately 3.31x the throughput of the naive CUDA implementation and 24.0% of cuBLAS SGEMM throughput.

## Profiling Findings

Nsight Compute profiling guided the optimization process.

Shared-memory tiling improved data reuse but exposed scheduler and memory-pipeline stalls. Register tiling reduced instruction overhead and increased arithmetic reuse. Nsight then identified shared-memory bank conflicts in the register-tiled kernel. Padding the shared-memory tile reduced excessive shared-memory wavefronts substantially and improved kernel throughput.

Performance comparisons should be interpreted at larger matrix sizes; small matrices are more sensitive to kernel-launch and library overhead.
