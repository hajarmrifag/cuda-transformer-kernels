# Softmax Benchmark Results

## Environment

- GPU: NVIDIA Tesla T4
- Compute capability: 7.5
- CUDA compiler: CUDA 12.8
- Build type: Release
- CUDA architecture: 75
- Timing: CUDA events
- Runs per implementation: 20
- Reported timing statistic: median
- Rows: 1024
- Data type: FP32

## Results

| Shape | Naive CUDA (MEl/s) | Register-cached (MEl/s) | Warp-shuffle (MEl/s) | Shuffle / Register |
|---|---:|---:|---:|---:|
| 1024x128 | 2370.370 | 2025.717 | 1732.656 | 0.855x |
| 1024x256 | 4538.504 | 4027.532 | 3443.464 | 0.855x |
| 1024x512 | 8035.312 | 7529.412 | 6377.579 | 0.847x |
| 1024x1024 | 14302.925 | 17355.932 | 14246.956 | 0.821x |
| 1024x2048 | 15746.276 | 27002.883 | 24290.585 | 0.900x |

All implementations passed comparison against the CPU reference.

## 2048-Wide Optimization Progression

- Naive CUDA: 15.746 billion elements/s
- Register-cached CUDA: 27.003 billion elements/s
- Warp-shuffle CUDA: 24.291 billion elements/s

Register caching achieved approximately 1.71x the throughput of the naive implementation for 2048-element rows.

The warp-shuffle variant was approximately 10% slower than the register-cached implementation at this width, so the register-cached implementation was retained as the best-performing version.

## Profiling Findings

Nsight Compute showed that the naive implementation suffered from long-scoreboard stalls associated with L1TEX memory dependencies.

Register caching kept each thread's input values and exponentials in registers, reducing redundant global-memory traffic.

For the 2048-wide workload, profiling showed:

- Issue-slot utilization increased from 42.75% to 63.36%.
- Scheduler cycles with no eligible warp decreased from 57.14% to 36.52%.
- Eligible warps per scheduler increased from 0.74 to 1.40.
- Warp cycles per issued instruction decreased from 17.42 to 11.70.
- Profiled kernel duration decreased from 150.27 us to 85.12 us.

The warp-shuffle reduction experiment did not improve end-to-end throughput on the Tesla T4, illustrating that lower-level optimization techniques must be evaluated empirically rather than assumed to be faster.

## Scope

The register-cached and warp-shuffle implementations are designed for rows of up to 2048 elements with the current configuration of 256 threads and eight cached values per thread.
