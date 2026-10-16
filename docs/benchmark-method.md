# Benchmark boundaries and reproduction

The committed `results/gemm_t4.md` and `results/softmax_t4.md` are historical summaries on Tesla T4/CUDA 12.8. Kernel timing excludes host allocation, host-device transfer and application/network work. After warm-up the harness records 20 CUDA-event timings and selects sorted index `n/2`: the **upper central observation**, not the average of both central observations for an even sample. Historical “median” references use that statistic.

The 1024-square register-tiled GEMM is 3.31x the naive kernel and approximately 24% of cuBLAS, not faster than cuBLAS. Softmax's 1.71x result is for 1024 rows x 2048 columns; register caching is not the best implementation for every width.

Rebuild Release, run the correctness target first, then run `benchmark_matmul_cuda` and `benchmark_softmax_cuda`. Capture exact source revision, GPU name/driver, `nvcc --version`, Nsight version, compiler flags, clocks/power/temperature, command and stdout. Preserve original result tables and create a dated new result directory. Retain Nsight `.ncu-rep` or CSV exports with the exact profiled kernel; memory-wavefront claims currently have narrative evidence only.

The local verification host cannot recreate raw historical samples or profiler files. Do not manufacture them or relabel a future run as the original. Fresh GPU compilation, execution and retained per-run samples remain the hardware-dependent verification boundary.
