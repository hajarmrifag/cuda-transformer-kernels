# Validation and supported inputs

CPU GEMM/Softmax require positive dimensions, representable float-buffer sizes and exactly sized input vectors. CPU Softmax rejects NaN and infinity. CUDA callers supply valid device buffers, finite input and nonaliasing input/output; wrappers cannot inspect device-buffer contents or allocation sizes without transfers.

All Softmax wrappers reject zero dimensions and null pointers. Register-cached and warp-shuffle variants reject widths above 2048 with `std::invalid_argument`; use the naive implementation for wider rows. Warp-shuffle already had a width check before this change, but terminated the process; the missing check was in register-cached.

`validation::close` rejects all NaN/infinity comparisons, including NaN with itself. It supports absolute plus relative tolerance in double precision. Existing historical benchmark tolerances remain absolute 1e-3 (GEMM) / 1e-4 (Softmax); saved results were not rewritten. Finite-reference checks failing to reject NaN was a validator defect, not proof that historical outputs were NaN.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Four CPU test groups cover comparator failures/tolerances, shared wrapper-shape contracts, independent double-accumulation rectangular GEMM oracles and Softmax normalization/shift invariance at tiny, odd and boundary widths. Checks remain active in Release builds.

With a CUDA compiler **and an NVIDIA GPU**:

```sh
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU_TESTS=ON
cmake --build build-gpu --parallel
ctest --test-dir build-gpu --output-on-failure
```

The optional GPU test exercises all three kernels on odd/rectangular shapes and widths 1, 3, 31, 257, 2048 and 2049 (the last is rejected by cached variants). It is prepared but was not compiled/run on the macOS verification host, which has no CUDA compiler/GPU. CPU CI does not establish GPU correctness. No current-GPU performance result is claimed.
