import matplotlib.pyplot as plt

# GEMM data
sizes = ["128", "256", "512", "1024"]
naive = [207.721, 327.680, 376.644, 385.648]
tiled = [231.168, 400.526, 473.157, 485.283]
register = [233.640, 645.675, 872.722, 1278.508]
cublas = [251.096, 837.521, 3030.567, 5322.721]

plt.figure(figsize=(9, 5))
plt.plot(sizes, naive, marker="o", label="Naive CUDA")
plt.plot(sizes, tiled, marker="o", label="Shared-memory tiled")
plt.plot(sizes, register, marker="o", label="Register-tiled")
plt.plot(sizes, cublas, marker="o", label="cuBLAS SGEMM")
plt.xlabel("Square matrix size")
plt.ylabel("GFLOPS")
plt.title("GEMM throughput on NVIDIA Tesla T4")
plt.legend()
plt.tight_layout()
plt.savefig("results/charts/gemm_t4.png", dpi=180)
plt.close()

# Softmax data
widths = ["128", "256", "512", "1024", "2048"]
softmax_naive = [2370.370, 4538.504, 8035.312, 14302.925, 15746.276]
softmax_register = [2025.717, 4027.532, 7529.412, 17355.932, 27002.883]
softmax_shuffle = [1732.656, 3443.464, 6377.579, 14246.956, 24290.585]

plt.figure(figsize=(9, 5))
plt.plot(widths, softmax_naive, marker="o", label="Naive CUDA")
plt.plot(widths, softmax_register, marker="o", label="Register-cached")
plt.plot(widths, softmax_shuffle, marker="o", label="Warp-shuffle")
plt.xlabel("Elements per row")
plt.ylabel("Million elements / second")
plt.title("Softmax throughput on NVIDIA Tesla T4")
plt.legend()
plt.tight_layout()
plt.savefig("results/charts/softmax_t4.png", dpi=180)
plt.close()

print("Created:")
print("results/charts/gemm_t4.png")
print("results/charts/softmax_t4.png")
