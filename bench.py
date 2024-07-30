import subprocess

BENCHMARK_FREQ = 5

if __name__ == "__main__":
    M = "1000"
    N = "1024" # Multiple of 8
    K = "2000"
    # MLIR vectorized
    subprocess.run(["./run.sh"], cwd="vectorized", capture_output=True)
    mlir = []
    for i in range(BENCHMARK_FREQ):
        output = subprocess.run(["./run.sh", M, N, K], cwd="vectorized", capture_output=True).stdout.decode("utf-8").split("\n")[-2]
        mlir.append(float(output))
    # Naive C++
    cpp = []
    subprocess.run(["g++", "-O3", "-march=native", "-mavx", "-mprefer-vector-width=512", "-o", "naive", "naive.cpp"])
    for i in range(BENCHMARK_FREQ):
        output = subprocess.run(["./naive", M, N, K], capture_output=True).stdout.decode("utf-8").split("\n")[-2]
        cpp.append(float(output))
    print("MLIR:", mlir, " ", sum(mlir) / BENCHMARK_FREQ)
    print("Naive C++:", cpp, " ", sum(cpp) / BENCHMARK_FREQ)
        