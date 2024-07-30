#include <iostream>
#include <sys/time.h>

template <typename T, size_t N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" {
    void _mlir_ciface_matmul(int64_t M, int64_t N, int64_t K, MemRefDescriptor<float, 1> *A, MemRefDescriptor<float, 1> *B, MemRefDescriptor<float, 1> *C);
}

void print_matrix(float *matrix, int64_t rows, int64_t cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int64_t M = std::stoi(argv[1]);
    int64_t N = std::stoi(argv[2]);
    int64_t K = std::stoi(argv[3]);

    if (N%8 != 0) {
        std::cout << "N must be a multiple of 8" << std::endl;
        return 1;
    }

    float *A = new float[M*N];
    for (int i=0; i<M*N; i++)
        A[i] = i+1;
    float *B = new float[N*K];
    for (int i=0; i<N*K; i++)
        B[i] = i+5;
    float *C = new float[M*K];
    for (int i=0; i<M*K; i++)
        C[i] = 0;

    MemRefDescriptor<float, 1> *memrefA = new MemRefDescriptor<float, 1> {
        A,    // allocated
        A,    // aligned
        0,    // offset
        {M * N}, // sizes[N]
        {1},  // strides[N]
    };

    MemRefDescriptor<float, 1> *memrefB = new MemRefDescriptor<float, 1> {
        B,    // allocated
        B,    // aligned
        0,    // offset
        {N * K}, // sizes[N]
        {1},  // strides[N]
    };

    MemRefDescriptor<float, 1> *memrefC = new MemRefDescriptor<float, 1> {
        C,    // allocated
        C,    // aligned
        0,    // offset
        {M * K}, // sizes[N]
        {1},  // strides[N]
    };

    std::cout << "Before matmul call\n" << std::endl;

    struct timeval t1;
    gettimeofday(&t1, NULL);
    long t1s = t1.tv_sec * 1000000L + t1.tv_usec;
    _mlir_ciface_matmul(M, N, K, memrefA, memrefB, memrefC);
    struct timeval t2;
    gettimeofday(&t2, NULL);
    long t2s = t2.tv_sec * 1000000L + t2.tv_usec;

    std::cout << "Result matrix:\n";
    print_matrix(C, M, K);

    std::cout << t2s - t1s << std::endl;

    return 0;
}
