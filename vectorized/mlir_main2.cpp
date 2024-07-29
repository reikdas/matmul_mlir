#include <iostream>

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

int main() {
    int64_t M = 8, N = 8, K = 8;
    // float* A = (float*)malloc(M * N * sizeof(float));
    // float* B = (float*)malloc(N * K * sizeof(float));
    // float *C = (float*)calloc(0, M * K * sizeof(float));
    // A[0] = 1;
    // A[1] = 2;
    // A[2] = 3;
    // A[3] = 4;
    // B[0] = 5;
    // B[1] = 6;
    // B[2] = 7;
    // B[3] = 8;

    // float A[] = {1.0, 2.0, 3.0, 4.0};
    // float B[] = {5.0, 6.0, 7.0, 8.0};
    // float C[] = {0, 0, 0, 0};

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

    _mlir_ciface_matmul(M, N, K, memrefA, memrefB, memrefC);

    std::cout << "Result matrix:\n";
    print_matrix(C, M, K);

    return 0;
}
