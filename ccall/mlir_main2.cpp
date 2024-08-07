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
    int64_t M = 2, N = 2, K = 2;

    float A[] = {1.0, 2.0, 3.0, 4.0};
    float B[] = {5.0, 6.0, 7.0, 8.0};
    float C[] = {0, 0, 0, 0};

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
