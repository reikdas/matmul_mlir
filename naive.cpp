#include <iostream>
#include <sys/time.h>

void print_matrix(float *matrix, int64_t rows, int64_t cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void matmul(int64_t M, int64_t N, int64_t K, float *A, float *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < K; j++) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
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

    std::cout << "Before matmul call\n" << std::endl;

    struct timeval t1;
    gettimeofday(&t1, NULL);
    long t1s = t1.tv_sec * 1000000L + t1.tv_usec;
    matmul(M, N, K, A, B, C);
    struct timeval t2;
    gettimeofday(&t2, NULL);
    long t2s = t2.tv_sec * 1000000L + t2.tv_usec;

    std::cout << "Result matrix:\n";
    print_matrix(C, M, K);

    std::cout << t2s - t1s << std::endl;

    return 0;
}
