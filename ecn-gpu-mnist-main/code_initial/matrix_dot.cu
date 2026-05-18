// matrix_dot.cu
// Compile:
//   nvcc -O3 -arch=sm_86 -c matrix_dot.cu -o matrix_dot.o
//   gcc  -O3 -c main.c ann.c mnist.c matrix.c
//   gcc  -o ann main.o ann.o mnist.o matrix.o matrix_dot.o -lcudart -lm

#include "matrix.h"
#include <assert.h>

#define TILE 16

__global__ void k_dot(const double *A, const double *B, double *C,
                      int M, int K, int N)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ double tA[TILE][TILE];
    __shared__ double tB[TILE][TILE];

    double acc = 0.0;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int kA = t * TILE + threadIdx.x;
        int kB = t * TILE + threadIdx.y;

        tA[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? A[row * K + kA] : 0.0;
        tB[threadIdx.y][threadIdx.x] = (kB  < K && col < N) ? B[kB  * N + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            acc += tA[threadIdx.y][k] * tB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

extern "C" void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert(m1->columns == m2->rows &&
           m1->rows    == res->rows &&
           m2->columns == res->columns);

    int M = m1->rows, K = m1->columns, N = m2->columns;

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE-1)/TILE, (M + TILE-1)/TILE);

    k_dot<<<blocks, threads>>>(m1->m, m2->m, res->m, M, K, N);
    cudaDeviceSynchronize();
}
