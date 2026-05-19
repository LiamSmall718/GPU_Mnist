// matrix_dot.cu
// Compile:
//   nvcc -O3 -arch=sm_86 -c matrix_dot.cu -o matrix_dot.o
//   gcc  -O3 -c main.c ann.c mnist.c matrix.c
//   gcc  -o ann main.o ann.o mnist.o matrix.o matrix_dot.o -lcudart -lm

#include "matrix.h"
#include <assert.h>

#define TILE 16

typedef double (*dev_fn_t)(double);

__device__ double apply_fn(double x, int fn_id)
{
    if (fn_id == 0) return 1.0 / (1.0 + exp(-x));  // sigmoid
    double s = 1.0 / (1.0 + exp(-x));
    return s * (1.0 - s);  // dsigmoid
}


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
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    //cudaDeviceSynchronize();
}

__global__ void k_hadamard(const double *a, const double *b, double *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void k_sum(const double *a, const double *b, double *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_minus(const double *a, const double *b, double *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void k_scalar(const double *a, double s, double *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * s;
}

__global__ void k_transpose(const double *src, double *dst, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
        dst[row + col * rows] = src[col + row * cols];
}

__global__ void k_function(const double *src, double *dst, int n, int fn_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = apply_fn(src[i], fn_id);
}

