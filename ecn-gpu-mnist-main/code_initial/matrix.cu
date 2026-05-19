#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// Variables globales GPU
double *d_m1 = nullptr;
double *d_m2 = nullptr;
double *d_res = nullptr;
size_t d_size_m1 = 0;
size_t d_size_m2 = 0;
size_t d_size_res = 0;

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc(sizeof(matrix_t));
    cudaMallocManaged(&res->m, columns * rows * sizeof(double));  
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    cudaFree(m->m);  // au lieu de free
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

/*
__global__
void matrix_sum_kernel(double *m1, double *m2, double *res, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size)
    {
        res[idx] = m1[idx] + m2[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int numBlocks = (size + 255) / 256;
    matrix_sum_kernel<<<numBlocks, threadsPerBlock>>>(m1->m, m2->m, res->m, size);
    cudaDeviceSynchronize();
}

__global__
void matrix_minus_kernel(double *m1, double *m2, double *res, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size)
    {
        res[idx] = m1[idx] - m2[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int numBlocks = (size + 255) / 256;
    matrix_minus_kernel<<<numBlocks, threadsPerBlock>>>(m1->m, m2->m, res->m, size);
    cudaDeviceSynchronize();
}
*/

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}

__global__
void matrix_dot_kernel(double *m1, double *m2, double *res,
                       int m1_rows, int m1_columns, int m2_columns)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (row < m1_rows && col < m2_columns)
    {
        double result = 0.0;
        
        for (int ii = 0; ii < m1_columns; ii++)
        {
            result += m1[ii + row * m1_columns] * m2[col + ii * m2_columns];
        }
        
        res[col + row * m2_columns] = result;
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows)   &&
           (m2->columns == res->columns));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m2->columns + 15) / 16,
                   (m1->rows    + 15) / 16);

    matrix_dot_kernel<<<numBlocks, threadsPerBlock>>>(
        m1->m, m2->m, res->m,
        m1->rows, m1->columns, m2->columns
    );
    cudaDeviceSynchronize();
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

/*__global__
void matrix_scalar_kernel(double *m1, double s, double *res, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size)
    {
        res[idx] = m1[idx] * s;  // multiplication 
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int numBlocks = (size + 255) / 256;

    matrix_scalar_kernel<<<numBlocks, threadsPerBlock>>>(m1->m, s, res->m, size);
    cudaDeviceSynchronize();
}
*/

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}