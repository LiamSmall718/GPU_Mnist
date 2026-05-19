#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// déclaration forward — défini dans matrix_dot.cu
__global__ void k_function(const double *src, double *dst, int n, int fn_id);
__global__ void k_hadamard(const double*, const double*, double*, int);
__global__ void k_sum(const double*, const double*, double*, int);
__global__ void k_minus(const double*, const double*, double*, int);
__global__ void k_scalar(const double*, double, double*, int);
__global__ void k_transpose(const double*, double*, int, int);

#ifdef __cplusplus
extern "C" {
#endif

extern double sigmoid(double x);
extern double dsigmoid(double x);

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    //res->m = (double *) calloc(columns * rows, sizeof(double));
    cudaMallocManaged(&res->m, columns * rows * sizeof(double));
    cudaMemset(res->m, 0, columns * rows * sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    //free(m->m);
    cudaFree(m->m);
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

void matrix_dot_cpu(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}


/*
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

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
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

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}
*/

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert(m1->columns == m2->columns && m1->rows == m2->rows &&
           m1->columns == res->columns && m1->rows == res->rows);
    int n = m1->rows * m1->columns;
    k_hadamard<<<(n+255)/256, 256>>>(m1->m, m2->m, res->m, n);
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert(m1->columns == m2->columns && m1->rows == m2->rows &&
           m1->columns == res->columns && m1->rows == res->rows);
    int n = m1->rows * m1->columns;
    k_sum<<<(n+255)/256, 256>>>(m1->m, m2->m, res->m, n);
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert(m1->columns == m2->columns && m1->rows == m2->rows &&
           m1->columns == res->columns && m1->rows == res->rows);
    int n = m1->rows * m1->columns;
    k_minus<<<(n+255)/256, 256>>>(m1->m, m2->m, res->m, n);
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert(m1->rows == res->rows && m1->columns == res->columns);
    int n = m1->rows * m1->columns;
    k_scalar<<<(n+255)/256, 256>>>(m1->m, s, res->m, n);
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert(m1->columns == res->rows && m1->rows == res->columns);
    dim3 threads(16, 16);
    dim3 blocks((m1->columns+15)/16, (m1->rows+15)/16);
    k_transpose<<<blocks, threads>>>(m1->m, res->m, m1->rows, m1->columns);
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert(dest->rows == src->rows && dest->columns == src->columns);
    cudaMemcpy(dest->m, src->m,
               src->rows * src->columns * sizeof(double),
               cudaMemcpyDefault);
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert((m1->columns == res->columns) && (m1->rows == res->rows));
    int n = m1->rows * m1->columns;
    int blocks = (n + 255) / 256;

    if (f == sigmoid)
        k_function<<<blocks, 256>>>(m1->m, res->m, n, FN_SIGMOID);
    else if (f == dsigmoid)
        k_function<<<blocks, 256>>>(m1->m, res->m, n, FN_DSIGMOID);
    else {
        cudaDeviceSynchronize();
        for (int i = 0; i < n; i++) res->m[i] = f(m1->m[i]);
    }
}

#ifdef __cplusplus
}
#endif