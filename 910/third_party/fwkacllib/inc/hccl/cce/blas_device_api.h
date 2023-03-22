/*
Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef __BLAS_CPU_DEVICE_FUNC_API__
#define __BLAS_CPU_DEVICE_FUNC_API__

#include "blas_struct.h"

/**
 * [ccblas_device_hnrm2 hnrm2]
 * @param [in] n     [n number of x]
 * @param [in] x     [vector]
 * @param [in] inc_x [inc_x strid between elements of x]
 * @param result    [the resulting norm, which is 0.0 if n,incx<=0.0]
 * @return          [OK or ERROR]
 */

[aicpu] int32_t ccblas_device_hnrm2(int32_t n, const __fp16 *x, int32_t inc_x, __fp16* result);

/**
 * [ccblas_device_haxpy axpy]
 * @param [in] n     [number of elements in input vector(s)]
 * @param [in] alpha [the scalar alpha ]
 * @param [in] x     [vector,X is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )]
 * @param [in] incx  [incx strid between elements of x]
 * @param [in/out] y [vector,X is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )]
 * @param [in] incy  [incy strid between elements of y]
 */
[aicpu] int32_t ccblas_device_haxpy(int32_t n, const __fp16 alpha, const __fp16 * x, int32_t incx, __fp16 * y, int32_t incy);

/**
 * [ccblas_device_hasum]
 * @param [in] n     [n number of x]
 * @param [in] x     [vector]
 * @param [in] inc_x [inc_x strid between elements of x]
 * @param [out]result [sum of the absolute values of the elements of x
 * which is 0.0 if n,incx<=0.0]
 */
[aicpu] int32_t ccblas_device_hasum(int32_t n, const __fp16* x, int32_t incx, __fp16*  result);
/**
 * [ccblas_device_ihamax]
 * @param  [in] n     [number of x]
 * @param  [in] x     [vector]
 * @param  [in] inc_x [strid between elements of x]
 * @return [ou]       [the resulting norm, which is 0.0 if n,incx<=0.0]
 */
[aicpu] int32_t ccblas_device_ihamax(int32_t n, const __fp16 *x, int32_t inc_x, int32_t *result);

/**
 * [ccblas_device_hdot]
 * @param  [in] n     [number of x]
 * @param  [in] x     [vector]
 * @param  [in] inc_x [strid between elements of x]
 * @param  [in] y     [vector]
 * @param  [in] inc_y [strid between elements of y]
 * @return [ou] result      [the resulting norm, which is 0.0 if n<=0.0]
 */
[aicpu] int32_t ccblas_device_hdot(int32_t n, const __fp16 *x, int32_t inc_x, const __fp16 *y, int32_t inc_y, __fp16 *result);

/**
 * [ccblas_device_hger]
 * @param  [in] m     [row number of A]
 * @param  [in] n     [col number of x]
 * @param  [in] alpha     [ const num]
 * @param  [in] x     [vector]
 * @param  [in] inc_x [strid between elements of x]
 * @param  [in] y     [vector]
 * @param  [in] inc_y [strid between elements of y]
 * @param  [in/ou] a     [ matrix a]
 * @param  [in] lda [ the dimension of row of a]
*/
[aicpu] int32_t ccblas_device_hger(int32_t m, int32_t n, const __fp16 alpha, const __fp16 *x, int32_t inc_x, const __fp16 *y, int32_t inc_y, __fp16 *a, int32_t lda);

/**
 * [ccblas_device_hscal hscal]
 * @param [in]     n     [number of elements in the vector x]
 * @param [in]     alpha [scalar used for multiplication]
 * @param [in|out] x     [vector,x is REAL array]
 * @param [in]     incx  [incx stride between consecutive elements of x]
 * @return               [0:the execute norm,-1:error parameter]
 */
[aicpu] int32_t ccblas_device_hscal(int32_t n, const __fp16 alpha, __fp16* x, int32_t incx);


/**
 * [ccblas_device_hcopy hcopy]
 * @param [in] n      [number of x]
 * @param [in] x      [vector]
 * @param [in] incx  [strid between elements of x]
 * @param [out] y    [result vector]
 * @param [in] incy  [strid between elements of y]
 * @result               [-1 is param exception, 0 is success]
 */
[aicpu] int32_t ccblas_device_hcopy(int32_t n, const __fp16 *x, int32_t incx, __fp16 *y, int32_t incy);

/**
 * [ccblas_device_hgemv gemv]
 * @param trans [trans type,  operation op(A) that is non- or (conj.) transpose.]
 * @param m     [number of rows of matrix A.]
 * @param n     [number of colums of matrix A.]
 * @param alpha [scalar used for multiplication]
 * @param A     [array of dimension lda x n with lda >=max(1,m).
 *              Before entry, the leading m by n part of thearray A must contain the matrix of coefficients.
 *              Unchanged on exit.]
 * @param lda   [leading dimension of two-dimensional array used to store matrix A. lda must be at least max(1,m).]
 * @param x     [vector at least (1+(n-1)*abs(incx)) elements if transa==CCBLAS_OP_N and
 *              at least (1+(m-1)*abs(incx)) elements otherwise.]
 * @param incx  [stride between consecutive elements of x.]
 * @param beta  [scalar used for multiplication, if beta==0 then y does not have to be a valid input.]
 * @param y     [vector at least (1+(m-1)*abs(incy)) elements if transa==CUBLAS_OP_N and
 *              at least (1+(n-1)*abs(incy)) elements otherwise.]
 * @param incy  [stride between consecutive elements of y]
 */
[aicpu] int32_t ccblas_device_hgemv(ccblasOperation_t trans, int32_t m, int32_t n,
                                   const __fp16 alpha, const __fp16* A,
                                   int32_t lda, const __fp16* x, int32_t incx,
                                   const __fp16 beta, __fp16* y, int32_t incy);

/*
 * [ccblas_device_htrsv htrsv]
 * @param  [in] uplo   [matrix A is upper or lower]
 * @param  [in] trans  [op(A) is non- or transpose]
 * @param  [in] diag   [whether or not A is unit triangular]
 * @param  [in] n      [order of the matrix A]
 * @param  [in] A      [A is an array  for the matrix, dimension (lda,n)]
 * @param  [in] lda    [the first dimension of A]
 * @param  [in|out] x  [ Before entry, the incremented array x must contain the
 *                       n element right-hand side vector b. On exit, x is
 *                       overwritten with the solution vector x.description]
 * @param  [in] incx   [increment for the elements of x. incx must not be zero]
 * @result             [-1 is param exception, 0 is success]
 */
[aicpu] int32_t ccblas_device_htrsv(ccblasFillMode_t uplo, ccblasOperation_t trans, ccblasDiagType_t diag,
                            int32_t n, const __fp16 *A, int32_t lda, __fp16 *x, int32_t incx);

/**
 * [ccblas_device_hgemm gemm]
 * @param  [in] transa [trans type of matrix A]
 * @param  [in] transb [trans type of matrix B]
 * @param  [in] m      [number of rows of matrix A and matrix C]
 * @param  [in] n      [number of colums of matrix B and matrix C]
 * @param  [in] k      [number of colums of matrix A and matric B]
 * @param  [in] alpha  [scalar used for multiplication]
 * @param  [in] A      [array of dimension lda x k with lda >=max(1,m).
 * @param  [in] lda    [leading dimension of two-dimensional array used to store matrix A. lda must be at least max(1,m).]
 * @param  [in] B      [array of dimension ldb x n with lda >=max(1,k).
 * @param  [in] ldb    [leading dimension of two-dimensional array used to store matrix B. lda must be at least max(1,k).]
 * @param  [in] beta   [scalar used for multiplication, if beta==0 then C does not have to be a valid input.]
 * @param  [in|out] C  [array of dimension ldc x n with lda >=max(1,m).
 * @param  [in] ldc    [leading dimension of two-dimensional array used to store matrix C]
 * @result             [-1 is param exception, 0 is success]
 */
[aicpu] int32_t ccblas_device_hgemm(ccblasOperation_t transa, ccblasOperation_t transb,
                                  int32_t m, int32_t n, int32_t k, const __fp16 alpha,
                                  const __fp16* A, int32_t lda,
                                  const __fp16* B, int32_t ldb,
                                  const __fp16 beta,
                                  __fp16* C, int32_t ldc);

/**
 * [ccblas_device_hrotg rotg]
 * @param [in/out] a      [scalar]
 * @param [in/out] b      [scalar]
 * @param [out]    c      [cosine element of the rotation matrix]
 * @param [out]    s      [sine element of the rotation matrix]
 * @return                [-1 is param exception, 0 is success]
 */
[aicpu] int32_t ccblas_device_hrotg(__fp16* a, __fp16* b, __fp16* c, __fp16* s);

/**
 * [ccblas_device_ihamin iamin]
 * @param [in]    n       [number of elements in the vector x]
 * @param [in]    x       [vector]
 * @param [in]    inc     [strid between elements of x]
 * @param [out]   result  [the resulting index,which is 0 if n,incx<=0]
 * @return                [status]
 */
[aicpu] int32_t ccblas_device_ihamin(int32_t n, const __fp16 *x, int32_t incx, int32_t* result);

/**
 * [ccblas_device_htbsv htbsv]
 * @param  [in] uplo     [specifies whether the matrix is an upper or lower triangular matrix]
 * @param  [in] trans    [specifies whether the matrix is an transposed matrix]
 * @param  [in] diag     [specifies whether the elements on the main diagonal of matrix A are unity]
 * @param  [in] n        [number of elements in the vector x]
 * @param  [in] k        [If uplo = CCBLAS_FILL_MODE_UPPER, k specifies the number of super-diagona
                          ls of the matrix A. If uplo = CCBLAS_FILL_MODE_LOWER, k specifies the number
                          of sub-diagonals of the matrix A]
 * @param  [in] A        [the input matrix, its dimension is (lda, n), with lda > k]
 * @param  [in] lda      [specifies the first dimension of A]
 * @param  [in | out] x  [vector,x is REAL array,result will overwritten in x]
 * @param  [in] incx     [incx stride between consecutive elements of x]
 * @return               [0:success, -1:error param]
 */
[aicpu] int32_t ccblas_device_htbsv(ccblasFillMode_t uplo,  ccblasOperation_t trans, ccblasDiagType_t diag,
                                    int32_t n, int32_t k, const __fp16* a, int32_t lda,
                                    __fp16* x, int32_t incx);

/**
 * [ccblas_device_hrotmg hrotmg]
 * @param  [in | out] d1     [scalar that is overwritten on exit]
 * @param  [in | out] d2     [scalar that is overwritten on exit]
 * @param  [in | out] x1     [scalar that is overwritten on exit]
 * @param  [in]       y1     [scalar]
 * @param  [out]      param  [vector of 5 elements, where param[0] is flag, and param[1] - param[4] is matrix H]
 * @return                   [0: success, -1: error param]
 */
[aicpu] int32_t ccblas_device_hrotmg(__fp16 *d1, __fp16 *d2, __fp16 *x1, const __fp16 *y1, __fp16 *param);


/**
 * [ccblas_device_hswap swap]
 * @param [in]     n      [number of x]
 * @param [in|out] x      [vector]
 * @param [in]     incx   [strid between elements of x]
 * @param [in|out] y      [result vector]
 * @param [in]     incy   [strid between elements of y]
 * @return                [status]
 */
[aicpu] int32_t  ccblas_device_hswap(int32_t n, __fp16 *x, int32_t incx, __fp16 *y,int32_t incy);

/*
 * [ccblas_device_htpsv htpsv]
 * @param  [in] uplo   [matrix A is upper or lower]
 * @param  [in] trans  [op(A) is non- or transpose]
 * @param  [in] diag   [whether or not A is unit triangular]
 * @param  [in] n      [order of the matrix A]
 * @param  [in] AP     [AP is an array  for the matrix]
 * @param  [in|out] x  [ Before entry, the incremented array x must contain the
 *                       n element right-hand side vector b. On exit, x is
 *                       overwritten with the solution vector x.description]
 * @param  [in] incx   [increment for the elements of x. incx must not be zero]
 * @return             [status]
 */
[aicpu] int32_t ccblas_device_htpsv(ccblasFillMode_t uplo, ccblasOperation_t trans, ccblasDiagType_t diag,
                            int32_t n, const __fp16 *AP, __fp16 *x, int32_t incx);

/**
 * [ccblas_device_hgemv_ex gemv]
 * @param trans [trans type,  operation op(A) that is non- or (conj.) transpose.]
 * @param m     [number of rows of matrix A.]
 * @param n     [number of colums of matrix A.]
 * @param alpha [scalar used for multiplication]
 * @param A     [array of dimension lda x n with lda >=max(1,m).
 *              Before entry, the leading m by n part of thearray A must contain the matrix of coefficients.
 *              Unchanged on exit.]
 * @param lda   [leading dimension of two-dimensional array used to store matrix A. lda must be at least max(1,m).]
 * @param x     [vector at least (1+(n-1)*abs(incx)) elements if transa==CCBLAS_OP_N and
 *              at least (1+(m-1)*abs(incx)) elements otherwise.]
 * @param incx  [stride between consecutive elements of x.]
 * @param beta  [scalar used for multiplication, if beta==0 then y does not have to be a valid input.]
 * @param y     [vector at least (1+(m-1)*abs(incy)) elements if transa==CUBLAS_OP_N and
 *              at least (1+(n-1)*abs(incy)) elements otherwise.]
 * @param incy  [stride between consecutive elements of y]
 * @return ccStatus_t    [status]
 *
 */
[aicpu] int32_t ccblas_device_hgemv_ex(int trans, int32_t m, int32_t n,
                                   const __fp16 alpha, const __fp16* A,
                                   int32_t lda, const __fp16* x, int32_t incx,
                                   const __fp16 beta, __fp16* y, int32_t incy);

/**
 * [ccblas_device_hgemm_ex gemm]
 * @param  [in] transa [trans type of matrix A]
 * @param  [in] transb [trans type of matrix B]
 * @param  [in] m      [number of rows of matrix A and matrix C]
 * @param  [in] n      [number of colums of matrix B and matrix C]
 * @param  [in] k      [number of colums of matrix A and matric B]
 * @param  [in] alpha  [scalar used for multiplication]
 * @param  [in] A      [array of dimension lda x k with lda >=max(1,m).
 * @param  [in] lda    [leading dimension of two-dimensional array used to store matrix A. lda must be at least max(1,m).]
 * @param  [in] B      [array of dimension ldb x n with lda >=max(1,k).
 * @param  [in] ldb    [leading dimension of two-dimensional array used to store matrix B. lda must be at least max(1,k).]
 * @param  [in] beta   [scalar used for multiplication, if beta==0 then C does not have to be a valid input.]
 * @param  [in|out] C  [array of dimension ldc x n with lda >=max(1,m).
 * @param  [in] ldc    [leading dimension of two-dimensional array used to store matrix C]
 * @result             [-1 is param exception, 0 is success]
 */
[aicpu] int32_t ccblas_device_hgemm_ex(int transa,
                                      int transb, int32_t m,
                                      int32_t n, int32_t k,
                                      const __fp16 alpha,
                                      const __fp16 * A, int32_t lda,
                                      const __fp16 * B, int32_t ldb,
                                      const __fp16 beta, __fp16 * C,
                                      int32_t ldc);


#endif /*__BLAS_CPU_DEVICE_FUNC_API__*/
