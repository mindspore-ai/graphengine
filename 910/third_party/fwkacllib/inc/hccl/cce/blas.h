#ifndef __CC_BLAS_API__
#define __CC_BLAS_API__

#include "cce/cce.h"
#include "cce/blas_struct.h"

namespace cce {


/**
 * [ccblasIhamax]
 * @param [in] handle
 * @param [in] n      [number of x]
 * @param [in] x      [vector]
 * @param [in] inc_x  [strid between elements of x]
 * @param [out] result [the resulting norm, which is 0.0 if n,incx<=0.0]
 */
ccStatus_t ccblasIhamax(ccHandle_t handle, int32_t n, const fp16_t* x, int32_t incx, int32_t* result);
/**
 * [ccblasHdot]
 * @param [in] handle
 * @param [in] n      [number of x]
 * @param [in] x      [vector]
 * @param [in] inc_x  [strid between elements of x]
 * @param [in] y      [vector]
 * @param [in] inc_y  [strid between elements of y]
 * @param [out] result [the resulting norm, which is 0.0 if n,incx<=0.0]
 */
ccStatus_t ccblasHdot(ccHandle_t handle, int32_t n, const fp16_t* x, int32_t incx, const fp16_t* y, int32_t incy, fp16_t* result);
/**
 * [ccblasHger]
 * @param  [in] handle [hanlder]
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
ccStatus_t ccblasHger(ccHandle_t handle,int32_t m, int32_t n, const fp16_t* alpha, const fp16_t* x, int32_t incx, const fp16_t* y, int32_t incy, fp16_t *a, int32_t lda);

/**
 * [ccblasHnrm2 nrm2]
 * @param  [in] handle [hanlder]
 * @param  [in] n      [number of x]
 * @param  [in] x      [vector]
 * @param  [in] incx   [strid between elements of x]
 * @param  [out] result [the resulting norm, which is 0.0 if n,incx<=0.0]
 * @return		   [sttus]
 */
ccStatus_t ccblasHnrm2(ccHandle_t handle, int32_t n, const fp16_t* x, int32_t incx, fp16_t* result);

/**
 * [ccblasHtrsv htrsv]
 * @param  [in] handle [handle]
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
 * @return             [status]
 */
ccStatus_t ccblasHtrsv(ccHandle_t handle, ccblasFillMode_t uplo, ccblasOperation_t trans,
        ccblasDiagType_t diag, int32_t n, const fp16_t* A, int32_t lda, fp16_t* x, int32_t incx);

/**
 * [ccblasHaxpy axpy]
 * @param [in] handle [hanlder]
 * @param [in] n     [number of elements in input vector(s)]
 * @param [in] alpha [the scalar alpha ]
 * @param [in] x     [vector,X is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )]
 * @param [in] incx  [incx strid between elements of x]
 * @param [in/out] y [vector,X is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )]
 * @param [in] incy  [incy strid between elements of y]
 * @return           [ccStatus_t]
 */

ccStatus_t ccblasHaxpy(ccHandle_t handle, int32_t n, const fp16_t *alpha, const fp16_t * x,int32_t incx, fp16_t * y, int32_t incy);


/**
 * [ccblasHasum asum]
 * @param [in] handle [hanlder]
 * @param [in] n     [n number of x]
 * @param [in] x     [vector]
 * @param [in] inc_x [inc_x strid between elements of x]
 * @param [out]result [sum of the absolute values of the elements of x
 * which is 0.0 if n,incx<=0.0]
 * @return [status]
 */
ccStatus_t ccblasHasum(ccHandle_t handle, int32_t n, const fp16_t* x, int32_t incx, fp16_t* result);


/**
 * [ccblasHscal hscal]
 * @param [in]     handle [hanlder]
 * @param [in]     n      [number of elements in the vector x]
 * @param [in]     alpha  [scalar used for multiplication]
 * @param [in|out] x      [vector,x is REAL array]
 * @param [in]     incx   [incx stride between consecutive elements of x]
 * @return                [CC_STATUS_BAD_PARAM | CC_STATUS_SUCCESS | KERNEL_ERROR_INTERNAL_ERROR]
 */
ccStatus_t ccblasHscal(ccHandle_t handle, int32_t n, const fp16_t* alpha, fp16_t* x, int32_t incx);

/**
 * [ccblasHcopy hcopy]
 * @param [in] handle [handle to the ccblas library context]
 * @param [in] n      [number of x]
 * @param [in] x      [vector]
 * @param [in] incx  [strid between elements of x]
 * @param [out] y    [result vector]
 * @param [in] incy  [strid between elements of y]
 * @return               [CC_STATUS_BAD_PARAM | CC_STATUS_SUCCESS]
 */
ccStatus_t ccblasHcopy(ccHandle_t handle, int32_t n, const fp16_t *x, int32_t incx, fp16_t *y,int32_t incy);

/**
 * [ccblasHgemv description]
 * @param  [in] handle [hanlder]
 * @param  [in] trans  [trans type]
 * @param  [in] m      [number of rows of matrix A.]
 * @param  [in] n      [number of colums of matrix A.]
 * @param  [in] alpha  [scalar used for multiplication]
 * @param  [in] A      [array of dimension lda x n with lda >=max(1,m).
 *                      Before entry, the leading m by n part of thearray A must contain the matrix of coefficients.
 *                      Unchanged on exit.]
 * @param  [in] lda    [leading dimension of two-dimensional array used to store matrix A. lda must be at least max(1,m).]
 * @param  [in] x      [vector at least (1+(n-1)*abs(incx)) elements if transa==CCBLAS_OP_N and
 *                      at least (1+(m-1)*abs(incx)) elements otherwise.]
 * @param  [in] incx   [stride between consecutive elements of x.]
 * @param  [in] beta   [scalar used for multiplication, if beta==0 then y does not have to be a valid input.]
 * @param  [in|out] y  [vector at least (1+(m-1)*abs(incy)) elements if transa==CUBLAS_OP_N and
 *                      at least (1+(n-1)*abs(incy)) elements otherwise.]
 * @param  [in] incy   [stride between consecutive elements of y]
 * @return        [status]
 */
ccStatus_t ccblasHgemv(ccHandle_t handle, ccblasOperation_t trans, int32_t m, int32_t n, const fp16_t* alpha, const fp16_t* A,
                       int32_t lda, const fp16_t* x, int32_t incx, const fp16_t* beta, fp16_t* y, int32_t incy);

/**
 * [ccblasHgemm description]
 * @param  [in] handle [hanlder]
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
 * @return        [status]
 */
ccStatus_t ccblasHgemm(ccHandle_t handle, ccblasOperation_t transa, ccblasOperation_t transb, int32_t m, int32_t n, int32_t k,
                       const fp16_t* alpha, const fp16_t* A, int32_t lda, const fp16_t* B, int32_t ldb, const fp16_t* beta,
                       fp16_t* C, int32_t ldc);

/**
 * [ccblasHrotg rotg]
 * @param [in]    handle  [handle to the cceblase library context]
 * @param [in/out] a      [scalar]
 * @param [in/out] b      [scalar]
 * @param [out]    c      [cosine element of the rotation matrix]
 * @param [out]    s      [sine element of the rotation matrix]
 * @return                [status]
 */
ccStatus_t ccblasHrotg(ccHandle_t handle, fp16_t* a, fp16_t* b, fp16_t* c, fp16_t* s);

/**
 * [ccblasIhamin iamin]
 * @param [in]    handle  [handle to the cceblase library context]
 * @param [in]    n       [number of elements in the vector x]
 * @param [in]    x       [vector]
 * @param [in]    inc     [strid between elements of x]
 * @param [out]   result  [the resulting index,which is 0 if n,incx<=0]
 * @return                [status]
 */
ccStatus_t ccblasIhamin(ccHandle_t handle, int32_t n, const fp16_t* x, int32_t incx, int32_t* result);

/**
 * [ccblasHtbsv htbsv]
 * @param  [in] handle   [handle]
 * @param  [in] uplo     [specifies whether the matrix is an upper or lower triangular matrix]
 * @param  [in] trans    [specifies whether the matrix is an transposed matrix]
 * @param  [in] diag     [specifies whether the elements on the main diagonal of matrix A are unity]
 * @param  [in] n        [number of elements in the vector x]
 * @param  [in] k        [If uplo = CCBLAS_FILL_MODE_UPPER, k specifies the number of super-diagona
                          ls of the matrix A. If uplo = CCBLAS_FILL_MODE_LOWER, k specifies the number
                          of sub-diagonals of the matrix A]
 * @param  [in] A        [the input matrix, its dimension is (lda, n), with lda > k]
 * @param  [in] lda      [specifies the first dimension of A]
 * @param  [in | out] x    [vector,x is REAL array,result will overwritten in x]
 * @param  [in] incx     [incx stride between consecutive elements of x]
 * @return               [CC_STATUS_BAD_PARAM | CC_STATUS_SUCCESS]
 */
ccStatus_t ccblasHtbsv(ccHandle_t handle, ccblasFillMode_t uplo, ccblasOperation_t trans, ccblasDiagType_t diag, int32_t n, int32_t k,
                        const fp16_t *a, int32_t lda, fp16_t *x, int32_t incx);

/**
 * [ccblasHrotmg hrotmg]
 * @param  [in] handle [handle]
 * @param  [in | out] d1     [scalar that is overwritten on exit]
 * @param  [in | out] d2     [scalar that is overwritten on exit]
 * @param  [in | out] x1     [scalar that is overwritten on exit]
 * @param  [in]       y1     [scalar]
 * @param  [out]      param  [vector of 5 elements, where param[0] is flag, and param[1] - param[4] is matrix H]
 * @return                   [CC_STATUS_BAD_PARAM | CC_STATUS_SUCCESS]
 */
ccStatus_t ccblasHrotmg(ccHandle_t handle, fp16_t *d1, fp16_t *d2, fp16_t *x1,const fp16_t *y1, fp16_t *param);

/**
 * [ccblasHswap swap]
 * @param [in]     handle [handle to the ccblas library context]
 * @param [in]     n      [number of x]
 * @param [in|out] x      [vector]
 * @param [in]     incx   [strid between elements of x]
 * @param [in|out] y      [result vector]
 * @param [in]     incy   [strid between elements of y]
 * @return                [CC_STATUS_BAD_PARAM | CC_STATUS_SUCCESS]
 */
ccStatus_t ccblasHswap(ccHandle_t handle, int32_t n, fp16_t *x, int32_t incx, fp16_t *y,int32_t incy);

/**
 * [ccblasHtpsv htpsv]
 * @param  [in] handle [handle]
 * @param  [in] uplo   [matrix A is upper or lower]
 * @param  [in] trans  [op(A) is packed store]
 * @param  [in] diag   [whether or not A is unit triangular]
 * @param  [in] n      [order of the matrix A]
 * @param  [in] AP     [AP is an array  for the matrix, dimension (lda,n)]
 * @param  [in|out] x  [ Before entry, the incremented array x must contain the
 *                       n element right-hand side vector b. On exit, x is
 *                       overwritten with the solution vector x.description]
 * @param  [in] incx   [increment for the elements of x. incx must not be zero]
 * @return             [status]
 */
ccStatus_t ccblasHtpsv(ccHandle_t handle, ccblasFillMode_t uplo, ccblasOperation_t trans,
        ccblasDiagType_t diag, int32_t n, const fp16_t* AP, fp16_t* x, int32_t incx);


};
#endif /*__CC_BLAS_API__*/
