/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GE_OP_LINALG_OPS_H_
#define GE_OP_LINALG_OPS_H_

#include "graph/operator_reg.h"
#include "../graph/operator.h"

namespace ge {

REG_OP(CholeskyGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(CholeskyGrad)

REG_OP(Cholesky)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Cholesky)

REG_OP(LogMatrixDeterminant)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(sign, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(LogMatrixDeterminant)

REG_OP(MatrixDeterminant)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(MatrixDeterminant)

REG_OP(MatrixInverse)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(adjoint, Bool, false)
    .OP_END_FACTORY_REG(MatrixInverse)

REG_OP(MatrixSolve)
    .INPUT(matrix, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rhs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(adjoint, Bool, false)
    .OP_END_FACTORY_REG(MatrixSolve)

REG_OP(MatrixSolveLs)
    .INPUT(matrix, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rhs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(l2, TensorType({DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(fast, Bool, true)
    .OP_END_FACTORY_REG(MatrixSolveLs)

REG_OP(MatrixTriangularSolve)
    .INPUT(matrix, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(rhs, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(lower, Bool, true)
    .ATTR(adjoint, Bool, false)
    .OP_END_FACTORY_REG(MatrixTriangularSolve)

REG_OP(Qr)
    .INPUT(x, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(q, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(r, TensorType({ DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .ATTR(full_matrices, Bool, false)
    .OP_END_FACTORY_REG(Qr)

REG_OP(SelfAdjointEig)
    .INPUT(x, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(eigen_value, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(eigen_vector, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .ATTR(compute_v, Bool, true)
    .OP_END_FACTORY_REG(SelfAdjointEig)

REG_OP(Svd)
    .INPUT(x, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(sigma, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(u, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .OUTPUT(v, TensorType({ DT_DOUBLE, DT_FLOAT }))
    .ATTR(compute_uv, Bool, true)
    .ATTR(full_matrices, Bool, false)
    .OP_END_FACTORY_REG(Svd)
}  // namespace ge

#endif  // GE_OP_LINALG_OPS_H_
