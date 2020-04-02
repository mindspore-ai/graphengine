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

#ifndef GE_OP_MATRIX_CALCULATION_OPS_H
#define GE_OP_MATRIX_CALCULATION_OPS_H

#include "../graph/operator_reg.h"

namespace ge {

/**
*@brief Multiplies matrix "a" by matrix "b", producing "a * b".

*@par Inputs:
*Two inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
* @li bias: A 1D Tensor. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC].

*@par Attributes:
*@li transpose_a: A bool. If True, changes the shape of "x1" from [M, K] to [K, M].
*@li transpose_b: A bool. If True, changes the shape of "x2" from [M, K] to [K, M].

*@par Outputs:
*y: The result matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32. Has format [ND, NHWC, FRACTAL_NZ].
*/
REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .OP_END_FACTORY_REG(MatMul)

REG_OP(MatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT16, DT_INT8, DT_INT8}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_FLOAT}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_FLOAT}))
    .OP_END_FACTORY_REG(MatMulV2)

/**
*@brief Multiplies matrix "a" by matrix "b", producing "a * b".

*@par Inputs:
*Three inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ].

*@par Attributes:
*@li adj_x: A bool. If True, changes the shape of "x1" from [B, M, K] to [B, K, M].
*@li adj_y: A bool. If True, changes the shape of "x2" from [B, M, K] to [B, K, M].

*@par Outputs:
*y: The result matrix Tensor. 2D or higher. Must be one of the following types: float16,
* float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ]. Has the same shape length as "x1" and "x2".
*/
REG_OP(BatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .ATTR(adj_x, Bool, false)
    .ATTR(adj_y, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMul)

REG_OP(MeanCCE)
    .INPUT(x, TensorType::ALL())
    .INPUT(indices, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(keep_dims, Bool, false)
    .ATTR(value1, ListInt, {})
    .ATTR(mode, Int, 3)                 // 0:max pooling or 1:avg pooling
    .ATTR(pad_mode, Int, 0)
    .ATTR(global_pooling, Bool, true)
    .ATTR(window, ListInt, {1,1})      // kernel size
    .ATTR(pad, ListInt, {0,0,0,0})     // pad size
    .ATTR(stride, ListInt, {1,1})      // stride size
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_mode, Int, 1)
    .ATTR(nan_opt, Int, 0)
    .ATTR(fomart, Int, 0)
    .OP_END_FACTORY_REG(MeanCCE)

REG_OP(MeanGrad)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(mode, Int, 1)                 // 0:max pooling or 1:avg pooling
    .ATTR(pad_mode, Int, 0)
    .ATTR(global_pooling, Bool, false)
    .ATTR(window, ListInt, {1,1})      // kernel size
    .ATTR(pad, ListInt, {0,0,0,0})     // pad size
    .ATTR(stride, ListInt, {1,1})      // stride size
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_mode, Int, 1)
    .ATTR(nan_opt, Int, 0)
    .ATTR(mean_grad_output_shape_value, ListInt, {1,1,1,1})
    .ATTR(mean_grad_output_shape_format, Int, 1) //must be NHWC
    .OP_END_FACTORY_REG(MeanGrad)

REG_OP(MatMulCCE)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .ATTR(has_bias, Bool, false)
    .OP_END_FACTORY_REG(MatMulCCE)

/**
*@brief Computes half the L2 norm of a tensor without the sqrt.

*@par Inputs:

* x: A Tensor.
*     TensorType::FloatingDataType().

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(L2Loss)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(L2Loss)

REG_OP(MatrixDiag)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiag)

REG_OP(MatrixDiagD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagD)

REG_OP(MatrixDiagPart)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPart)

REG_OP(MatrixDiagPartD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPartD)

REG_OP(MatrixSetDiag)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiag)

REG_OP(MatrixSetDiagD)
    .INPUT(x, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(assist, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiagD)

REG_OP(ScatterNdUpdate)
    .INPUT(var, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType::BasicType())
    .OUTPUT(var,  TensorType::BasicType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdUpdate)

REG_OP(ScatterAdd)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterAdd)

REG_OP(ScatterDiv)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterDiv)

REG_OP(ScatterNdAdd)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdAdd)

REG_OP(ScatterNdSub)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdSub)

REG_OP(ScatterSub)
    .INPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16, DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterSub)

REG_OP(DiagPartD)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(assist, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(DiagPartD)

REG_OP(DiagPart)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(DiagPart)

REG_OP(InnerProduct)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_INT8}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(num_output, Int)
    .ATTR(transpose, Bool, false)
    .ATTR(bias_term, Bool, true)
    .ATTR(axis, Int, 1)
    .ATTR(offset_a, Int, 0)
    .OP_END_FACTORY_REG(InnerProduct)

REG_OP(ConfusionMatrix)
    .INPUT(labels, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .INPUT(predictions, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(weights, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .REQUIRED_ATTR(num_classes, Int)
    .REQUIRED_ATTR(dtype, String)
    .OP_END_FACTORY_REG(ConfusionMatrix)

REG_OP(ScatterMul)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMul)

REG_OP(ScatterMin)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMin)

REG_OP(ScatterMax)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterMax)

REG_OP(SparseApplyAdagrad)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyAdagrad)

REG_OP(SparseApplyAdagradD)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(lr, Float)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyAdagradD)

REG_OP(ScatterUpdate)
    .INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterUpdate)

/**
* @brief Update relevant entries in '*var' according to the Ftrl-proximal scheme.
* That is for rows we have grad for, we update var, accum and linear

* @par Inputs:
* Ten inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li linear: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li indices: A vector of indices into the first dimension of var and accum.
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li l2_shrinkage: A Tensor of the same type as "var", L2 shrinkage regulariation. Must be a scalar.
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.

* @par Attributes:
* use_locking: An optional bool. Defaults to "False".
* If "True", updating of the "var" and "accum" tensors will be
* rotected by a lock; otherwise the behavior is undefined,
* but may exhibit less contention.

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var".
*/
REG_OP(SparseApplyFtrlV2)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(linear, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(l1, TensorType({DT_FLOAT}))
    .INPUT(l2, TensorType({DT_FLOAT}))
    .INPUT(l2_shrinkage, TensorType({DT_FLOAT}))
    .INPUT(lr_power, TensorType({DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyFtrlV2)

/**
* @brief Update relevant entries in '*var' according to the Ftrl-proximal scheme.
* That is for rows we have grad for, we update var, accum and linear

* @par Inputs:
* Ten inputs, including:
* @li var: A mutable Tensor. Must be of type TensorType::NumberType().
*     Should be a Variable Tensor.
* @li accum: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li linear: A mutable Tensor of the same type as "var".
*     Should be a Variable Tensor.
* @li grad: A Tensor of the same type as "var", for the gradient.
* @li indices: A vector of indices into the first dimension of var and accum.

* @par Attributes:
* @li lr: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li l1: A Tensor of the same type as "var", for L1 regulariation. Must be a scalar.
* @li l2: A Tensor of the same type as "var", for L2 regulariation. Must be a scalar.
* @li l2_shrinkage: A Tensor of the same type as "var", L2 shrinkage regulariation. Must be a scalar.
* @li lr_power: A Tensor of the same type as "var", for the scaling factor. Must be a scalar.
* @li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" and "accum" tensors will be
*     rotected by a lock; otherwise the behavior is undefined,
*     but may exhibit less contention.

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var".
*/
REG_OP(SparseApplyFtrlV2D)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(accum, TensorType({DT_FLOAT}))
    .INPUT(linear, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(lr, Float)
    .REQUIRED_ATTR(l1, Float)
    .REQUIRED_ATTR(l2, Float)
    .REQUIRED_ATTR(l2_shrinkage, Float)
    .REQUIRED_ATTR(lr_power, Float)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(SparseApplyFtrlV2D)

}  // namespace ge

#endif  // GE_OP_MATRIX_CALCULATION_OPS_H
