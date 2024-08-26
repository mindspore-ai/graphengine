/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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

/*!
 * \file nn_optimizer.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OPTIMIZER_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OPTIMIZER_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Computes the ApplyCamePart1.
*
* @par Inputs:
* including:
* @li grad: A mutable Tensor with rank 2, such as [n, m] , support types: 
* float16, float32, bfloat16.
* @li eps: A mutable Tensor with rank 1, such as n , support types: float32. \n
*
* @par Outputs:
* @li sum_grad_r: A mutable Tensor with rank 1, such as [n, 1], support types: 
* float32.
* @li sum_grad_c: A mutable Tensor with rank 1, such as [1, m], support types: 
* float32.
* @li sum_grad_rc: A mutable Tensor with randk 0, such as [1, 1], support 
* types: float32. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ApplyCamePart1)
    .INPUT(grad, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16}))
    .INPUT(eps, TensorType({DT_FLOAT}))
    .OUTPUT(sum_grad_r, TensorType({DT_FLOAT}))
    .OUTPUT(sum_grad_c, TensorType({DT_FLOAT}))
    .OUTPUT(sum_grad_rc, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ApplyCamePart1)


/**
* @brief Computes the ApplyCamePart2.
*
* @par Inputs:
* including:
* @li grad: A multi-dimensional Tensor of type bfloat16, float16 or float32.
* @li sum_grad_r: A 1-dimensional Tensor of type float32.
* @li sum_grad_c: A 1-dimensional Tensor of type float32.
* @li sum_grad_rc: A 1-dimensional Tensor of type float32.
* @li r: A 1-dimensional Tensor of type bfloat16, float16 or float32.
* @li c: A 1-dimensional Tensor of type bfloat16, float16 or float32.
* @li beta2: A 1-dimensional Tensor of type float32.
* @li sum_r: A 1-dimensional Tensor of type float32.
* @li global_shape: A 1-dimensional Tensor, specifying the original shape M, N. \n
*
* @par Outputs:
* @li r: A mutable tensor. Must have the same type as input "r".
* @li c: A mutable tensor. Must have the same type as input "c".
* @li u: A mutable tensor.
* @li sum_square_u: A mutable tensor. \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ApplyCamePart2)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum_grad_r, TensorType({DT_FLOAT}))
    .INPUT(sum_grad_c, TensorType({DT_FLOAT}))
    .INPUT(sum_grad_rc, TensorType({DT_FLOAT}))
    .INPUT(r, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sum_r, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(global_shape, TensorType({DT_INT64}))
    .OUTPUT(r, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(c, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(u, TensorType({DT_FLOAT}))
    .OUTPUT(sum_square_u, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ApplyCamePart2)


/**
* @brief Computes the ApplyCamePart3.
*
* @par Inputs:
* including:
* @li u: A 2-dimensional Tensor of type float32.
* @li m: A 2-dimensional Tensor of type bfloat16, float16 or float32.
* @li eps: A 1-dimensional Tensor, specifying the epsilon value.
* @li beta1: A 1-dimensional Tensor, specifying the beta1 value.
* @li clip_threshold: A 1-dimensional Tensor, specifying the clip_threshold 
* value.
* @li sum_square_u: A 1-dimensional Tensor, specifying the sum_square_u value.
* @li global_shape: A 1-dimensional Tensor, specifying the original shape M, N. \n
*
* @par Attributes:
* use_first_moment: A bool Scalar. If true, update the computed output m. 
* Default to false. \n
*
* @par Outputs:
* @li m: A multi-dimensional tensor. Must have the same type as input "m".
* @li sum_u_r:  A 1-dimensional tensor. Must have the same type as input "u".
* @li sum_u_c:  A 1-dimensional tensor. Must have the same type as input "u".
* @li sum_u_rc: A 1-dimensional tensor. Must have the same type as input "u". \n
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ApplyCamePart3)
    .INPUT(u, TensorType({DT_FLOAT}))
    .INPUT(m, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(eps, TensorType({DT_FLOAT}))
    .INPUT(beta1, TensorType({DT_FLOAT}))
    .INPUT(clip_threshold, TensorType({DT_FLOAT}))
    .INPUT(sum_square_u, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(global_shape, TensorType({DT_INT64}))
    .OUTPUT(m, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sum_u_r, TensorType({DT_FLOAT}))
    .OUTPUT(sum_u_c, TensorType({DT_FLOAT}))
    .OUTPUT(sum_u_rc, TensorType({DT_FLOAT}))
    .ATTR(use_first_moment, Bool, false)
    .OP_END_FACTORY_REG(ApplyCamePart3)


/**
* @brief ApplyCamePart4.
*
* @par Inputs:
* including:
* @li param: A multi-dimensional Tensor of type bfloat16, float16 or float32.
* @li m: A multi-dimensional Tensor of type bfloat16, float16 or float32.
* @li r: A 1-dimensional Tensor of type bfloat16, float16 or float32.
* @li c: A 1-dimensional Tensor of type bfloat16, float16 or float32.
* @li weight_decay: A 1-dimensional Tensor of type float32.
* @li lr: A 1-dimensional Tensor of type float32.
* @li beta3: A 1-dimensional Tensor of type float32.
* @li sum_r: A 1-dimensional Tensor of type float32.
* @li sum_u_r: A 1-dimensional Tensor of type bfloat16, float16 or float32.
* @li sum_u_c: A 1-dimensional Tensor of type bfloat16, float16 or float32.
* @li sum_u_rc: A 1-dimensional Tensor of type float32.
* @li global_shape: A 1-dimensional Tensor, specifying the original shape M, N. \n
*
* @par Outputs:
* @li param: A mutable tensor.
* @li r: A mutable tensor. Must have the same type as input "r".
* @li c: A mutable tensor. Must have the same type as input "c".
*
* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(ApplyCamePart4)
    .INPUT(param, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(m, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(r, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(weight_decay, TensorType({DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(beta3, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sum_r, TensorType({DT_FLOAT}))
    .INPUT(sum_u_r, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum_u_c, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum_u_rc, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(global_shape, TensorType({DT_INT64}))
    .OUTPUT(param, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(r, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(c, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(ApplyCamePart4)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_OPTIMIZER_H_
