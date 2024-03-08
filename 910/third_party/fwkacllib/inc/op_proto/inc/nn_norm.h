/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file nn_norm.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type float16 or float32, with format NCHW for 4D.
* @li gamma: A Tensor of type float16 or float32. Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type float16 or float32. Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.0001".
* @li data_format: An optional string, specifying the format of "x".
Defaults to "NHWC".
* @li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True" . \n

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type float16 or float32 for the normalized "x",
with format NCHW for 4D.
* @li mean: A Tensor of type float16 or float32. Must be 1D. Specifies the mean of "x".
* @li variance: A Tensor of type float16 or float32. Must be 1D. Specifies the variance of "x". \n

* @attention Constraints:
* @li For Ascend 310, only support NCHW which can be trans to 5HD. \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.0001)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNorm)

/**
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type float16 or float32, with format NCHW for 4D.
* @li gamma: A Tensor of type float16 or float32. Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type float16 or float32. Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.0001".
* @li data_format: An optional string, specifying the format of "x".
Defaults to "NHWC".
* @li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True" . \n

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type float16 or float32 for the normalized "x",
with format NCHW for 4D.
* @li mean: A Tensor of type float16 or float32. Must be 1D. Specifies the mean of "x".
* @li rstd: A Tensor of type float16 or float32. Must be 1D. Specifies the rstd of "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNormV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.0001)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNormV2)

/**
 * @brief backward operator for group normalization. \n
 * @par Inputs:
 * Five input, including:
 * @li dy: A Tensor. Group grad. Must be one of the following types:
 *     float32, float16
 * @li mean: A Tensor. Mean of each group. Support float32, float16
 * @li rstd: A Tensor. Reciprocal standard deviation of each group. Support float32, float16
 * @li x: A Tensor. Specifies the offset. Support float32, float16
 * @li gamma: A Tensor. Specifies the scaling factor. Support float32, float16

 * @par Attributes:
 * @li num_groups: Int.Number specifying the number of group.
 * @li data_format: An optional String, Defaults to NCHW.
 * @li gamma_is_defined: An optional bool, controls whether to return dgamma and dbeta. Defaults to false.

 * @par Outputs:
 * Three output, including:
 * @li dx: A Tensor. Datatype and format is same as input_data. Data sorted.
 * @li dgamma: A Tensor. scale factor grad.
 * @li dbeta: A Tensor. offset factor grad.
 * @par Third-party framework compatibility
 * @li Compatible with the PyTorch operator GroupNorm.
 */

REG_OP(GroupNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(dx_is_require, Bool, true)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormGrad)

/**
* @brief Performs group normalization and silu. \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type bfloat16/float16/float32, with format NCHW for 4D.
* @li gamma: A Tensor of type bfloat16/float16/float32. Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type bfloat16/float16/float32. Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32/int64, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.0001".
* @li activate_silu: An optional bool.  Defaults to "true".

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type bfloat16/float16/float32 for the normalized "x",
with format NCHW for 4D.
* @li mean: A Tensor of type bfloat16/float16/float32. Must be 1D. Specifies the mean of "x".
* @li rstd: A Tensor of type bfloat16/float16/float32. Must be 1D. Specifies the rstd of "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm and Silu.

*/
REG_OP(GroupNormSilu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(eps, Float, 0.00001)
    .ATTR(activate_silu, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSilu)

/**
* @brief AddRmsNorm operator interface implementation
*  calculating: x1, x2, gamma
*  x = x1 + x2
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon))
*  y = gamma * (x * rstd)

* @par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li x2: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n

* @par Attributes:
* @li epsilon: A optional attribute, the type is float32. Defaults to 1e-6 . \n

* @par Outputs:
* Two outputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li rstd: A Tensor. Must be one of the following types: float32.
* @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
*/
REG_OP(AddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(AddRmsNorm)
/**
* @brief QuantizeAddLayerNorm operator interface implementation
* @par Inputs:
* @li x1: A tensor of type float16/bfloat16/float.
* @li x2: A tensor of type float16/bfloat16/float.
* @li gamma: A tensor of type float16/bfloat16/float.
* @li beta: A tensor of type float16/bfloat16/float.
* @li bias: A tensor of type float16/bfloat16/float.
* @li scales: A tensor of type bfloat16/float.
* @li zero_points: A optional tensor of type int8/uint8/bfloat16/int32.

* @par Attributes:
* @li dtype: A required int.
* @li axis: A optional int.
* @li epsilon: A optional float.
* @li additional_output: A optional bool.

* @par Outputs:
* @li y: A tensor of type int8/uint8/int32, describing the result. \n
* @li x: A tensor of type float16/bfloat16/float, describing the result. \n
*/
REG_OP(QuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(QuantizeAddLayerNorm)

/**
* @brief DuaQuantizeAddLayerNorm operator interface implementation
* @par Inputs:
* @li x1: A tensor of type float16/bfloat16/float.
* @li x2: A tensor of type float16/bfloat16/float.
* @li gamma: A tensor of type float16/bfloat16/float.
* @li beta: A tensor of type float16/bfloat16/float.
* @li bias: A tensor of type float16/bfloat16/float.
* @li scales1: A tensor of type bfloat16/float.
* @li scales2: A tensor of type bfloat16/float.
* @li zero_points1: A optional tensor of type int8/uint8/bfloat16/int32.
* @li zero_points2: A optional tensor of type int8/uint8/bfloat16/int32.

* @par Attributes:
* @li dtype: A required int.
* @li axis: A optional int.
* @li epsilon: A optional float.
* @li additional_output: A optional bool.

* @par Outputs:
* @li y1: A tensor of type int8/uint8/int32, describing the result. \n
* @li y2: A tensor of type int8/uint8/int32, describing the result. \n
* @li x: A tensor of type float16/bfloat16/float, describing the result. \n
*/
REG_OP(DuaQuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(scales1, ge::TensorType({DT_BF16, DT_FLOAT}))
    .INPUT(scales2, ge::TensorType({DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points1, ge::TensorType({DT_INT8, DT_UINT8, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(zero_points2, ge::TensorType({DT_INT8, DT_UINT8, DT_BF16, DT_INT32}))
    .OUTPUT(y1, ge::TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(y2, ge::TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(DuaQuantizeAddLayerNorm)

/**
* @brief InplaceAddRmsNorm operator interface implementation
*  calculating: x1, x2, gamma
*  x2 = x1 + x2
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon))
*  x1 = gamma * (x2 * rstd)

* @par Inputs:
* Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li x2: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li gamma: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n

* @par Attributes:
* @li epsilon: A optional attribute, the type is float32. Defaults to 1e-6 . \n

* @par Outputs:
* Two outputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li rstd: A Tensor. Must be one of the following types: float32.
* @li x2: A Tensor. Must be one of the following types: float16, float32, bfloat16.
*/
REG_OP(InplaceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(InplaceAddRmsNorm)

/*
* @brief Fused Operator of Add and LayerNorm . \n

* @par Inputs:
* @li x1: A tensor of type float16/bfloat16/float, describing the feature_map.
* @li x2: A tensor of type float16/bfloat16/float, describing the feature_map.
* @li gamma: A tensor of type float16/bfloat16/float, describing the feature_map.
* @li beta: A tensor of type float16/bfloat16/float, describing the feature_map.
* @li bias: A tensor of type float16/bfloat16/float, describing the feature_map.

* @par Attributes:
* @li epsilon: A optional float.
* @li additional_output: A optional bool.

* @par Outputs:
* @li x1: A tensor of type float16/bfloat16/float, describing the result. \n
* @li mean: A tensor of type float32, describing the result. \n
* @li rstd: A tensor of type float32, describing the result. \n
* @li x2: A tensor of type float16/bfloat16/float, describing the result. \n

*/
REG_OP(InplaceAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(mean, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(rstd, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-5)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(InplaceAddLayerNorm)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_