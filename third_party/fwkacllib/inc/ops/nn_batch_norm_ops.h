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

#ifndef GE_OP_NN_BATCH_NORM_OPS_H
#define GE_OP_NN_BATCH_NORM_OPS_H

#include "../graph/operator_reg.h"

namespace ge {

/**
*@brief A fusion operator for batchnorm.

*@par Inputs:
*Ten inputs, including:
* @li x: A Tensor. Must be one of the following types: float32.
* @li scale: A Tensor. Must be one of the following types: float32.
* @li b: A Tensor. Must be one of the following types: float32.
* @li mean: A Tensor. Must be one of the following types: float32.
* @li variance: A Tensor. Must be one of the following types: float32.

*@par Attributes:
* @li mode: A Tensor. Must be one of the following types: int.
* @li epsilon: A Tensor. Must be one of the following types: float32.
* @li momentum: A Tensor. Must be one of the following types: float32.
* @li is_training: A Tensor. Must be one of the following types: bool.
* @li is_training_fusion: A Tensor. Must be one of the following types: bool.
* @li moving_average_fraction: A Tensor. Must be one of the following types: float32.

*@par Outputs:
*Three outputs, including:
* @li y: A Tensor. Must be one of the following types: float32.
* @li running_mean: A Tensor. Must be one of the following types: float32.
* @li running_variance: A Tensor. Must be one of the following types: float32.
* @li save_mean: A Tensor. Must be one of the following types: float32.
* @li save_inv_variance: A Tensor. Must be one of the following types: float32.
* @li save_inv_variance1: A Tensor. Must be one of the following types: float32.

*/
REG_OP(FusedBatchNorm)
    .INPUT(x, TensorType{DT_FLOAT})
    .INPUT(scale, TensorType{DT_FLOAT})
    .INPUT(b, TensorType{DT_FLOAT})
    .INPUT(mean, TensorType{DT_FLOAT})
    .INPUT(variance, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType{DT_FLOAT})
    .OUTPUT(running_mean, TensorType{DT_FLOAT})
    .OUTPUT(running_variance, TensorType{DT_FLOAT})
    .OUTPUT(save_mean, TensorType{DT_FLOAT})
    .OUTPUT(save_inv_variance, TensorType{DT_FLOAT})
    .OUTPUT(save_inv_variance1, TensorType{DT_FLOAT})
    .ATTR(mode, Int, 1)
    .ATTR(epsilon, Float,  1e-5f)
    .ATTR(momentum, Float, 0.9)
    .ATTR(is_training, Bool, true)
    .ATTR(is_training_fusion, Bool, true)
    .ATTR(moving_average_fraction, Float, 0.00300002098)
    .OP_END_FACTORY_REG(FusedBatchNorm)

/**
*@brief A fusion operator for batchnorm.

*@par Inputs:
*Ten inputs, including:
* @li dy: A Tensor. Must be one of the following types: float32.
* @li x: A Tensor. Must be one of the following types: float32.
* @li scale: A Tensor. Must be one of the following types: float32.
* @li save_mean: A Tensor. Must be one of the following types: float32.
* @li save_inv_variance: A Tensor. Must be one of the following types: float32.
* @li save_inv_variance1: A Tensor. Must be one of the following types: float32.

*@par Attributes:
* @li epsilon: A Tensor. Must be one of the following types: float32.
* @li momentum: A Tensor. Must be one of the following types: float32.

*@par Outputs:
*Three outputs, including:
* @li dx: A Tensor. Must be one of the following types: float32.
* @li bn_scale: A Tensor. Must be one of the following types: float32.
* @li bn_bias: A Tensor. Must be one of the following types: float32.
*/

REG_OP(FusedBatchNormGrad)
    .INPUT(dy, TensorType{DT_FLOAT})
    .INPUT(x, TensorType{DT_FLOAT})
    .INPUT(scale, TensorType{DT_FLOAT})
    .INPUT(save_mean, TensorType{DT_FLOAT})
    .INPUT(save_inv_variance, TensorType{DT_FLOAT})
    .INPUT(save_inv_variance1, TensorType{DT_FLOAT})
    .OUTPUT(dx, TensorType{DT_FLOAT})
    .OUTPUT(bn_scale, TensorType{DT_FLOAT})
    .OUTPUT(bn_bias, TensorType{DT_FLOAT})
    .ATTR(epsilon, Float, 0.0)
    .ATTR(momentum, Float, 0.0)
    .OP_END_FACTORY_REG(FusedBatchNormGrad)

REG_OP(L2Normalize)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {})
    .ATTR(eps, Float, 1e-4)
    .OP_END_FACTORY_REG(L2Normalize)

REG_OP(L2NormalizeGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(dim, ListInt, {})
    .ATTR(eps, Float, 0.0001)
    .OP_END_FACTORY_REG(L2NormalizeGrad)

REG_OP(BatchNorm)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm)

REG_OP(BatchNormExt2)
    .INPUT(input_x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(input_scale, TensorType({DT_FLOAT}))
    .INPUT(input_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(input_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output_mean, TensorType({DT_FLOAT}))
    .OUTPUT(output_variance, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(output_reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNormExt2)

REG_OP(BatchNormGrad)
    .INPUT(y_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_5, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNormGrad)

REG_OP(BatchNormGradExt2)
    .INPUT(y_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .INPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(scale_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(offset_backprop, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_4, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BatchNormGradExt2)

REG_OP(BninferenceD)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(scale_factor, Float,0.999)
    .ATTR(epsilon, Float,1e-5f)
    .ATTR(moving_average_fraction, Float,0.999)
    .ATTR(use_global_stats, Bool,true)
    .OP_END_FACTORY_REG(BninferenceD)
REG_OP(Bninference)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale_factor, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(epsilon, Float,1e-5f)
    .ATTR(moving_average_fraction, Float,0.999)
    .ATTR(use_global_stats, Bool,true)
    .OP_END_FACTORY_REG(Bninference)

}  // namespace ge

#endif  // GE_OP_NN_BATCH_NORM_OPS_H
