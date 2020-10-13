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

/*!
 *\file max_pool_v3_grad.h
 *\brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL_V3_GRAD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL_V3_GRAD_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes gradients of the maxpooling function . \n

* @par Inputs:
* @li orig_input: A mutable NC1HWC0 tensor of type RealNumberType.
* @li orig_output: A mutable NC1HWC0 tensor of type RealNumberTypex.
* @li grad: A mutable NC1HWC0 tensor of type RealNumberType . \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor. No default value.
* @li padding_mode: A required string. Defaults to "CALCULATED".
* @li pads:A required list of int8, int16, int32, or int64 values,
* a data to caculate when padding_mode is "SAME" and "CALCULATED".
* @li data_format: An optional string. Defaults to "NHWC" .
* @li global_pooling bool, Whether to use the global pooling.
* If global_pooling = true, kernel size and paddings will be ignored.
* Default False
* @li ceil_mode:global_pooling (bool) – (bool) Whether to use the global pooling.
* If global_pooling = true, kernel size and paddings will be ignored.
* Default False \n

* @par Outputs:
* y: A mutable tensor. Has the same shape and type as "x1" . \n

* @attention Constraints:
* @li Computing gradients of global pooling is not supported, which means
* "ksize < x1".
* @li "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGrad.
*/
REG_OP(MaxPoolV3Grad)
    .INPUT(orig_input, TensorType::RealNumberType())
    .INPUT(orig_output, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(out_grad, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mod, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling, Bool, false)
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolV3Grad)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL_V3_GRAD_H_
