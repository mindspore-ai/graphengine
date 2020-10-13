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

#ifndef OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL_V3_H_
#define BUILT_IN_OP_PROTO_INC_MAX_POOL_V3_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Performs max pooling on the input . \n

* @par Inputs:
* One input:
* x: An NC1HWC0 Tensor. Supported type:float16, float32, double, int8, int16,
* int32, int64, uint8, uint16, qint8

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
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
* ksize[1] * ksize[2] <= 255.
* @li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
* strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
* @li "padding" is  "SAME" "VALID" or "CACULATE" .


* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool.
*/
REG_OP(MaxPoolV3)
    .INPUT(x,TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0,0,0,0})
    .ATTR(data_format, String, "NCHW")
    .ATTR(global_pooling,Bool,false)
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolV3)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MAX_POOL_V3_H_
