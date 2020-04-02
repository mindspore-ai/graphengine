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

#ifndef GE_OP_BITWISE_OPS_H_
#define GE_OP_BITWISE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Elementwise computes the bitwise right-shift of x and y.

*@par Inputs:
*The input x can be k-dimensional tensor, num_lower and num_upper can be zero-dimensional scalar. Inputs include: \n
* @li x:A Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, uint16, uint32, uint64. \n
* @li y:A Tensor. Must have the same type as x. \n

*@par Outputs:
*@li z:A Tensor. Has the same type as x. \n

*@attention Constraints:\n
*-The implementation for Unique on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(RightShift)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(z, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, \
            DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(RightShift)

}  // namespace ge

#endif  // GE_OP_BITWISE_OPS_H_
