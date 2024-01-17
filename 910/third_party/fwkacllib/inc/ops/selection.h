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
 * \file selection.h
 * \brief
 */


#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief According to the indices and indices_mask, return the value.

* @par Inputs:
* Four inputs, including:
* @li x: A ND Tensor.
* @li indices: Dynamic input. A ND Tensor of int64. return the value according to the indices.

* @par Attributes:
* @li indices_mask: A list int. Indicates which dimensions of input needs to be indexed.

* @par Outputs:
* @li y: The indexed output tensor. Has the same type and format as input "x".
*/
REG_OP(IndexByTensor)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(indices_mask, ListInt, {})
    .OP_END_FACTORY_REG(IndexByTensor)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SELECTION_H_
