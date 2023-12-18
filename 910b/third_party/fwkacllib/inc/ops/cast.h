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
 * \file cast.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CAST_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CAST_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief data conversion operator
Convert uint16 to uint32, convert to int32, convert to float32, 
multiply by the reciprocal of pixel, convert to float16 . \n

*@par Inputs:
*one inputs, including:
*@li x: A Tensor. Must be one of the following types: uint16.

*@par Outputs:
*y: A Tensor. Must be one of the following types: float16. \n

*@par Third-party framework compatibility
*only for use by corresponding operators in HDRnet networks
*/
REG_OP(AdaCast)
    .INPUT(x, "T1")
    .OUTPUT(y, "T2")
    .ATTR(pixel, Int, 65535)
    .DATATYPE(T1, TensorType({DT_UINT16}))
    .DATATYPE(T2, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(AdaCast)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_CAST_H_
