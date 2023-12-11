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
 * \file transformation.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_H_
#define OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief HDRNet and ISP direct data conversion
returned tensor's dimension will correspond to input dimension [0, 3, 4, 2, 1],
convert tensor dtype float16 to int16 . \n

*@par Inputs:
*one inputs, including:
*@li x: A Tensor. Must be one of the following types: float16.

*@par Outputs:
*y: A Tensor. Must be one of the following types: int16. \n

*@par Third-party framework compatibility
*only for use by corresponding operators in HDRnet networks
*/
REG_OP(TransArgb)
    .INPUT(x, "T1")
    .OUTPUT(y, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT16}))
    .DATATYPE(T2, TensorType({DT_INT16}))
    .OP_END_FACTORY_REG(TransArgb)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_H_
