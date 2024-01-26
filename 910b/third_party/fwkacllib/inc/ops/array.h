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
 * \file array.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_ARRAY_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ARRAY_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Creates a Const PlaceHolder tensor.
This operator is used to handle inputs where the address does not change but the value is uncertain.
Converting the input to this type will help improve the execution performance of the graph
(avoiding the input address from being repeatedly refreshed). \n

*@par Attributes:
 * origin_shape: Required. The origin shape of a tensor. \n
 * origin_format: Required. The origin format of a tensor. \n
 * storage_shape: Required. The storage shape of a tensor. \n
 * storage_format: Required. The storage format of a tensor.
 * expand_dim_rules: Required. The expand dim rules while trans tensor with \n
 *                             origin_shape、origin_format and storage_format into storage_shape \n
 * dtype: Required. The dtype of a tensor. \n
 * addr: Required. The address of a tensor \n
 * size: Required. The size of address. \n
 * placement: The placement of a tensor, 0 represents host, 1 represents device, and the default value is 1.. \n

*@par Outputs:
*y: The ConstPlaceHolder tensor. \n
*/

REG_OP(ConstPlaceHolder)
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(origin_shape, ListInt)
    .REQUIRED_ATTR(origin_format, Int)
    .REQUIRED_ATTR(storage_shape, ListInt)
    .REQUIRED_ATTR(storage_format, Int)
    .REQUIRED_ATTR(expand_dim_rules, String)
    .REQUIRED_ATTR(dtype, Type)
    .REQUIRED_ATTR(addr, Int)
    .REQUIRED_ATTR(size, Int)
    .ATTR(placement, Int, 1)
    .OP_END_FACTORY_REG(ConstPlaceHolder)

} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_ARRAY_H_
