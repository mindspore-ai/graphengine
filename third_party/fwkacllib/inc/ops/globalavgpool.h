/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file globalavgpool.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_GLOBALAVERAGEPOOL_H_
#define OPS_BUILT_IN_OP_PROTO_INC_GLOBALAVERAGEPOOL_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief GlobalAveragePool consumes an input tensor X and applies average pooling across the values in the same channel.
This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor \n

*@par Inputs:
*@li x: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W),
where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.

*@par Outputs:
*y: Output data tensor from pooling across the input tensor. The output tensor has the same rank as the input.
The first two dimensions of output shape are the same as the input (N x C), while the other dimensions are all 1

*@par Restrictions:
*Warning: This operator can be integrated only by configuring INSERT_OP_FILE of aclgrphBuildModel. Please do not use it directly.
*/
REG_OP(GlobalAveragePool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(GlobalAveragePool)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_GLOBALAVGPOOL_H_