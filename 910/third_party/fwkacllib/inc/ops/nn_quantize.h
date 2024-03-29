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
 * \file nn_quantize.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief trans quant param form float32 to uint64 . \n

* @par Inputs:
* @li scale: A tensor of type float32.
* @li offset: A tensor of type float32.


* @par Outputs:
* @li y: output tensor of type uint64.

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, Onnx, Tensorflow or Pythorch.
*/
REG_OP(TransQuantParamV2)
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT64}))
    .OP_END_FACTORY_REG(TransQuantParamV2)

/**
*@brief Fake-quantize the data of 'x' tensor with scale, zero_point, quant_min and quant_max. \n

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32.
*@li scale: A Tensor of type float32 or float16. Has the same type and format as "x".
*@li zero_point: A Tensor of type int32, float16 or float32.\n

*@par Attributes:
*@li axis: An required attribute of type int64.
*@li quant_min: An required attribute of type int64.
*@li quant_max: An required attribute of type int64.\n

*@par Outputs:
*y: A Tensor of type float32 or float16. 
*mask: A Tensor of type bool. \n

*@par Third-party framework compatibility
* Compatible with Pytorch operator FakeQuantAffineCachemask.
*/

REG_OP(FakeQuantAffineCachemask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(zero_point, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axis, Int)
    .REQUIRED_ATTR(quant_min, Int)
    .REQUIRED_ATTR(quant_max, Int)
    .OP_END_FACTORY_REG(FakeQuantAffineCachemask)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
