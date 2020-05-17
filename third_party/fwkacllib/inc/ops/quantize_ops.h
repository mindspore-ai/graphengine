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

#ifndef GE_OP_QUANTIZE_OPS_H
#define GE_OP_QUANTIZE_OPS_H
#include "../graph/operator_reg.h"

namespace ge {
REG_OP(QuantizedInnerProduct)
    .INPUT(x, TensorType({DT_UINT8}))
    .INPUT(w, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(b, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scale_q, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_q, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(scale_deq_req, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_req, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(quant_algo, ListInt)
    .REQUIRED_ATTR(scale_sqrt, ListInt)
    .REQUIRED_ATTR(num_output, Int)
    .ATTR(transpose, Bool, false)
    .ATTR(bias_term, Bool, false)
    .ATTR(axis, Int, 1)
    .OP_END_FACTORY_REG(QuantizedInnerProduct)

/**
* @brief Dequantizes the input tensor into a float tensor.\n
* [input_min_range, input_max_range] are scalar floats that specify the range
* for "output_data". \n
* The "mode" attribute controls exactly which calculations are used to convert\n
* the float values to their quantized equivalents.
* @par Inputs:
* @li input_data: A Tensor. Must be one of the following types: int8, uint8,
* int32.
* @li input_min_range: A Tensor of type float32.
* Specifies the minimum scalar value possibly produced for the input.
* @li input_max_range: A Tensor of type float32.
* Specifies the maximum scalar value possibly produced for the input.

* @par Attributes:
* mode: An optional string from: "MIN_COMBINED", "MIN_FIRST", and "SCALED".
* Defaults to "MIN_COMBINED".

* @par Outputs:
* output_data: A dictionary of type float32.

* @attention Constraints:
* @li "input_min_range" and "input_max_range" have the same shapes.
* @li "input_data" and "output_data" have the same shapes.
*/
REG_OP(Dequantize)
    .INPUT(x, TensorType(DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16))
    .INPUT(min_range, TensorType{DT_FLOAT})
    .INPUT(max_range, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(mode, String, "MIN_COMBINED")
    .OP_END_FACTORY_REG(Dequantize)

/**
*@brief Quantizes the input.

*@par Inputs:
*x: An NC1HWC0 tensor of type float16 or float32, specifying the input.

*@par Attributes:
*@li scale: A required float32, specifying the scaling ratio.
*@li offset: A required float16, specifying the offset.
*@li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False". Defaults to "False".
*@li round_mode: An optional string, specifying the float16 to int8 cast type.
* The value range is [Round, Floor, Ceiling, Truncate]. Defaults to "Round".

*@par Outputs:
*y: The quantized output tensor of type int8 and with format NC1HWC0.
*/
REG_OP(AscendQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(round_mode, String, "Round")
    .OP_END_FACTORY_REG(AscendQuant)

/**
*@brief Dequantizes the input.

*@par Inputs:
*@li x: An NC1HWC0 tensor of type int32, specifying the input.
*@li deq_scale: An NC1HWC0 tensor of type float16 or uint64, specifying the scaling ratio.

*@par Attributes:
*@li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False". Defaults to "False".
*@li relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False".
*@li dtype: A optional int32, specifying the output data type. Defaults to "DT_FLOAT".

*@par Outputs:
*y: The dequantized output tensor of type float16 or float32 and with format NC1HWC0.
*/
REG_OP(AscendDequant)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_FLOAT16, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(relu_flag, Bool, false)
    .ATTR(dtype, Int, DT_FLOAT)
    .OP_END_FACTORY_REG(AscendDequant)

} // namespace ge

#endif // GE_OP_QUANTIZE_OPS_H
