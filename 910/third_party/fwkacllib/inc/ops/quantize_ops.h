/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file quantize_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Dequantizes the input tensor into a float tensor.
* [min_range, max_range] are float32 tensors that specify the range
* for "y".
* The "mode" attribute controls exactly which calculations are used to convert
* the float values to their quantized equivalents.
* @par Inputs:
* @li x: A Tensor. Must be one of the following types: qint8, quint8, qint32, quint16, qint16.
* @li min_range: A Tensor of type float32.
* Specifies the minimum scalar value possibly produced for the input.
* @li max_range: A Tensor of type float32.
* Specifies the maximum scalar value possibly produced for the input . \n

* @par Attributes:
* mode: An optional string from: "MIN_COMBINED", "MIN_FIRST", and "SCALED".
* Defaults to "MIN_COMBINED" . \n

* @par Outputs:
* y: A dictionary of type float32 . \n

* @attention Constraints:
* @li "min_range" and "max_range" have the same shapes.
* @li "x" and "y" have the same shapes . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Dequantize.
*/
REG_OP(Dequantize)
    .INPUT(x, TensorType(DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16))
    .INPUT(min_range, TensorType{DT_FLOAT})
    .INPUT(max_range, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(mode, String, "MIN_COMBINED")
    .OP_END_FACTORY_REG(Dequantize)

/**
* @brief Quantizes the input . \n
* @par Inputs:
* @li x: shape and dtype of input_x. \n
* @li scales: shape and dtype of input_scales. \n
* @li zero_points: shape and dtype of input_zero_points \n
* @par Attributes:
* @li dtype: required, type.
* @li axis: the processed dim. \n
* @par Outputs:
* y: shape and dtype of output_y, should be same shape as input, dtype is same as the quantified type . \n
*/
REG_OP(Quantize)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points, TensorType({DT_INT8, DT_UINT8, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .REQUIRED_ATTR(dtype, String)
    .ATTR(axis, Int, 1)
    .OP_END_FACTORY_REG(Quantize)

/**
* @brief Quantizes the input.

* @par Inputs:
* x: An tensor of type float16 or float32, specifying the input. The format must 
* be NC1HWC0, FRACTAL_NZ, NDC1HWC0 or ND. \n

* @par Attributes:
* @li scale: A required float32, specifying the scaling ratio.
* @li offset: A required float32, specifying the offset.
* @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
* Defaults to "False".
* @li round_mode: An optional string, specifying the float16 to int8 cast type.
* The value range is [Round, Floor, Ceil, Trunc]. Defaults to "Round" .
* @li dst_type: A optional int32, specifying the output data type. Defaults to 
* "2", represents detype "DT_INT8". \n

* @par Outputs:
* y: The quantized output tensor of type int8 or int4. The format must be 
* NC1HWC0, FRACTAL_NZ, NDC1HWC0 or ND. \n

* @attention Constraints:
* round_mode value range is [Round, Floor, Ceil, Trunc].
* @li Round: round to nearest, tie to even(c language rint).
* @li Floor: round to minus infinity(c language floor).
* @li Ceil: round to positive infinity(c language ceil).
* @li Trunc: round to zero(c language trunc). \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(round_mode, String, "Round")
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AscendQuant)

/**
* @brief Quantizes the input.
* @par Inputs:
* @li x:A required Tensor. Must be one of the following types: float16,
* float32, bfloat16.
* @li scale:A required Tensor. Must be one of the following types: float16,
* float32, bfloat16. If scale is 1D tensor, shape must be same as the last
* dimension of x. Otherwise the number of dimensions should be equal to x,
* the last dimension of shape should be same as x, others must be 1.
* @li offset:A optional Tensor. Must be one of the following types: float16,
* float32, bfloat16. Shape is same as scale. \n
* @par Attributes:
* @li sqrt_mode: A optional bool, specifying whether to perform square root
* on "scale", either "True" or "False". Defaults to "False".
* @li round_mode: An optional string, specifying the cast type.
* The value range is [round, floor, ceil, trunc]. Defaults to "round".
* @li dst_type: A optional int32, specifying the output data type.
* Defaults to "DT_INT8".
* @li axis: A optional int32, specifying axis to scale and offset.
* Defaults to "-1" . \n
* @par Outputs:
* y: The quantized output tensor of type int8 or int4, shape is same as x. \n
*/
REG_OP(AscendQuantV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(round_mode, String, "round")
    .ATTR(dst_type, Int, DT_INT8)
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(AscendQuantV2)

/**
* @brief Dequantizes the input.

 *@par Inputs:
* @li x: An tensor of type int32, specifying the input. The format must be 
* FRACTAL_NZ, NC1HWC0 or NDC1HWC0.
* @li deq_scale: An tensor of type float16 or uint64, specifying the scaling ratio.
* The format must be NC1HWC0, NC1HWC0 or NDC1HWC0. \n

* @par Attributes:
* @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
* Defaults to "False".
* @li relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False".
* @li dtype: A optional int32, specifying the output data type. Defaults to "0"
* , represents dtype "DT_FLOAT". \n

* @par Outputs:
* y: The dequantized output tensor of type float16 or float32. The format must be 
* FRACTAL_NZ, NC1HWC0 or NDC1HWC0. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendDequant)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_FLOAT16, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(relu_flag, Bool, false)
    .ATTR(dtype, Int, DT_FLOAT)
    .OP_END_FACTORY_REG(AscendDequant)

/**
* @brief Anti quantizes the input . \n

* @par Inputs:
* x: An tensor of type int8, specifying the input . \n

* @par Attributes:
* @li scale: A required float32 scale.
* @li offset: A required float32 offset.
* @li dtype: A optional int32, specifying the output data type. Defaults to "DT_FLOAT".
* @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
* Defaults to "False" . \n

* @par Outputs:
* y: The dequantized output tensor of type float16 or float32. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendAntiQuant)
    .INPUT(x, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(dtype, Int, DT_FLOAT)
    .ATTR(sqrt_mode, Bool, false)
    .OP_END_FACTORY_REG(AscendAntiQuant)

/**
* @brief Anti quantizes the input . \n

* @par Inputs:
* @li x: A multi-dimensional tensor of type int8/int4, specifying the input.
  The maximum dimension should not exceed 8 dimensions.
* @li scale: A 1-D tensor of type float32/bfloat16, specifying the scale.
  Shape is (n,), where n can be 1. If n is not 1, it must be the same as
  the size of last dimension of x.
* @li offset: A 1-D tensor of type float32/bfloat16, specifying the offset.
  The shape and dtype of offset should be same to scale.

* @par Attributes:
* @li dst_type: A optional int32, specifying the output data type. Defaults to "DT_FLOAT16".
* @li sqrt_mode: A optional bool, specifying whether to perform square root on "scale", either "True" or "False".
* Defaults to "False" . \n

* @par Outputs:
* y: The dequantized output tensor of type float16 or bfloat16. \n

*/
REG_OP(AscendAntiQuantV2)
    .INPUT(x, TensorType({DT_INT8, DT_INT4}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BFLOAT16}))
    .ATTR(dst_type, Int, DT_FLOAT16)
    .ATTR(sqrt_mode, Bool, false)
    .OP_END_FACTORY_REG(AscendAntiQuantV2)

/**
* @brief Dequantizes the input of int16 . \n

* @par Inputs:
* @li x0: An tensor of type int32, specifying the input.
* @li deq_scale: An tensor of type uint64, specifying the scaling ratio.
* @li x1: An tensor of type int16, specifying the input . \n

* @par Attributes:
* relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False" . \n

* @par Outputs:
* y: The dequantized output tensor of type int16. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendDequantS16)
  .INPUT(x0, TensorType({DT_INT32}))
  .INPUT(deq_scale, TensorType({DT_UINT64}))
  .OPTIONAL_INPUT(x1, TensorType({DT_INT16}))
  .OUTPUT(y, TensorType({DT_INT16}))
  .ATTR(relu_flag, Bool, false)
  .OP_END_FACTORY_REG(AscendDequantS16)

/**
* @brief Requantizes the input.

* @par Inputs:
* @li x: An tensor of type int32, specifying the input. The format must be 
* FRACTAL_NZ or NC1HWC0.
* @li req_scale: An tensor of type uint64, specifying the scaling ratio. The 
* format must be NC1HWC0,NDC1HWC0. \n

* @par Attributes:
* relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False" . \n

* @par Outputs:
* y: The dequantized output tensor of type int8. The format must be FRACTAL_NZ, 
* NC1HWC0 or NDC1HWC0. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendRequant)
  .INPUT(x, TensorType({DT_INT32}))
  .INPUT(req_scale, TensorType({DT_UINT64}))
  .OUTPUT(y, TensorType({DT_INT8}))
  .ATTR(relu_flag, Bool, false)
  .OP_END_FACTORY_REG(AscendRequant)

/**
* @brief Requantizes the input of int16 . \n

* @par Inputs:
* @li x0: An tensor of type int16, specifying the input.
* @li req_scale: An tensor of type uint64, specifying the scaling ratio.
* @li x1: An tensor of type int16 . \n

* @par Attributes:
* @li dual_output: A optional bool, specifying whether to perform dual ouput, either "True" or "False".
* Defaults to "False".
* @li relu_flag: A optional bool, specifying whether to perform ReLU, either "True" or "False". Defaults to "False" . \n

* @par Outputs:
* @li y0: The dequantized output tensor of type int8.
* @li y1: The dequantized output tensor of type int16. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendRequantS16)
  .INPUT(x0, TensorType({DT_INT16}))
  .INPUT(req_scale, TensorType({DT_UINT64}))
  .OPTIONAL_INPUT(x1, TensorType({DT_INT16}))
  .OUTPUT(y0, TensorType({DT_INT8}))
  .OUTPUT(y1, TensorType({DT_INT16}))
  .ATTR(dual_output, Bool, false)
  .ATTR(relu_flag, Bool, false)
  .OP_END_FACTORY_REG(AscendRequantS16)

/**
* @brief Quantizes the input of int8.

* @par Inputs:
* @li x: A tensor. Must be one of the following types: int8. The format support NZ.
* @li offset: A tensor. Must be one of the following types: int8. The format support NZ. \n

* @par Attributes:
* @li dst_type: Declare the output dtype. Support DT_INT8, DT_INT4. Defaults to DT_INT8. \n

* @par Outputs:
* @li y: A output Tensor. Must be one of the following types: int8, int4. The format support NZ. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, Onnx, Tensorflow or Pythorch.
*/
REG_OP(AscendWeightQuant)
  .INPUT(x, TensorType({DT_INT8}))
  .INPUT(offset, TensorType({DT_INT8}))
  .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
  .ATTR(dst_type, Int, DT_INT8)
  .OP_END_FACTORY_REG(AscendWeightQuant)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_OPS_H_
