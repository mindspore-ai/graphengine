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

#ifndef GE_OP_NN_OPS_H_
#define GE_OP_NN_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Computes gradient of the FractionalMaxPool function.

*@par Inputs:
*Inputs include: \n
* @li orig_input: A Tensor. Must be one of the following types: float32, float64, int32, int64.
* @li orig_output: A Tensor. Must have the same type as orig_input.
* @li out_backprop: A Tensor. Must have the same type as orig_input. \n
      4-D with shape [batch, height, width, channels].
* @li row_pooling_sequence: A Tensor of type int64.
* @li col_pooling_sequence: A Tensor of type int64.

*@par Attributes:
*overlapping: An optional bool. Defaults to False.

*@par Outputs:
*y: A Tensor. Has the same type as orig_input.

*@attention Constraints:\n
*-The implementation for FractionalMaxPoolGrad on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(FractionalMaxPoolGrad)
    .INPUT(orig_input, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(orig_output, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(row_pooling_sequence, TensorType({ DT_INT64 }))
    .INPUT(col_pooling_sequence, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64 }))
    .ATTR(overlapping, Bool, false)
    .OP_END_FACTORY_REG(FractionalMaxPoolGrad)

/**
*@brief Performs fractional average pooling on the input.

*@par Inputs:
*Inputs include: \n
*x: A Tensor. Must be one of the following types: float32, float64, int32, int64. \n
 4-D with shape [batch, height, width, channels].

*@par Attributes:
*@li pooling_ratio: A list of floats that has length >= 4.
*@li pseudo_random: An optional bool. Defaults to False.
*@li overlapping: An optional bool. Defaults to False. When set to True, it means when pooling.
*@li deterministic: An optional bool. Defaults to False.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*@li y: A Tensor. Has the same type as x.
*@li row_pooling_sequence: A Tensor of type int64.
*@li col_pooling_sequence: A Tensor of type int64.

*@attention Constraints:\n
*-The implementation for FractionalAvgPool on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(FractionalAvgPool)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .ATTR(pooling_ratio, ListFloat, {})
    .ATTR(pseudo_random, Bool, false)
    .ATTR(overlapping, Bool, false)
    .ATTR(deterministic, Bool, false)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(FractionalAvgPool)

/**
*@brief Performs fractional max pooling on the input.

*@par Inputs:
*Inputs include: \n
*x: A Tensor. Must be one of the following types: float32, float64, int32, int64. \n
 4-D with shape [batch, height, width, channels].

*@par Attributes:
*@li pooling_ratio: A list of floats that has length >= 4. Pooling ratio for each dimension of value.
*@li pseudo_random: An optional bool. Defaults to False.
*@li overlapping: An optional bool. Defaults to False.
*@li deterministic: An optional bool. Defaults to False.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0.

*@par Outputs:
*@li y: A Tensor. Has the same type as x.
*@li row_pooling_sequence: A Tensor of type int64.
*@li col_pooling_sequence: A Tensor of type int64.

*@attention Constraints:\n
*-The implementation for FractionalMaxPool on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(FractionalMaxPool)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .ATTR(pooling_ratio, ListFloat, {})
    .ATTR(pseudo_random, Bool, false)
    .ATTR(overlapping, Bool, false)
    .ATTR(deterministic, Bool, false)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(FractionalMaxPool)

/**
*@brief Finds values of the n-th order statistic for the last dimension.

*@par Inputs:
*Inputs include: \n
* @li x: A Tensor. Must be one of the following types: float32, float64, int32, uint8, \n
      int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
* @li n: A Tensor of type int32. 0-D.

*@par Attributes:
*reverse: An optional bool. Defaults to False.

*@par Outputs:
*y: A Tensor. Has the same type as x.

*@attention Constraints:\n
*-The implementation for NthElement on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(NthElement)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(n, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(NthElement)

/**
*@brief Computes gradient of the FractionalAvgPool function.

*@par Inputs:
*Inputs include: \n
* @li orig_input_tensor_shape: A Tensor of type int64.
* @li out_backprop: A Tensor. Must be one of the following types: float32, float64, \n
      int32, int64. 4-D with shape [batch, height, width, channels].
* @li row_pooling_sequence: A Tensor of type int64.
* @li col_pooling_sequence: A Tensor of type int64.

*@par Attributes:
*overlapping: An optional bool. Defaults to False.

*@par Outputs:
*y: A Tensor. Has the same type as out_backprop.

*@attention Constraints:\n
*-The implementation for FractionalAvgPoolGrad on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(FractionalAvgPoolGrad)
    .INPUT(orig_input_tensor_shape, TensorType({DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .INPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .ATTR(overlapping, Bool, false)
    .OP_END_FACTORY_REG(FractionalAvgPoolGrad)

/**
*@brief Returns the permuted vector/tensor in the destination data format given the.

*@par Inputs:
*Inputs include: \n
*x: A Tensor. Must be one of the following types: int32, int64. Vector of size 4 \n
 or Tensor of shape (4, 2) in source data format.

*@par Attributes:
*@li src_format: An optional string. Defaults to "NHWC". source data format.
*@li dst_format: An optional string. Defaults to "NCHW". destination data format.

*@par Outputs:
*y: A Tensor. Has the same type as x.

*@attention Constraints:\n
*-The implementation for DataFormatVecPermute on Ascend uses AICPU, with bad performance.\n

*/
REG_OP(DataFormatVecPermute)
    .INPUT(x, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT32, DT_INT64 }))
    .ATTR(src_format, String, "NHWC")
    .ATTR(dst_format, String, "NCHW")
    .OP_END_FACTORY_REG(DataFormatVecPermute)

}  // namespace ge

#endif  // GE_OP_NN_OPS_H_
