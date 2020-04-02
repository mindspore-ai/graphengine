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

#ifndef GE_OP_SPLIT_COMBINATION_OPS_H
#define GE_OP_SPLIT_COMBINATION_OPS_H
#include "../graph/operator_reg.h"

namespace ge {
REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(value, TensorType::BasicType())
    .DYNAMIC_OUTPUT(output, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

REG_OP(SplitD)
    .INPUT(value, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                    DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(output, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                             DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitD)

REG_OP(SplitV)
    .INPUT(input_value, TensorType::BasicType())
    .INPUT(input_size_splits, TensorType::IndexNumberType())
    .INPUT(input_split_dim, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(output_data, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)

REG_OP(SplitVD)
    .INPUT(input_value, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                    DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(output_data, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                             DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(size_splits, ListInt)
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitVD)

/**
*@brief Concatenates a list of N tensors along the first dimension.
*@par Inputs:
* Two inputs, including:
* @li values: A list of Tensors. Must be one of the following types: int8, int16, int32, \n
*     int64, uint8, uint16, uint32, uint64, float16, float32. \n
*     Tensors to be concatenated. \n
*     All must have size 1 in the first dimension and same shape.
* @li shape: A Tensor of the same type as "x". \n
* The final shape of the result. Should be equal to the shapes of any input
* but with the number of input values in the first dimension.

*@par Attributes:
* shape: A required list of ints.

*@par Outputs:
*output_data: The concatenated tensor with same type as "values".
*/
REG_OP(ParallelConcat)
    .DYNAMIC_INPUT(values, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .OUTPUT(output_data, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .REQUIRED_ATTR(shape, ListInt)    
    .OP_END_FACTORY_REG(ParallelConcat)
  
REG_OP(ConcatExt2)
    .DYNAMIC_INPUT(input_values, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_INT64, DT_UINT64, DT_UINT32, DT_INT16, DT_UINT16, DT_UINT8}))
    .OUTPUT(output_data, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_INT64, DT_UINT64, DT_UINT32, DT_INT16, DT_UINT16, DT_UINT8}))
    .REQUIRED_ATTR(axis, Int)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatExt2)

REG_OP(ConcatV2)
    .DYNAMIC_INPUT(input_values, TensorType::BasicType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(output_data, TensorType::BasicType())
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatV2)

REG_OP(ConcatD)
    .DYNAMIC_INPUT(input_values, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .OUTPUT(output_data, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .REQUIRED_ATTR(concat_dim, Int)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatD)

REG_OP(Concat)
    .DYNAMIC_INPUT(input_values, TensorType::BasicType())
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(output_data, TensorType::BasicType())
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Concat)

/**
*@brief Packs the list of tensors in values into a tensor with rank one higher than each tensor in
* values, by packing them along the axis dimension. Given a list of length N of tensors of  
* shape (A, B, C); if axis == 0 then the output tensor will have the shape (N, A, B, C).

*@par Inputs:
* x: A list of N Tensors. Must be one of the following types: int8, int16, int32, 
*     int64, uint8, uint16, uint32, uint64, float16, float32, bool.

*@par Attributes:
*@li axis: A required int.
*     Dimension along which to pack. The range is [-(R+1), R+1).
*@li N: A required int. Number of tensors.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(Pack)
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(axis, Int)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Pack)

/**
*@brief Computes offsets of concat inputs within its output.

*@par Inputs:
*Two inputs, including:
* @li concat_dim: A Tensor of type int32.
* @li x: A list of 1D Tensor objects of type int32.

*@par Attributes:
*@li Concat_dim: A required int. Must be within the rank of input "x".
*@li N: A required int. 

*@par Outputs:
*y: A Tensor list with same type as "x".
*/
REG_OP(ConcatOffset)
    .INPUT(concat_dim, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatOffset)

/**
*@brief Computes offsets of concat inputs within its output.

*@par Inputs:
*Two inputs, including:
* @li concat_dim: A Tensor of type int32.
* @li x: A list of 1D Tensor objects of type int32.

*@par Attributes:
*@li Concat_dim: A required int. Must be within the rank of input "x".
*@li N: A required int. 

*@par Outputs:
*y: A Tensor list with same type as "x".
*/
REG_OP(ConcatOffsetD)
    .DYNAMIC_INPUT(x, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(concat_dim, Int)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatOffsetD)
}  // namespace ge

#endif  // GE_OP_SPLIT_COMBINATION_OPS_H
