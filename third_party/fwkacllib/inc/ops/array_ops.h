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

#ifndef GE_OP_ARRAY_OPS_H_
#define GE_OP_ARRAY_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Applies lower_bound(sorted_search_values, values) along each row.

*@par Inputs:
*The input sorted_x and values can be one-dimensional vector. Inputs include: \n
* @li sorted_x:A `Tensor`. 2-D Tensor where each row is ordered.
* @li values:A `Tensor`. Must have the same type as `sorted_x`.

*@par Attributes:
*@li out_type:An optional `DType` from: `int32, int64`. Defaults to `int32`.

*@par Outputs:
*y: A `Tensor` of type `out_type`.

*@attention Constraints: \n
*-The implementation for LowerBound on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(LowerBound)
    .INPUT(sorted_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_type, Type, DT_INT32)
    .OP_END_FACTORY_REG(LowerBound)

/**
*@brief Reverses variable length slices.

*@par Inputs:
*The input x can be k-dimensional tensor, num_lower and num_upper can be zero-dimensional scalar. Inputs include: \n
* @li x:A Tensor. The input to reverse.
* @li seq_lengths:A Tensor. Must be one of the following types: int32, int64. 1-D.

*@par Attributes:
*@li seq_dim:An optional int. Defaults to 0. The dimension along which reversal is performed.
*@li batch_dim:An optional int. Defaults to 0. The dimension along which reversal is performed.

*@par Outputs:
*y: Rank k tensor of the same shape as input. The extracted banded tensor.

*@attention Constraints: \n
*-The implementation for ReverseSequence on Ascend uses AI CPU, with bad performance.

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(ReverseSequence)
    .INPUT(x,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(seq_lengths, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(seq_dim, Int)
    .ATTR(batch_dim, Int, 0)
    .OP_END_FACTORY_REG(ReverseSequence)

/**
*@brief Copy a tensor setting everything outside a central band in each innermost matrix.

*@par Inputs:
*The input x can be k-dimensional tensor, num_lower and num_upper can be zero-dimensional scalar. Inputs include: \n
* @li x:Rank `k` tensor.
* @li num_lower:0-D tensor. Number of superdiagonals to keep. If negative, keep entire upper triangle.
* @li num_upper:0-D tensor. Number of superdiagonals to keep. If negative, keep entire upper triangle.

*@par Outputs:
*y: Rank k tensor of the same shape as input. The extracted banded tensor.

*@attention Constraints: \n
*-The implementation for MatrixBandPart on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(MatrixBandPart)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, \
           DT_INT16, DT_UINT16, DT_INT32, DT_INT64,
           DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL }))
    .INPUT(num_lower, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(num_upper, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL }))
    .OP_END_FACTORY_REG(MatrixBandPart)

/**
*@brief Finds unique elements in a 1-D tensor.

*@par Inputs:
*The input x can be k-dimensional tensor, num_lower and num_upper can be zero-dimensional scalar. Inputs include: \n
*x:1-D tensor.

*@par Attributes:
*out_idx:An optional DType from: int32, int64. Defaults to int32. \n

*@par Outputs:
*@li y:A Tensor. Has the same type as x.
*@li idx:A Tensor of type out_idx.
*@li count:A Tensor of type out_idx.

*@attention Constraints: \n
*-The implementation for UniqueWithCounts on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(UniqueWithCounts)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(idx, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(count, TensorType({ DT_INT32, DT_INT64 }))
    .REQUIRED_ATTR(out_idx, Type)
    .OP_END_FACTORY_REG(UniqueWithCounts)

/**
*@brief Finds unique elements in a 1-D tensor.

*@par Inputs:
*The input x can be k-dimensional tensor, num_lower and num_upper can be zero-dimensional scalar. Inputs include: \n
*x:1-D tensor.

*@par Attributes:
*out_idx:An optional DType from: int32, int64. Defaults to int32.

*@par Outputs:
*@li y:x in the unique output y.
*@li idx:A tensor idx the same size as x that contains the index of each value of x.

*@attention Constraints: \n
*-The implementation for Unique on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(Unique)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(Unique)

/**
*@brief Finds unique elements in a 1-D tensor.

*@par Inputs:
*The input x can be k-dimensional tensor, num_lower and num_upper can be zero-dimensional scalar. Inputs include: \n
* @li x:1-D tensor.
* @li axis:A `Tensor` of type `int32` (default: None). The axis of the Tensor to.

*@par Attributes:
*out_idx:An optional DType from: int32, int64. Defaults to int32.

*@par Outputs:
*@li y:x in the unique output y.
*@li idx:A tensor idx the same size as x that contains the index of each value of x.

*@attention Constraints: \n
*-The implementation for UniqueExt2 on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(UniqueExt2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(UniqueExt2)

/**
*@brief Computes the inverse permutation of a tensor.

*@par Inputs:
*The input x can be k-dimensional tensor. Inputs include: \n
*x:K-D tensor.

*@par Outputs:
*y:1-D tensor.

*@attention Constraints:\n
*-The implementation for InvertPermutation on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(InvertPermutation)
    .INPUT(x, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(InvertPermutation)

/**
*@brief Checks a tensor for NaN and Inf values.

*@par Inputs:
*The input x can be k-dimensional tensor. Inputs include: \n
*x:The input tensor.

*@par Attributes:
*message:Prefix of the error message.

*@par Outputs:
*y:The output tensor.

*@attention Constraints: \n
*-The implementation for CheckNumerics on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(CheckNumerics)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(message, String)
    .OP_END_FACTORY_REG(CheckNumerics)

/**
*@brief Converts an array of flat indices into a tuple of coordinate arrays.

*@par Inputs:
*The input indices can be 0-D or 1-D tensor, dims can be 1-D. Inputs include: \n
* @li indices: A 0-D or 1-D int Tensor whose elements are indices into the flattened version of an array of dimensions dims.
* @li dims:A Tensor. Must have the same type as indices. An 1-D int Tensor. The shape of the array to use for unraveling indices.

*@par Outputs:
*y:A Tensor. Has the same type as indices.

*@attention Constraints: \n
*-The implementation for UnravelIndex on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(UnravelIndex)
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(dims, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(UnravelIndex)

/**
*@brief Applies upper_bound(sorted_search_values, values) along each row.

*@par Inputs:
*The input sorted_x can be 2-D tensor, values can be 2-D. Inputs include:
* @li sorted_x: 2-D Tensor where each row is ordered.
* @li values:2-D Tensor with the same numbers of rows as `sorted_x.

*@par Attributes:
*out_type:sets the optional out_type attribute to value.

*@par Outputs:
*y:A `Tensor` with the same shape as `values`.

*@attention Constraints: \n
*-The implementation for UpperBound on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(UpperBound)
    .INPUT(sorted_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
      DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
      DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(out_type, Type)
    .OP_END_FACTORY_REG(UpperBound)

/**
*@brief Finds unique elements in a 1-D tensor.

*@par Inputs:
*The input x can be 1-D vector, axis can be 1-D vector. Inputs include: \n
* @li x:1-D tensor.
* @li axis:1-D tensor.

*@par Attributes:
*out_idx:An optional DType from: int32, int64. Defaults to int32.

*@par Outputs:
*@li y:x in the unique output y.
*@li idx:A tensor idx the same size as x that contains the index of each value of x.
*@li count:A tensor idx the same size as x that contains the index of each value of x.

*@attention Constraints: \n
*-The implementation for UniqueWithCountsExt2 on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(UniqueWithCountsExt2)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .INPUT(axis, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .OUTPUT(idx, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(count, TensorType({ DT_INT32, DT_INT64 }))
    .REQUIRED_ATTR(out_idx, Type)
    .OP_END_FACTORY_REG(UniqueWithCountsExt2)

/**
*@brief Fill the tensor with the mirror value.

*@par Inputs:
*The input x and paddings can be one-dimensional scalar. Inputs include: \n
* @li x: input tensor to be padded.
* @li paddings: A two-column matrix specifying the padding sizes. The number of rows must be the same as the rank of `input`.

*@par Attributes:
*mode:Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions do not include the borders, while in symmetric mode the padded regions do include the borders.

*@par Outputs:
*y: The padded tensor.

*@attention Constraints: \n
-The implementation for MirrorPad on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(MirrorPad)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL }))
    .INPUT(paddings, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL }))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPad)

/**
*@brief Calculate the difference between two numbers or a list of strings.

*@par Inputs:
*The input x and y can be one-dimensional vector. Inputs include: \n
* @li x:A Tensor. 1-D. Values to keep.
* @li y:A Tensor. Must have the same type as x. 1-D. Values to remove.

*@par Attributes:
*out_idx:An optional DType from: int32, int64. Defaults to int32.

*@par Outputs:
*@li out:A Tensor. Has the same type as x.
*@li idx:A Tensor of type out_idx.

*@attention Constraints:\n
-The implementation for ListDiff on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(ListDiff)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(ListDiff)

/**
*@brief Creates a constant tensor from a tensor-like object. This operator is used for inference. \n
Operator Const has the same definition as operator Constant.

*@par Attributes:
*@li value: Required. The value and type of the resulting tensor.
*@li dtype: Optional. The type of the elements of the resulting tensor. \n
The data type specified by this parameter must be the same as that of the "value" attribute.

*@par Outputs:
*y: A constant tensor.
*/
REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())  // This is the value of the const op
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(Const)

/**
*@brief Creates a constant tensor for training.

*@par Attributes:
*@li value: Required. The value and type of the resulting tensor.
*@li dtype: Optional. The type of the elements of the resulting tensor. \n
The data type specified by this parameter must be the same as that of the "value" attribute.

*@par Outputs:
*y: The constant tensor.
*/
REG_OP(Constant)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())  // This is the value of the constant op
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(Constant)

/**
*@brief Returns a copy of the input tensor.

*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: A tensor.
*/
REG_OP(Snapshot)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Snapshot)

/**
*@brief Gives a guarantee to the runtime that the input tensor is a constant.

*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: The input tensor.
*/
REG_OP(GuaranteeConst)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(GuaranteeConst)

/**
*@brief Returns the target shape for broadcasting shapes "x1" and "x2".

*@par Inputs:
*@li x1: A tensor of type int32 or int64. A shape.
*@li x2: A tensor of the same type as "x1". The other shape.

*@par Outputs:
*y: A tensor. The broadcasted shape.
*/
REG_OP(BroadcastArgs)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(BroadcastArgs)

/**
*@brief Outputs its input tensor as is and triggers an error if a gradient is requested.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*message: Will be printed in the error at the attempt to request a gradient.

*@par Outputs:
*y: The input tensor.
*/
REG_OP(PreventGradient)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(message, String, "")
    .OP_END_FACTORY_REG(PreventGradient)

/**
*@brief Returns the reduction indices for computing gradients of "x1" and "x2" with broadcast.

*@par Inputs:
*@li x1: A tensor of type int32 or int64.
*@li x2: A tensor of type int32 or int64. \n
"x2" has the same type as "x1".

*@par Outputs:
*@li y1: A tensor. Reduction indices of "x1".
*@li y2: A tensor. Reduction indices of "x2".
*/
REG_OP(BroadcastGradientArgs)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y1, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y2, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(BroadcastGradientArgs)

/**
*@brief Stops gradient computation. None is returned for the node where the gradient computation is stopped.


*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: The input tensor.
*/
REG_OP(StopGradient)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(StopGradient)

/**
*@brief Return a tensor with the same shape and contents as input.

*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: A tensor.
*/
REG_OP(Identity)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Identity)

/**
*@brief Returns a list of tensors with the same shapes and contents as the input tensors.

*@par Inputs:
*x: A list of input tensors.

*@par Outputs:
*y: A list of Tensor objects, with the same length as the input tensor list.
*/
REG_OP(IdentityN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IdentityN)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data.

*@par Inputs:
*@li x: A tensor.
*@li axis: The dimension index at which to expand.

*@par Outputs:
*y: A tensor.
*/
REG_OP(ExpandDims)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(T, Int, 0)
    .ATTR(Tdim, Int, 0)
    .OP_END_FACTORY_REG(ExpandDims)

/**
*@brief Reshapes a tensor. Only the tensor shape is changed, without changing the data.

*@par Inputs:
*@li x: A tensor.
*@li shape: A tensor. Defines the shape of the output tensor.

*@par Attributes:
*@li axis: An optional int32 or int64. The first dimension to reshape. Defaults to "0".
*@li num_axes: An optional int32 or int64. The extent of the reshape. Defaults to "-1".

*@par Outputs:
*y: A tensor.
*/
REG_OP(Reshape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)

/**
*@brief Removes dimensions of size 1 from the shape of a tensor.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*axis: An optional list of int32 or int64. If not specified, squeezes all dimensions of size 1. \n If specified, only squeezes the dimensions listed. It is an error to squeeze a dimension that is not 1.

*@par Outputs:
*y: A tensor.
*/
REG_OP(Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(T, Int, 0)
    .ATTR(squeeze_dims, ListInt, {})
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(Squeeze)

/**
*@brief Returns an integer representing the rank of input tensor. The rank of a tensor is the number of indices required to uniquely select each element of the tensor, that is, the dimension size of the tensor.

*@par Inputs:
*x: A tensor.

*@par Outputs:
*y: A tensor. The rank of input tensor.
*/
REG_OP(Rank)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Rank)

/**
*@brief Returns the size of a tensor, that is, an integer of the number of elements of the tensor.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*out_type: An optional int32 or int64. The output data type. Defaults to "int32".

*@par Outputs:
*y: A tensor. The size of the input tensor.
*/
REG_OP(Size)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32,DT_INT64}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .ATTR(out_type, Int, DT_INT32)
    .OP_END_FACTORY_REG(Size)

REG_OP(Data)
    .INPUT(data, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

/**
*@brief Inserts a placeholder for a tensor that will be always fed.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*@li peerIndex: An integer type. The index of the corresponding "end" node connected to.
*@li parentId: A string, used to check if the nodes are from the saved parent node.
*@li parentOpType: A string. Op type of the original node.
*@li anchorIndex: An integer, used to check if the node is from the saved anchor.

*@par Outputs:
*y: The created placeholder tensor.
*/
REG_OP(PlaceHolder)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0) // the index of the corresponding 'end' node it's connected to
    .ATTR(parentId, String, "")     // check if these node are from save parent node
    .ATTR(parentOpType, String, "") // op type of original node
    .ATTR(anchorIndex, Int, 0)  // check if these node are from save anchor
    .OP_END_FACTORY_REG(PlaceHolder)

REG_OP(End)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0) // the index of the corresponding 'placeholder' node it's connected to
    .ATTR(parentOpType, String, "") // op type of original node
    .OP_END_FACTORY_REG(End)

REG_OP(Summary)
    .INPUT(x, TensorType::ALL())
    .OP_END_FACTORY_REG(Summary)

/**
*@brief Returns the shape of a tensor.

*@par Inputs:
*x: A tensor.

*@par Attributes:
*out_type: An optional int32 or int64. The output data type. Defaults to int32.

*@par Outputs:
*y: A tensor. The shape of the input tensor.
*/
REG_OP(Shape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .ATTR(out_type, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape)

/**
*@brief Returns shape of tensors.

*@par Inputs:
*x: A list of input tensors.

*@par Attributes:
*out_type: An optional int32 or int64. The output data type. Defaults to "int32".

*@par Outputs:
*y: A list of tensors with the same length as the input list of tensors.
*/
REG_OP(ShapeN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .ATTR(out_type, Int, DT_INT32)
    .OP_END_FACTORY_REG(ShapeN)

/**
*@brief Creates a tensor with the given "shape" and "dtype".

*@par Inputs:
*shape: The shape of the output tensor.

*@par Attributes:
*@li dtype: Optional. The data type of the output tensor. Defaults to "int32".
*@li init: An optional bool. If true, initializes the returned tensor with the default value of "dtype". Defaults to "false".

*@par Outputs:
*y: A tensor.
*/
REG_OP(Empty)
    .INPUT(shape, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Int, DT_INT32)
    .ATTR(init, Bool, 0)
    .OP_END_FACTORY_REG(Empty)

/**
*@brief Gradient op for MirrorPad op. This op folds a mirror-padded tensor.

*@par Inputs:
*The input x and y can be one-dimensional vector. Inputs include: \n
* @li x:A Tensor. The input tensor to be folded.
* @li paddings:A Tensor. Must be one of the following types: int32, int64. A two-column matrix specifying the padding sizes.

*@par Attributes:
*mode:A string from: "REFLECT", "SYMMETRIC". The mode used in the MirrorPad op.

*@par Outputs:
*y:A Tensor. Has the same type as x.

*@attention Constraints: \n
-The implementation for MirrorPadGrad on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not
*/

REG_OP(MirrorPadGrad)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
              DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .INPUT(paddings, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
              DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE }))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPadGrad)

REG_OP(Where)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(Where)

}  // namespace ge

#endif  // GE_OP_ARRAY_OPS_H_
