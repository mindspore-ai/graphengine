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

#ifndef GE_OP_TRANSFORMATION_OPS_H
#define GE_OP_TRANSFORMATION_OPS_H

#include "../graph/operator_reg.h"

namespace ge {
REG_OP(DepthwiseWeight4DTo6D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT16}))
    .OP_END_FACTORY_REG(DepthwiseWeight4DTo6D)

REG_OP(DepthwiseWeight6DTo4D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT16}))
    .ATTR(channel_size, Int, 16)
    .OP_END_FACTORY_REG(DepthwiseWeight6DTo4D)

/**
*@brief Permutes the dimensions according to perm.\n
        The returned tensor's dimension i will correspond to the input dimension perm[i].

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.

*@par Attributes:
*perm: A permutation of the dimensions of "x".

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(TransposeD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(perm, ListInt, {})
    .OP_END_FACTORY_REG(TransposeD)

/**
*@brief Permutes the dimensions according to perm.\n
        The returned tensor's dimension i will correspond to the input dimension perm[i].

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
*@li perm: A Tensor of type int32 or int64. A permutation of the dimensions of "x".

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(Transpose)
    .INPUT(x, TensorType::BasicType())
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Transpose)

REG_OP(Flatten)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                          DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                           DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(Flatten)

REG_OP(BatchToSpaceND)
    .INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(BatchToSpaceND)

REG_OP(BatchToSpaceNDD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_shape, ListInt)
    .REQUIRED_ATTR(crops, ListInt)
    .OP_END_FACTORY_REG(BatchToSpaceNDD)

REG_OP(SpaceToBatchND)
    .INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(SpaceToBatchND)

REG_OP(SpaceToBatchNDD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_shape, ListInt)
    .REQUIRED_ATTR(paddings, ListInt)
    .OP_END_FACTORY_REG(SpaceToBatchNDD)

REG_OP(SpaceToDepth)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(SpaceToDepth)

/**
*@brief Rearranges data from depth into blocks of spatial data.

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, double, int32, uint8,
*     int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64

*@par Attributes:
*Two attributes, including:
* @li block_size: An int >= 2, specifying the size of the spatial block.
* @li data_format: An optional string, specifying the data format. Defaults to "NHWC".

*@par Outputs:
*y: A Tensor of the same type as "x".
*/
REG_OP(DepthToSpace)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(DepthToSpace)

/**
*@brief Permutes data into spatial data blocks and then prunes them.

*@par Inputs:
*x: A 4D Tensor with format NC1HWC0. \n

*Must be one of the following types: float16, float32

*@par Attributes:
*@li crops: A required list of int8, int16, int32, or int64. No default value.
*@li block_size: A required int8, int16, int32, or int64. No default value.

*@par Outputs:
*y: A 4D Tensor with format NC1HWC0, \n

* of type float16 or float32.

*@attention Constraints:
*@li The size of the first dimension of input "x" must be divisible by (block_size * block_size).
*@li "crops" is a 2D tensor of non-negative integers with shape (2, 2).
*@li block_size >= 2
*/
REG_OP(BatchToSpace)
    .INPUT(x, TensorType::BasicType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(BatchToSpace)

/**
*@brief Rearrange the batch (permutes) data into spatial data blocks, and then crop them.

*@par Inputs:
* One input:
*x: An Tensor of shape [batch*block_size*block_size, height_pad/block_size, width_pad/block_size, depth].\n
*The batch size of the input tensor must be divisible by (block size * block size).

*@par Attributes:
*@li block_size: Must be one of the following types: `int32`, `int64`.
*@li crops: An Tensor. Must be one of the following types: int32, Int64.\n
*2D tensor with non negative integer of shape [2, 2]. It specifies how many\n
*elements are clipped from the intermediate result of spatial dimension.

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".

*@attention Constraints:
*@li The size of the first dimension of input "x" must be divisible by (block_size * block_size).
*@li "crops" is a 2D tensor of non-negative integers with shape (2, 2).
*@li block_size >= 2
*/
REG_OP(BatchToSpaceD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8,
                        DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_COMPLEX64,
                        DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8,
                        DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_COMPLEX64,
                        DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32}))
    .REQUIRED_ATTR(block_size, Int)
    .REQUIRED_ATTR(crops, ListInt)
    .OP_END_FACTORY_REG(BatchToSpaceD)

REG_OP(SpaceToBatch)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(SpaceToBatch)

REG_OP(SpaceToBatchD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .REQUIRED_ATTR(paddings, ListInt)
    .OP_END_FACTORY_REG(SpaceToBatchD)

/**
* @brief Unpacks the given dimension of a rank-R tensor "value" into rank-(R-1)
* tensors.

* @par Inputs:
* @ value: A rank-R tensor (R > 0) of type BasicType, with format ND or NC1HWC0.

* @par Attributes:
* @li num: An optional int, specifying the number of tensors to be unpacked to.
* Defaults to "None".
* @li axis: A required int, specifying the axis to unpack along. The value range
* is [-R, R).

* @par Outputs:
* output: The list of Tensor objects unpacked from "value", of type BasicType.

* @attention Constraints:
* @li If "num" is not specified, it is inferred from the shape of "value".
* @li For the ND format, "axis" is in the range [-R, R); For the NC1HWC0 format,
* "axis" must not be 2, 3, -2, or -3.
*/
REG_OP(Unpack)
    .INPUT(value, TensorType::BasicType())
    .DYNAMIC_OUTPUT(output, TensorType::BasicType())
    .REQUIRED_ATTR(num, Int)
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(Unpack)

/**
* @brief Extract "patches" from "images" and stacks them in the "depth"
* dimension of the output.

* @par Inputs:
* images: A 4D Tensor with shape [batch, in_rows, in_cols, depth].

* @par Attributes:
* @li ksizes: An optional tuple or list. size of the sliding window for
* each dimension of images.
* @li strides: An optional tuple or list. How far the centers of two
* consecutive patches are in the images.\n
* Must be: [1, stride_rows, stride_cols, 1].
* @li rates: Must be: An optional tuple or list. [1, rate_rows, rate_cols, 1].
* This is the input stride,\n
* specifying how far two consecutive patch samples are in the input. Equivalent\n
* to extracting patches with patch_sizes_eff = patch_sizes + (patch_sizes - 1) *\n
* (rates - 1), followed by subsampling them spatially by a factor of rates. This\n
* is equivalent to rate in dilated (a.k.a. Atrous) convolutions.
* @li padding: An optional string. The type of padding algorithm to use.

* @par Outputs:
* Output: A 4D Tensor with shape [batch, out_rows, out_cols, ksize_rows *\n
* ksize_cols * depth] containing image patches with size ksize_rows x ksize_cols\n
* x depth vectorized in the "depth" dimension. Note "out_rows" and "out_cols"\n
* are the dimensions of the output patches.

* @attention Constraints:
* "ksizes", "strides" and "rates" are lists of integers.
*/
REG_OP(ExtractImagePatches)
    .INPUT(images, TensorType::REALNUMBERTYPE())
    .OUTPUT(y, TensorType::REALNUMBERTYPE())
    .ATTR(ksizes, ListInt, {1,3,3,1})
    .ATTR(strides, ListInt, {1,1,1,1})
    .ATTR(rates, ListInt, {1,1,1,1})
    .ATTR(padding, String, "SAME")
    .OP_END_FACTORY_REG(ExtractImagePatches)

/**
*@brief Confuse reshape and transpose.

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.

*@par Attributes:
*@li perm: A permutation of the dimensions of "x".
*@li shape: The shape of the input.
*@li transpose_first: If True, the transpose is first, otherwise the reshape is first.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(ConfusionTransposeD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(perm, ListInt)
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(transpose_first, Bool)
    .OP_END_FACTORY_REG(ConfusionTransposeD)

/**
*@brief Confuse reshape and transpose.

*@par Inputs:
*@li x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
*@li shape: The shape of the input.

*@par Attributes:
*@li perm: A permutation of the dimensions of "x".
*@li transpose_first: If True, the transpose is first, otherwise the reshape is first.

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(ConfusionTranspose)
    .INPUT(x, TensorType::BasicType())
    .INPUT(shape, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(perm, ListInt)
    .REQUIRED_ATTR(transpose_first, Bool)
    .OP_END_FACTORY_REG(ConfusionTranspose)

}  // namespace ge

#endif  // GE_OP_TRANSFORMATION_OPS_H
