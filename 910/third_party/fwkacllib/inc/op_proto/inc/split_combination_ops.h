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
 * \file split_combination_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors .

* @par Inputs:
* Two inputs, including:
* @li x: An ND Tensor.
* Must be one of the types:float16, float32, double, int64, int32, uint8,
  uint16, uint32, uint64, int8, int16, bool, complex64, complex128, qint8,
  quint8, qint16, quint16, qint32, bfloat16.
* @li split_dim: Must be the following type:int32. Specifies the dimension along which to split . \n

* @par Attributes:
* @li num_split: A required int includes all types of int.
  Specifies the number of output tensors. No default value . \n

* @par Outputs:
* @li y: Dynamic output.A list of output tensors. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "num_split" is greater than or equals to 1.
* @li "num_split" is divisible by the size of dimension "split_dim".
* @li "split_dim" is in the range [-len(x.shape), len(x.shape)-1] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Split.
*/
REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                          DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                          DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                          DT_BF16,       DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                                   DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                                   DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                                   DT_BF16,       DT_BOOL}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

/**
* @brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors .

* @par Inputs:
* One input:
* x:An ND Tensor.
* Must be one of the following types: float16, float32, int32, int8, int16,
  int64, uint8, uint16, uint32, uint64, bool, bfloat16. \n

* @par Attributes:
* @li split_dim: A required int includes all types of int.
  Specifies the dimension along which to split. No default value.
* @li num_split: A required int includes all types of int.
  Specifies the number of output tensors. No default value . \n

* @par Outputs:
* y:Dynamic output. A list of output tensors. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "num_split" is greater than or equals to 1.
* @li "num_split" is divisible by the size of dimension "split_dim".
* @li "split_dim" is in the range [-len(x.shape), (x.shape)-1] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Split. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Split instead.
*/
REG_OP(SplitD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                          DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                                   DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitD)

/**
* @brief Splits a tensor along dimension "split_dim" into "num_split"
  smaller tensors according to "size_splits" .

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor.
* Must be one of the types:float16, float32, double, int64, int32, uint8,
  uint16, uint32, uint64, int8, int16, bool, complex64, complex128, qint8,
  quint8, qint16, quint16, qint32, string, bfloat16.
* @li size_splits: Must be one of the IndexNumberType:int32, int64.
  Specifies a list containing the sizes of each output tensor along the split dimension.
* @li split_dim: Must be the following type:int32, int64. Specifies the
  dimension along which to split . \n

* @par Attributes:
* @li num_split: A required int includes all types of int. Specifies the number of output tensors.
  No default value . \n

* @par Outputs:
* @li y:  Dynamic output.A list of output tensors.
  Has the same type and format as "x" . \n

* @attention Constraints:
* @li Each element in "size_splits" is greater than or equal to 1.
* @li "size_splits" and "num_split" have the same length.
* @li The elements in "size_splits" sum to the size of dimension "split_dim" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SplitV.
*/
REG_OP(SplitV)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                          DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                          DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                          DT_BF16, DT_BOOL, DT_STRING}))
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                                   DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                                   DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                                   DT_BF16, DT_BOOL, DT_STRING}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)

/**
* @brief Splits a tensor along dimension "split_dim" into "num_split"
  smaller tensors according to "size_splits".

* @par Inputs:
* One input:
* @li x: An ND Tensor.
* Must be one of the following types: float16, float32, int32, int8, int16,
  int64, uint8, bfloat16, uint16, uint32, uint64, bool.

* @par Attributes:
* @li size_splits: A required list of int32. Specifies a list containing
  the sizes of each output tensor along the split dimension.
* @li split_dim: A required int32. Specifies the dimension along which to split. No default value.
* @li num_split: A required int32. Specifies the number of output tensors. No default value .

* @par Outputs:
* @li y: Dynamic output.A list of output tensors. Has the same type and format as "x" .

* @attention Constraints:
* @li Each element in "size_splits" is greater than or equal to 1.
* @li "size_splits" and "num_split" have the same length.
 Under the caffe framework, the conversion of slice_point through
 the cut point to cut segment is mapped to size_splits.
* @li The elements in "size_splits" sum to the size of dimension "split_dim".
 Under the caffe framework,size_splits or axis transformat to split_dim.Only one can effect.
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SplitV.

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use SplitV instead.
*/
REG_OP(SplitVD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                          DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                                   DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .REQUIRED_ATTR(size_splits, ListInt)
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitVD)

/**
* @brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors. 
        All outputs of the PhonySplit are allocated with the same memory block. 
        GE calculates the memory offset of each output, 
        and the custom operators next read the memory by offset.
* Warning: This operator is used only to identify that the GE allocates continuous memory and does not perform any calculation..

* @par Inputs:
* One input:
* @li x:An ND Tensor.
* Must be one of the following types: float16, float32, int32, int8, int16,
  int64, uint8, uint16, uint32, uint64, bool, bfloat16.

* @par Attributes:
* @li split_dim: A required int32. Specifies the dimension along which to split. No default value.
* @li num_split: A required int32. Specifies the number of output tensors. No default value .
* @li keep_output_offset: A optional Bool. Specifies whether calculate the memory offset of outpute tensors. Default True .

* @par Outputs:
* @li y:Dynamic output. A list of output tensors. Has the same type and format as "x" .

* @attention Constraints:
* @li "num_split" is greater than or equals to 1.
* @li "num_split" is divisible by the size of dimension "split_dim".
* @li "split_dim" is in the range [-len(x.shape), (x.shape)-1] .

* @par Third-party framework support
* Support ONNX  framework.
*/
REG_OP(PhonySplit)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                          DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                                   DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .REQUIRED_ATTR(split_dim, ListInt)
    .REQUIRED_ATTR(num_split, ListInt)
    .ATTR(keep_output_offset, Bool, true)
    .OP_END_FACTORY_REG(PhonySplit)


/**
* @brief Concatenates a list of "N" tensors along "concat_dim". 
        All input of the PhonyConcat are allocated with the same memory block. 
        GE calculates the memory offset of each input, 
        and the custom operators before write the memory by offset.
* Warning: This operator is used only to identify that the GE allocates continuous memory and does not perform any calculation..

* @par Inputs:
* Dynamic input: A list of input tensors. Has the same type and format as "x" .
* @li x:An ND Tensor.
* Must be one of the following types: float16, float32, int32, int8, int16,
  int64, uint8, uint16, uint32, uint64, bool, bfloat16.

* @par Attributes:
* @li concat_dim: A required list of int32. Specifies the dimensions along which to concat. No default value.
* @li N: A required list of int32. Specifies the number of concat tensors. No default value .
* @li keep_input_offset: A optional Bool. Specifies whether calculate the memory offset of input tensors. Default True .

* @par Outputs:
* @li y:One output.

* @attention Constraints:
* @li "concat_dim" is in the range [-len(x.shape), (x.shape)-1] .
* @li "N" is greater than or equals to 1.

* @par Third-party framework support
* Support ONNX  framework.
*/
REG_OP(PhonyConcat)
    .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                          DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_BF16,
                                   DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16, DT_BOOL}))
    .REQUIRED_ATTR(concat_dim, ListInt)
    .REQUIRED_ATTR(N, ListInt)
    .ATTR(keep_input_offset, Bool, true)
    .OP_END_FACTORY_REG(PhonyConcat)
    
/**
*@brief Concatenates a list of N tensors along the first dimension.
*@par Inputs:
* One input, including:
* values: A list of Tensors. Must be one of the following types: int8, int16,
* int32, int64, uint8, uint16, uint32, uint64, float16, float32.
* Tensors to be concatenated. All must have size 1 in the first dimension
* and same shape. It's a dynamic input.

*@par Attributes:
* @li shape: A required list of ints.
* @li N: A required int. The numble of dynamic_input "values" .

*@par Outputs:
*output_data: The concatenated tensor with same type as "values".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ParallelConcat.
*/
REG_OP(ParallelConcat)
    .DYNAMIC_INPUT(values, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .OUTPUT(output_data, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ParallelConcat)

/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* One input:
* x: Dynamic input.A ND Tensor.
*    Must be one of the following types: bfloat16, float16, float32, int32,
*    int8, int16, int64, uint8, uint16, uint32, uint64. \n

* @par Attributes:
* concat_dim: A required int include all types of int.
              Specifies the dimension along which to concatenate. No default value.
* N: An optional int include all types of int. Specifies the number of elements in "x". Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "x" is a list of at least 2 "tensor" objects of the same type.
* @li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ConcatV2.
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ConcatV2 instead.
*/
REG_OP(ConcatV2D)
    .DYNAMIC_INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_INT64,
                                  DT_UINT64, DT_UINT32, DT_INT16, DT_UINT16, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_INT64, DT_UINT64,
                           DT_UINT32, DT_INT16, DT_UINT16, DT_UINT8}))
    .REQUIRED_ATTR(concat_dim, Int)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2D)

/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* Two inputs, including:
* @li Dynamic input "x" is A ND Tensor.
* Must be one of the following types: bfloat16, float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, uint16,
*     complex128, uint32, uint64, qint16, quint16, bool, string.
* @li concat_dim: An int32, or int64. Specifies the dimension along which to concatenate . \n

* @par Attributes:
* N: An optional int includes all types of int.
* Specifies the number of elements in "x". Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* "x" is a list of at least 2 "tensor" objects of the same type . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ConcatV2.
*/
REG_OP(ConcatV2)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2)

/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* One input:
* x: Dynamic input. A ND Tensor.
*    Must be one of the following types: bfloat16, float16, float32, int32,
*    int8, int16, int64, uint8, uint16, uint32, uint64. \n

* @par Attributes:
* @li concat_dim: A required int8, int16, int32, or int64.
                  Specifies the dimension along which to concatenate.
                  No default value.
* @li N:  An optional int8, int16, int32, or int64.
  Specifies the number of elements in "x". Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "x" is a list of at least 2 "tensor" objects of the same type.
* @li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Concat.
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Concat instead.
*/
REG_OP(ConcatD)
    .DYNAMIC_INPUT(x, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                                  DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_INT32,
                           DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(concat_dim, Int)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatD)

/**
* @brief Concatenates tensors along one dimension .

* @par Inputs:
* Two inputs, including:
* @li x: Dynamic input.A ND Tensor.
* Must be one of the BasicType: 
  complex128, complex64, double, float32, float16, int16, int32, int64, int8,
  qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
  bfloat16, complex32.
* @li concat_dim: Must be one of the IndexNumberType: int32, int64.
* Specifies the dimension along which to concatenate . \n

* @par Attributes:
* N: An optional int8, int16, int32, or int64. Specifies the number of elements in "x" .
  Defaults to "1". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as "x" . \n

* @attention Constraints:
* @li "x" is a list of at least 2 "tensor" objects of the same type.
* @li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Concat. \n
*/
REG_OP(Concat)
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Concat)

/**
*@brief Packs the list of tensors in values into a tensor with rank one higher
* than each tensor in values, by packing them along the axis dimension.
* Given a list of length N of tensors of shape (A, B, C); if axis == 0 then
* the output tensor will have the shape (N, A, B, C) .

*@par Inputs:
* x: A list of N Tensors. Must be one of the following types: complex128,
* complex64, double, float32, float16, int16, int32, int64, int8, qint16,
* qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8, bfloat16,
* complex32. It's a dynamic input.

*@par Attributes:
*@li axis: An optional int, default value is 0.
*     Dimension along which to pack. The range is [-(R+1), R+1).
*@li N: An optional int, default value is 1. Number of tensors.

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Pack.
*/
REG_OP(Pack)
    .DYNAMIC_INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(axis, Int, 0)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Pack)

/**
*@brief Computes offsets of concat inputs within its output .

*@par Inputs:
*Two inputs, including:
* @li concat_dim: A Tensor of type int32.
* @li x: A list of 1D Tensor objects of type int32 . It's a dynamic input. \n

*@par Attributes:
*N: A required int includes all types of int. \n

*@par Outputs:
*y: A Tensor list with same type as "x" . It's a dynamic output. \n

*@par Third-party framework compatibility
*@ Compatible with the TensorFlow operator ConcatOffset.
*/
REG_OP(ConcatOffset)
    .INPUT(concat_dim, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatOffset)

/**
*@brief Computes offsets of concat inputs within its output .

*@par Inputs:
*x: A list of 1D Tensor objects of type int32 . It's a dynamic input. \n

*@par Attributes:
*@li concat_dim: A required int includes all types of int.
*Must be within the rank of input "x".
*@li N: A required int includes all types of int. \n

*@par Outputs:
*y: A Tensor list with same type as "x" . It's a dynamic output. \n

*@par Third-party framework compatibility
*@ Compatible with the TensorFlow operator ConcatOffset. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use ConcatOffset instead.
*/
REG_OP(ConcatOffsetD)
    .DYNAMIC_INPUT(x, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(concat_dim, Int)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatOffsetD)

/**
*@brief Compute combinations of length of the given tensor.

*@par Inputs:
*x:  A list of 1D Tensor objects.

*@par Attributes:
*@li r: An optional int indicates number of elements to combine. Defaults to 2.
*@li with_replacement: An optional bool indicates whether to allow duplication
*in combination. Defaults to "False".

*@par Outputs:
*y: A Tensor list with same type as "x" .

*@par Third-party framework compatibility
*@ Compatible with the Pytorch operator Combinations.
*/
REG_OP(Combinations)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(r, Int, 2)
    .ATTR(with_replacement, Bool, false)
    .OP_END_FACTORY_REG(Combinations)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
