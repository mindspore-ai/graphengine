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

#ifndef GE_OP_RAGGED_CONVERSION_OPS_H
#define GE_OP_RAGGED_CONVERSION_OPS_H
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Converts a RaggedTensor into a SparseTensor with the same values.

*@par Inputs:
*Two inputs, including: \n
*@li rt_nested_splits: A list of at least 1 Tensor objects with the same type \n
in: int32, int64. The row_splits for the RaggedTensor.
*@li rt_dense_values: A Tensor. The flat_values for the RaggedTensor \n
Must be one of the following types: bool, int8, int16, uint16, int32, \n
int64, double, float, float16.

*@par Attributes:
*@li RAGGED_RANK: the dynamic of input rt_nested_splits with type int.
*@li Tsplits: A required attribute, the type is int64.

*@par Outputs:
*@li sparse_indices: A Tensor of type int64.
*@li sparse_values: A Tensor. Has the same type as rt_dense_values.
*@li sparse_dense_shape: A Tensor of type int64.

*/
REG_OP(RaggedTensorToSparse)
    .DYNAMIC_INPUT(rt_nested_splits, TensorType({DT_INT32, DT_INT64}))
    .INPUT(rt_dense_values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(sparse_indices, TensorType({DT_INT64}))
    .OUTPUT(sparse_values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(sparse_dense_shape, TensorType({DT_INT64}))
    .ATTR(RAGGED_RANK, Int, 1)
    .ATTR(Tsplits, Type, DT_INT64)
    .OP_END_FACTORY_REG(RaggedTensorToSparse)
} // namespace ge
#endif // GE_OP_RAGGED_CONVERSION_OPS_H