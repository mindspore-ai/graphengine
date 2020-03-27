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

#ifndef GE_OP_SPARSE_OPS_H_
#define GE_OP_SPARSE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(SparseSoftmax)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSoftmax)

REG_OP(SparseTensorDenseAdd)
    .INPUT(x1_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT}))
    .INPUT(x1_shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT}))
    .OP_END_FACTORY_REG(SparseTensorDenseAdd)

REG_OP(SparseReorder)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseReorder)

REG_OP(SparseReshape)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(shape, TensorType({DT_INT64}))
    .INPUT(new_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseReshape)

REG_OP(SparseDenseCwiseAdd)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseDenseCwiseAdd)

REG_OP(SparseDenseCwiseDiv)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseDenseCwiseDiv)

REG_OP(SparseDenseCwiseMul)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, \
        DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseDenseCwiseMul)

REG_OP(AddSparseToTensorsMap)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(handle, TensorType({DT_INT64}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(AddSparseToTensorsMap)

REG_OP(SparseSliceGrad)
    .INPUT(backprop_val_grad, TensorType({ DT_INT8, DT_UINT8, DT_INT16,
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE }))
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(start, TensorType({DT_INT64}))
    .INPUT(new_indices, TensorType({DT_INT64}))
    .OUTPUT(y_grad, TensorType({ DT_INT8, DT_UINT8, DT_INT16,
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE }))
    .OP_END_FACTORY_REG(SparseSliceGrad)

REG_OP(SparseSlice)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE }))
    .INPUT(shape, TensorType({DT_INT64}))
    .INPUT(start, TensorType({DT_INT64}))
    .INPUT(size, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE }))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseSlice)

REG_OP(SparseAddGrad)
    .INPUT(backprop_val_grad, TensorType({DT_INT8, DT_INT16, DT_INT32,
                                          DT_INT64, DT_FLOAT, DT_DOUBLE}))
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(sum_indices, TensorType({DT_INT64}))
    .OUTPUT(x1_val_grad, TensorType({DT_INT8, DT_INT16, DT_INT32,
                                     DT_INT64, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(x2_val_grad, TensorType({DT_INT8, DT_INT16, DT_INT32,
                                     DT_INT64, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseAddGrad)

REG_OP(SparseFillEmptyRowsGrad)
    .INPUT(reverse_index_map, TensorType({DT_INT64}))
    .INPUT(grad_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_value, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y_default_value, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseFillEmptyRowsGrad)

REG_OP(SparseTensorDenseMatMul)
    .INPUT(x1_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_INT32, DT_DOUBLE}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_DOUBLE}))
    .ATTR(adjoint_a, Bool, false)
    .ATTR(adjoint_b, Bool, false)
    .OP_END_FACTORY_REG(SparseTensorDenseMatMul)

REG_OP(SparseToDense)
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(output_shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_DOUBLE}))
    .INPUT(default_value, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, \
        DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_DOUBLE}))
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SparseToDense)

REG_OP(SparseConcat)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_INPUT(shapes, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(concat_dim, Int, 0)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(SparseConcat)

REG_OP(SparseAdd)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, \
        DT_INT64, DT_DOUBLE}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .INPUT(thresh, TensorType({DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, \
        DT_INT64, DT_DOUBLE}))
    .OUTPUT(sum_indices, TensorType({DT_INT64}))
    .OUTPUT(sum_values, TensorType({DT_FLOAT, DT_INT8, DT_INT16, \
        DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(sum_shape, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseAdd)

REG_OP(SparseFillEmptyRows)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(dense_shape, TensorType({DT_INT64}))
    .INPUT(default_value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, \
        DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, \
        DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(empty_row_indicator, TensorType({DT_BOOL}))
    .OUTPUT(reverse_index_map, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(SparseFillEmptyRows)

REG_OP(SparseSparseMaximum)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSparseMaximum)

REG_OP(SparseSparseMinimum)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseSparseMinimum)

REG_OP(SparseReduceMax)
    .INPUT(x_indices, TensorType({DT_INT64}))
    .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x_shape, TensorType({DT_INT64}))
    .INPUT(reduction_axes, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SparseReduceMax)

REG_OP(SparseReduceMaxSparse)
    .INPUT(x_indices, TensorType({DT_INT64}))
    .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x_shape, TensorType({DT_INT64}))
    .INPUT(reduction_axes, TensorType({DT_INT32}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SparseReduceMaxSparse)

REG_OP(SparseReduceSum)
    .INPUT(x_indices, TensorType({DT_INT64}))
    .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x_shape, TensorType({DT_INT64}))
    .INPUT(reduction_axes, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SparseReduceSum)

REG_OP(SparseReduceSumSparse)
    .INPUT(x_indices, TensorType({DT_INT64}))
    .INPUT(x_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(x_shape, TensorType({DT_INT64}))
    .INPUT(reduction_axes, TensorType({DT_INT32}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SparseReduceSumSparse)

REG_OP(SparseSplit)
    .INPUT(split_dim, TensorType({DT_INT64}))
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y_values, TensorType({DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_BOOL, \
        DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(num_split, Int, 1)
    .OP_END_FACTORY_REG(SparseSplit)

REG_OP(SparseCross)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(values, TensorType({DT_INT64, DT_STRING}))
    .DYNAMIC_INPUT(shapes, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(dense_inputs, TensorType({DT_INT64, DT_STRING}))
    .OUTPUT(output_indices, TensorType({DT_INT64}))
    .OUTPUT(output_values, TensorType({DT_INT64, DT_STRING}))
    .OUTPUT(output_shape, TensorType({DT_INT64}))
    .ATTR(N, Int, 0)
    .REQUIRED_ATTR(hashed_output, Bool)
    .ATTR(num_buckets, Int, 0)
    .REQUIRED_ATTR(hash_key, Int)
    .REQUIRED_ATTR(out_type, Type)
    .REQUIRED_ATTR(internal_type, Type)
    .OP_END_FACTORY_REG(SparseCross)

REG_OP(AddManySparseToTensorsMap)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(handles, TensorType({DT_INT64}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(AddManySparseToTensorsMap)

REG_OP(TakeManySparseFromTensorsMap)
    .INPUT(handles, TensorType({DT_INT64}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLAOT16}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(TakeManySparseFromTensorsMap)

REG_OP(SerializeSparse)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLAOT16}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(serialized_sparse, TensorType({DT_STRING}))
    .ATTR(out_type, Type, DT_STRING)
    .OP_END_FACTORY_REG(SerializeSparse)

REG_OP(SerializeManySparse)
    .INPUT(indices, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLAOT16}))
    .INPUT(shape, TensorType({DT_INT64}))
    .OUTPUT(serialized_sparse, TensorType({DT_STRING}))
    .ATTR(out_type, Type, DT_STRING)
    .OP_END_FACTORY_REG(SerializeManySparse)

REG_OP(DeserializeSparse)
    .INPUT(serialized_sparse, TensorType({DT_STRING}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLAOT16}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(DeserializeSparse)

REG_OP(DeserializeManySparse)
    .INPUT(serialized_sparse, TensorType({DT_STRING}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, \
        DT_UINT16, DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT, DT_FLAOT16}))
    .OUTPUT(shape, TensorType({DT_INT64}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(DeserializeManySparse)
}  // namespace ge

#endif  // GE_OP_SPARSE_OPS_H_
