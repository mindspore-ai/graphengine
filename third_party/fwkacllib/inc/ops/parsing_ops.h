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
 * \file parsing_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Converts each string in the input Tensor to the specified numeric type . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: string . \n

*@par Attributes:
*out_type: The numeric type to interpret each string in string_tensor as . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU, with bad performance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow StringToNumber operator.
*/
REG_OP(StringToNumber)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .ATTR(out_type, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StringToNumber)

/**
*@brief Convert serialized tensorflow.TensorProto prototype to Tensor.
*@brief Parse an Example prototype. 
*@par Input:
*serialized: A Tensor of type string.
*dense_defaults:  DYNAMIC INPUT Tensor type as string, float, int64. \n

*@par Attributes:
*num_sparse: type int num of inputs sparse_indices , sparse_values, sparse_shapes
*out_type: output type
*sparse_keys: ListString
*sparse_types: types of sparse_values
*dense_keys: ListString
*dense_shapes: output of dense_defaults shape
*dense_types: output of dense_defaults type  \n

*@par Outputs:
*sparse_indices: A Tensor of type string. 
*sparse_values:  Has the same type as sparse_types.
*sparse_shapes: A Tensor of type int64
*dense_values:  Has the same type as dense_defaults.

*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
**/
REG_OP(ParseSingleExample)
    .INPUT(serialized, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(dense_defaults, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_values, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(dense_values, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .ATTR(num_sparse, Int, 0)
    .ATTR(sparse_keys, ListString, {})
    .ATTR(dense_keys, ListString, {})
    .ATTR(sparse_types, ListType, {})
    .ATTR(Tdense, ListType, {})
    .ATTR(dense_shapes, ListListInt, {})
    .OP_END_FACTORY_REG(ParseSingleExample)

/**
*@brief Decodes raw file into  tensor . \n
*@par Input:
*bytes: A Tensor of type string.

*@par Attributes:
*little_endian: bool ture
*out_type: output type

*@par Outputs:
*Output: A Tensor
**/
REG_OP(DecodeRaw)
    .INPUT(bytes, TensorType({DT_STRING}))
    .OUTPUT(output, TensorType({DT_BOOL,DT_FLOAT16,DT_DOUBLE,DT_FLOAT,
                                    DT_INT64,DT_INT32,DT_INT8,DT_UINT8,DT_INT16,
                                    DT_UINT16,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(out_type, Type, DT_FLOAT)
    .ATTR(little_endian, Bool, true)
    .OP_END_FACTORY_REG(DecodeRaw)

/**
*@brief Convert serialized tensorflow.TensorProto prototype to Tensor. \n

*@par Inputs:
*serialized: A Tensor of string type. Scalar string containing serialized
*TensorProto prototype. \n

*@par Attributes:
*out_type: The type of the serialized tensor. The provided type must match the
*type of the serialized tensor and no implicit conversion will take place. \n

*@par Outputs:
*output: A Tensor of type out_type. \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU,
*with badperformance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow ParseTensor operator.
*/
REG_OP(ParseTensor)
    .INPUT(serialized, TensorType({DT_STRING}))
    .OUTPUT(output, TensorType(DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                          DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(out_type, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(ParseTensor)

/**
*@brief Converts each string in the input Tensor to the specified numeric
*type . \n

*@par Inputs:
*Inputs include:
*records: Each string is a record/row in the csv and all records should have the
*same format. \n
*record_defaults: One tensor per column of the input record, with either a
*scalar default value for that column or an empty vector if the column is
*required. \n

*@par Attributes:
*OUT_TYPE: The numeric type to interpret each string in string_tensor as . \n
*field_delim: char delimiter to separate fields in a record. \n
*use_quote_delim: If false, treats double quotation marks as regular characters
*inside of the string fields (ignoring RFC 4180, Section 2, Bullet 5). \n
*na_value: Additional string to recognize as NA/NaN. \n

*@par Outputs:
*output: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU, with bad
*performance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow StringToNumber operator.
*/
REG_OP(DecodeCSV)
    .INPUT(records, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(record_defaults, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32,
                                        DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32,
                                        DT_INT64, DT_STRING}))
    .ATTR(OUT_TYPE, ListType, {})
    .ATTR(field_delim, String, ",")
    .ATTR(use_quote_delim, Bool, true)
    .ATTR(na_value, String, ",")
    .ATTR(select_cols, ListInt, {})
    .OP_END_FACTORY_REG(DecodeCSV)

/**
*@brief Convert serialized tensorflow.TensorProto prototype to Tensor.
*@brief Parse an Example prototype.
*@par Input:
*serialized: A Tensor of type string. \n
*name:A Tensor of type string. \n
*sparse_keys: Dynamic input tensor of string. \n
*dense_keys: Dynamic input tensor of string \n
*dense_defaults:  Dynamic input tensor type as string, float, int64. \n

*@par Attributes:
*Nsparse: Number of sparse_keys, sparse_indices and sparse_shapes \n
*Ndense: Number of dense_keys \n
*sparse_types: types of sparse_values \n
*Tdense: Type of dense_defaults dense_defaults and dense_values \n
*dense_shapes: output of dense_defaults shape  \n

*@par Outputs:
*sparse_indices: A Tensor of type string. \n
*sparse_values:  Has the same type as sparse_types. \n
*sparse_shapes: A Tensor of type int64 \n
*dense_values:  Has the same type as dense_defaults. \n
*@par Third-party framework compatibility \n
*@li compatible with tensorflow StringToNumber operator. \n
*/
REG_OP(ParseExample)
    .INPUT(serialized, TensorType({DT_STRING}))
    .INPUT(name, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(sparse_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(dense_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(dense_defaults, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(dense_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .ATTR(Nsparse, Int, 0)
    .ATTR(Ndense, Int, 0)
    .ATTR(sparse_types, ListType, {})
    .ATTR(Tdense, ListType, {})
    .ATTR(dense_shapes, ListListInt, {})
    .OP_END_FACTORY_REG(ParseExample)

/**
*@brief Transforms a scalar brain.SequenceExample proto (as strings) into typed
*tensors.
*@par Input:
*serialized: A Tensor of type string. \n
*feature_list_dense_missing_assumed_empty:A Tensor of type string. \n
*context_sparse_keys: Dynamic input tensor of string. \n
*context_dense_keys: Dynamic input tensor of string \n
*feature_list_sparse_keys:  Dynamic input tensor of string \n
*feature_list_dense_keys:  Dynamic input tensor of string \n
*context_dense_defaults:  Dynamic input tensor of string, float, int64 \n
*debug_name: A Tensor of type string. \n

*@par Attributes:
*Ncontext_sparse: Number of context_sparse_keys, context_sparse_indices and context_sparse_shapes \n
*Ncontext_dense: Number of context_dense_keys \n
*Nfeature_list_sparse: Number of feature_list_sparse_keys \n
*Nfeature_list_dense: Number of feature_list_dense_keys \n
*context_sparse_types: Types of context_sparse_values \n
*Tcontext_dense: Number of dense_keys \n
*feature_list_dense_types: Types of feature_list_dense_values \n
*context_dense_shapes: Shape of context_dense \n
*feature_list_sparse_types: Type of feature_list_sparse_values \n
*feature_list_dense_shapes: Shape of feature_list_dense \n

*@par Outputs:
*context_sparse_indices: Dynamic output tensor of type int64. \n
*context_sparse_values:  Dynamic output tensor of type string, float, int64. \n
*context_sparse_shapes: Dynamic output tensor of type int64 \n
*context_dense_values:  Dynamic output tensor of type string, float, int64. \n
*feature_list_sparse_indices: Dynamic output tensor of type int64. \n
*feature_list_sparse_values:  Dynamic output tensor of type string, float, int64. \n
*feature_list_sparse_shapes: Dynamic output tensor of type int64 \n
*feature_list_dense_values:  Dynamic output tensor of type string, float, int64. \n
*@par Third-party framework compatibility \n
*@li compatible with tensorflow StringToNumber operator. \n
*/
REG_OP(ParseSingleSequenceExample)
    .INPUT(serialized, TensorType({DT_STRING}))
    .INPUT(feature_list_dense_missing_assumed_empty, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(context_sparse_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(context_dense_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(feature_list_sparse_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(feature_list_dense_keys, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(context_dense_defaults, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .INPUT(debug_name, TensorType({DT_STRING}))
    .DYNAMIC_OUTPUT(context_sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(context_sparse_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(context_sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(context_dense_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(feature_list_sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(feature_list_sparse_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .DYNAMIC_OUTPUT(feature_list_sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(feature_list_dense_values, TensorType({DT_FLOAT, DT_INT64, DT_STRING}))
    .ATTR(Ncontext_sparse, Int, 0)
    .ATTR(Ncontext_dense, Int, 0)
    .ATTR(Nfeature_list_sparse, Int, 0)
    .ATTR(Nfeature_list_dense, Int, 0)
    .ATTR(context_sparse_types, ListType, {})
    .ATTR(Tcontext_dense, ListType, {})
    .ATTR(feature_list_dense_types, ListType, {})
    .ATTR(context_dense_shapes, ListListInt, {})
    .ATTR(feature_list_sparse_types, ListType, {})
    .ATTR(feature_list_dense_shapes, ListListInt, {})
    .OP_END_FACTORY_REG(ParseSingleSequenceExample)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_
