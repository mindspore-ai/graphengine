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

#ifndef GE_OP_LOOKUP_OPS_H_
#define GE_OP_LOOKUP_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(LookupTableImport)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(keys, TensorType({DT_BOOL, DT_DOUBLE, \
        DT_FLOAT, DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_DOUBLE, \
        DT_FLOAT, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(LookupTableImport)

REG_OP(LookupTableInsert)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(keys, TensorType({DT_BOOL, DT_DOUBLE, DT_FLOAT, \
        DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_BOOL, DT_DOUBLE, DT_FLOAT, \
        DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(LookupTableInsert)

REG_OP(LookupTableExport)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(keys, TensorType({DT_BOOL, DT_DOUBLE, DT_FLOAT, \
        DT_INT32, DT_INT64}))
    .OUTPUT(values, TensorType({DT_BOOL, DT_DOUBLE, DT_FLOAT, \
        DT_INT32,DT_INT64}))
    .REQUIRED_ATTR(Tkeys, Type)
    .REQUIRED_ATTR(Tvalues, Type)
    .OP_END_FACTORY_REG(LookupTableExport)
REG_OP(LookupTableSize)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(size, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(LookupTableSize)

REG_OP(LookupTableFind)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(keys, TensorType({DT_DOUBLE, DT_FLOAT, \
        DT_INT32, DT_INT64}))
    .INPUT(default_value, TensorType({DT_DOUBLE, DT_FLOAT, \
        DT_INT32, DT_INT64}))
    .OUTPUT(values, TensorType({DT_DOUBLE, DT_FLOAT, DT_INT32, \
        DT_INT64}))
    .REQUIRED_ATTR(Tout, Type)
    .OP_END_FACTORY_REG(LookupTableFind)

REG_OP(HashTable)
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(use_node_name_sharing, Bool, false)
    .REQUIRED_ATTR(key_dtype, Type)
    .REQUIRED_ATTR(value_dtype, Type)
    .OP_END_FACTORY_REG(HashTable)

REG_OP(InitializeTable)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(keys, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(InitializeTable)

REG_OP(MutableDenseHashTable)
    .INPUT(empty_key, TensorType({DT_INT32, DT_INT64}))
    .INPUT(deleted_key, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(handle, TensorType({DT_RESOURSE}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(use_node_name_sharing, Bool, false)
    .REQUIRED_ATTR(value_dtype, Type)
    .ATTR(value_shape, ListInt, {})
    .ATTR(initial_num_buckets, Int, 131072)
    .ATTR(max_load_factor, Float, 0.8)
    .OP_END_FACTORY_REG(MutableDenseHashTable)

REG_OP(MutableHashTableOfTensors)
    .OUTPUT(handle, TensorType({DT_RESOURSE}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(use_node_name_sharing, Bool, false)
    .REQUIRED_ATTR(key_dtype, Type)
    .REQUIRED_ATTR(value_dtype, Type)
    .ATTR(value_shape, ListInt, {})
    .OP_END_FACTORY_REG(MutableHashTableOfTensors)

REG_OP(MutableHashTable)
    .OUTPUT(handle, TensorType({DT_RESOURSE}))
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .ATTR(use_node_name_sharing, Bool, false)
    .REQUIRED_ATTR(key_dtype, Type)
    .REQUIRED_ATTR(value_dtype, Type)
    .OP_END_FACTORY_REG(MutableHashTable)
}   // namespace ge

#endif  // GE_OP_LOOKUP_OPS_H_
