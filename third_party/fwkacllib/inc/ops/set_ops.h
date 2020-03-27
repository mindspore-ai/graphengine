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

#ifndef GE_OP_SET_OPS_H_
#define GE_OP_SET_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

REG_OP(DenseToDenseSetOperation)
  .INPUT(x1, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                         DT_INT32, DT_INT64, DT_STRING}))
  .INPUT(x2, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                         DT_INT32, DT_INT64, DT_STRING}))
  .OUTPUT(y_indices, TensorType({DT_INT64}))
  .OUTPUT(y_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                DT_INT32, DT_INT64, DT_STRING}))
  .OUTPUT(y_shape, TensorType({DT_INT64}))
  .ATTR(set_operation, String, "")
  .ATTR(validate_indices, Bool, true)
  .OP_END_FACTORY_REG(DenseToDenseSetOperation)

REG_OP(DenseToSparseSetOperation)
    .INPUT(x1, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                           DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(set_operation, String, "")
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(DenseToSparseSetOperation)

REG_OP(SparseToSparseSetOperation)
    .INPUT(x1_indices, TensorType({DT_INT64}))
    .INPUT(x1_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(x1_shape, TensorType({DT_INT64}))
    .INPUT(x2_indices, TensorType({DT_INT64}))
    .INPUT(x2_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .INPUT(x2_shape, TensorType({DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT64}))
    .OUTPUT(y_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, \
                                  DT_INT32, DT_INT64, DT_STRING}))
    .OUTPUT(y_shape, TensorType({DT_INT64}))
    .ATTR(set_operation, String, "")
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SparseToSparseSetOperation)

REG_OP(SetSize)
    .INPUT(set_indices, TensorType({DT_INT64}))
    .INPUT(set_values, TensorType({DT_INT8, DT_INT16, \
        DT_UINT8, DT_UINT16, DT_INT32, DT_INT64}))
    .INPUT(set_shape, TensorType({DT_INT64}))
    .OUTPUT(size, TensorType({DT_INT32}))
    .ATTR(validate_indices, Bool, true)
    .OP_END_FACTORY_REG(SetSize)
}  // namespace ge

#endif  // GE_OP_SET_OPS_H_
