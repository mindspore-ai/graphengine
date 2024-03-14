/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file index.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_INDEX_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INDEX_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Tutel dispatch function in moe.
*
* @par Inputs:
* @li x: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
* @li locations: A mutable Tensor of the type DT_INT32, for token size.
*
* @par Attributes:
* capacity: expert capacity.
*
* @par Outputs:
* y: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
*/
REG_OP(MoeTutelDispatch)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(gates, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(indices, TensorType({ DT_INT32 }))
    .INPUT(locations, TensorType({ DT_INT32 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .REQUIRED_ATTR(capacity, Int)
    .OP_END_FACTORY_REG(MoeTutelDispatch)

/**
* @brief Tutel combine function in moe.
*
* @par Inputs:
* @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
* @li locations: A mutable Tensor of the type DT_INT32, for token size.
*
* @par Outputs:
* @li x_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
*/
REG_OP(MoeTutelCombineX)
    .INPUT(y_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(gates, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(indices, TensorType({ DT_INT32 }))
    .INPUT(locations, TensorType({ DT_INT32 }))
    .OUTPUT(x_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OP_END_FACTORY_REG(MoeTutelCombineX)

/**
* @brief Tutel combine function in moe.
*
* @par Inputs:
* @li x: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
* @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
* @li locations: A mutable Tensor of the type DT_INT32, for token size.
*
* @par Outputs:
* gates_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
*/
REG_OP(MoeTutelCombineGates)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(y_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .INPUT(indices, TensorType({ DT_INT32 }))
    .INPUT(locations, TensorType({ DT_INT32 }))
    .OUTPUT(gates_grad, TensorType({ DT_FLOAT, DT_FLOAT16, DT_BF16 }))
    .OP_END_FACTORY_REG(MoeTutelCombineGates)

/**
* @brief Tutel combine function in moe.
*
* @par Inputs:
* @li unique_len: A one-dimensional tensor of the type DT_INT32, DT_INT64.
* @li unique_indices:A one-dimensional tensor of the type DT_INT32, DT_INT64.
* @li indices: A two-dimensional tensor of the type DT_INT32, DT_INT64.
* @li values: A mutable Tensor of the type DT_INT32, DT_FLOAT16, DT_FLOAT32.
*
* @par Outputs:
* @li new_inidces: A two-dimensional tensor of the type DT_INT32, DT_INT64.
* @li new_values: A mutable Tensor of the type DT_INT32, DT_FLOAT16, DT_FLOAT32.\n
*/
REG_OP(CoalesceSparse)
    .INPUT(unique_len, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(unique_indices, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(indices, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(values, TensorType({ DT_INT32, DT_FLOAT16, DT_FLOAT32 }))
    .OUTPUT(new_inidces, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(new_values, TensorType({ DT_INT32, DT_FLOAT16, DT_FLOAT32 }))
    .OP_END_FACTORY_REG(CoalesceSparse)

/**
* @brief Give transparency to the image.
*
* @par Inputs:
* @li rgb: A tensor of the type DT_UINT8.
* @li alpha:A tensor of the type DT_UINT8.
*
* @par Outputs:
* @li dst: A tensor of the type DT_UINT8.
*/
REG_OP(MrgbaCustom)
    .INPUT(rgb, TensorType({ DT_UINT8 }))
    .INPUT(alpha, TensorType({ DT_UINT8 }))
    .OUTPUT(dst, TensorType({ DT_UINT8 }))
    .OP_END_FACTORY_REG(MrgbaCustom)

/**
* @brief Replace Background to the image.
* @li bkg: A tensor of the type DT_UINT8, DT_FLOAT16.
* @li src:A tensor of the type DT_UINT8, DT_FLOAT16.
* @li mask:A tensor of the type DT_FLOAT16.
* @li out: A tensor of the type DT_UINT8, DT_FLOAT16.
*/
REG_OP(BackgroundReplace)
    .INPUT(bkg, TensorType({ DT_UINT8, DT_FLOAT16 }))
    .INPUT(src, TensorType({ DT_UINT8, DT_FLOAT16 }))
    .INPUT(mask, TensorType({ DT_FLOAT16, DT_FLOAT16 }))
    .OUTPUT(out, TensorType({ DT_UINT8, DT_FLOAT16 }))
    .OP_END_FACTORY_REG(BackgroundReplace)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_INDEX_H_
