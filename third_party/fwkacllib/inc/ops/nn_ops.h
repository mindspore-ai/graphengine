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
 * \file nn_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OPS_H_
#include "graph/operator_reg.h"
#include "nn_pooling_ops.h"

namespace ge {
/**
* @brief Says whether the targets are in the top "k" predictions . \n

* @par Inputs:
* Three inputs, including:
* @li predictions: A 2D Tensor of type float32. A "batch_size * classes" tensor.
* @li targets: A 1D Tensor of type IndexNumberType. A batch_size tensor of class ids.
* @li k: A 1D Tensor of the same type as "targets".
* Specifies the number of top elements to look at for computing precision . \n

* @par Outputs:
* precision: A Tensor of type bool . \n

* @attention Constraints:
* @li targets must be non-negative tensor.

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator InTopKV2.
*/
REG_OP(InTopKV2)
    .INPUT(predictions, TensorType({DT_FLOAT}))
    .INPUT(targets, TensorType(IndexNumberType))
    .INPUT(k, TensorType({IndexNumberType}))
    .OUTPUT(precision, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(InTopKV2)
}// namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_OPS_H_
