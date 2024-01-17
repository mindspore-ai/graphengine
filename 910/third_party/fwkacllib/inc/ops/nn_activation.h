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
 * \file nn_activation.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_

#include "graph/operator_reg.h"

namespace ge{
        /**
    * @brief Compute the SwiGlu,
    * where the activations function in GLU is Swish.

    * @par Inputs:
    * One input, including:
    * @x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Outputs:
    * one output, including:
    * @y: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Attributes:
    * two attributes, including:
    * @li dim: A optional int. The dimension to be split, default is -1.

    * @par Third-party framework compatibility:
    * New operator SwiGlu.

    * @par Restrictions:
    * Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(SwiGlu)
        .INPUT(x, "T")
        .OUTPUT(y, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .OP_END_FACTORY_REG(SwiGlu)

    /**
    * @brief Compute the SwiGluGrad,
    * where the activations function in GLU is SwishGrad.

    * @par Inputs:
    * two input, including:
    * @li y_grad: A Tensor. Must be one of the following types: bfloat16, float16, float32.
    * @li x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Outputs:
    * one Output, including:
    * @x_grad: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Attributes:
    * one attributes, including:
    * @li dim: A optional int. The dimension to be split, default is -1.

    * @par Third-party framework compatibility:
    * New operator SwiGluGrad.

    * @par Restrictions:
    * Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(SwiGluGrad)
        .INPUT(y_grad, "T")
        .INPUT(x, "T")
        .OUTPUT(x_grad, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .OP_END_FACTORY_REG(SwiGluGrad)
        
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_
