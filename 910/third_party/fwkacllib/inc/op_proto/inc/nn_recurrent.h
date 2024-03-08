/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file nn_recurrent.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_RECURRENT_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_RECURRENT_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief BasicLSTMInplaceFillWindowCache calculation.
* @par Inputs:
* eight inputs: \n
* @li x:Each time step is a 3D Tensor. Must be one of the following types: float16.
* @li w:Each direction is a 3D Tensor. Must be one of the following types: int8.
* @li r:Each direction is a 3D Tensor. Must be one of the following types: int8.
* @li h:Each direction is a 3D Tensor. Must be one of the following types: float16.
* @li c:Each direction is a 3D Tensor. Must be one of the following types: float16.
* @li b:An optional input. Each direction is a 2D Tensor. Must be one of the following types: int32.
* @li sequence_lens:An optional input. A 1D Tensor. Must be one of the following types: int32.
* @li clean_cache:An optional input. A 1D Tensor. Must be one of the following types: bool.
* @li deq_scale:A 1D Tensor. Must be one of the following types: uint64.

* @par Attributes:
* @li hidden_size:Number of neurons in the hidden layer. Requied. Reserved.
* @li activation_alpha: Optional scaling values used by some activation functions. Empty is currently supported.
* @li activation_beta: Optional scaling values used by some activation functions. Empty is currently supported.
* @li activations: A list of strings of activation functions. Empty is currently supported.
* @li clip:An float identifying the cell clip in the op. Default to -1.
* @li direction: Specify if the RNN is forward, reverse, or bidirectional. Must be forward(default).
* @li input_forget:Couple the input and forget gates if 1. Reserved.
* @li quant_scale_x: A float identifying the quant_scale of x_tensor. Default to -0.0.
* @li quant_offset_x:A float identifying the quant_offset of x_tensor. Default to -0.0.
* @li quant_sqrt_mode_x:A sqrt_mode of x_tensor. Default to False.
* @li quant_scale_h:A float identifying the quant_scale of h_tensor. Default to -0.0.
* @li quant_offset_h:A float identifying the quant_offset of h_tensor. Default to -0.0.
* @li quant_sqrt_mode_h:A sqrt_mode of h_tensor. Default to False.
* @li quant_dtype:An Int number identifying the dtype of quant. Default to 2(DT_INT8).

* @par Outputs:
* three outputs: \n
* @li y:First dimension is time step, second dimension is direction, others is a 4D Tensor. Must be one of the following types: float16.
* @li y_h:Each direction is a 3D Tensor. Must be one of the following types: float16.
* @li y_c:Each direction is a 3D Tensor. Must be one of the following types: float16.
*/

REG_OP(BasicLSTMInplaceFillWindowCache)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(w, TensorType({DT_INT8}))
    .INPUT(r, TensorType({DT_INT8}))
    .INPUT(h, TensorType({DT_FLOAT16}))
    .INPUT(c, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(b, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(sequence_lens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(clean_cache, TensorType({DT_BOOL}))
    .INPUT(deq_scale, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(h, TensorType({DT_FLOAT16}))
    .OUTPUT(c, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(hidden_size, Int)
    .ATTR(activation_alpha, ListFloat, {})
    .ATTR(activation_beta, ListFloat, {})
    .ATTR(activations, ListString, {})
    .ATTR(clip, Float, -1.0)
    .ATTR(direction, String, "forward")
    .ATTR(input_forget, Int, 0)
    .ATTR(quant_scale_x, Float, 0.0)
    .ATTR(quant_offset_x, Float, 0.0)
    .ATTR(quant_sqrt_mode_x, Bool, false)
    .ATTR(quant_scale_h, Float, 0.0)
    .ATTR(quant_offset_h, Float, 0.0)
    .ATTR(quant_sqrt_mode_h, Bool, false)
    .ATTR(quant_dtype, Int, DT_INT8)
    .OP_END_FACTORY_REG(BasicLSTMInplaceFillWindowCache)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_RECURRENT_H_
