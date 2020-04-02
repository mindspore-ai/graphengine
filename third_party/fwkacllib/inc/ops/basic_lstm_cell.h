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

#ifndef GE_OP_BASIC_LSTM_CELL_H
#define GE_OP_BASIC_LSTM_CELL_H

#include "../graph/operator_reg.h"

namespace ge {
/**
*@brief: Basic LSTM Cell forward calculation.
*@par Inputs:
*five inputs: \n
*@li x:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li h:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li w:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li b:A 1D Tensor. Must be one of the following types: float16. The format must be ND.

*@par Attributes:
*@li keep_prob:An integer identifying the keep prob in the op. Default to 1.
*@li forget_bias:An integer identifying the forget bias in the op. Default to 1.
*@li state_is_tuple:An bool identifying if the hidden state and cell state is tuple. Default to true.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is currently supported.

*@par Outputs:
*seven outputs: \n
*@li mask:A 1D Tensor. Must be one of the following types: uint8.
*@li ct:A 4D Tensor. Must be one of the following types: float16, float32.
*@li ht:A 4D Tensor. Must be one of the following types: float16.
*@li it:A 4D Tensor. Must be one of the following types: float16, float32.
*@li jt:A 4D Tensor. Must be one of the following types: float16, float32.
*@li ft:A 4D Tensor. Must be one of the following types: float16, float32.
*@li ot:A 4D Tensor. Must be one of the following types: float16, float32.
*@li tanhct:A 4D Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(BasicLSTMCell)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(h, TensorType({DT_FLOAT16}))
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(ct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(ht, TensorType({DT_FLOAT16}))
    .OUTPUT(it, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(jt, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(ft, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(ot, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(tanhct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(forget_bias, Float, 1.0)
    .ATTR(state_is_tuple, Bool, true)
    .ATTR(activation, String, "tanh")
    .OP_END_FACTORY_REG(BasicLSTMCell)

/**
*@brief: Basic LSTM Cell backward calculation.Calculate the gradient of input and hidden state.
*@par Inputs:
*three inputs: \n
*@li dgate:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li w:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_Z.
*@li dropout_mask:A 1D Tensor. Must be one of the following types: uint8. The format must be ND.

*@par Attributes:
*keep_prob:An integer identifying the keep prob in the op. Default to 1.

*@par Outputs:
*two outputs: \n
*@li dxt:A 4D Tensor. Must be one of the following types: float16, float32.
*@li dht:A 4D Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(BasicLSTMCellInputGrad)
    .INPUT(dgate, TensorType({DT_FLOAT16}))
    .INPUT(w, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(dropout_mask, TensorType({DT_UINT8}))
    .OUTPUT(dxt, TensorType({DT_FLOAT16}))
    .OUTPUT(dht, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .ATTR(keep_prob, Float, 1.0)
    .OP_END_FACTORY_REG(BasicLSTMCellInputGrad)

/**
*@brief: Basic LSTM Cell backward calculation.Calculate the gradient of weight and bias.
*@par Inputs:
*three inputs: \n
*@li x:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li h:A 4D Tensor. Must be one of the following types: float16. The format must be FRACTAL_NZ.
*@li dgate:A 4D Tensor. Must be one of the following types: uint8. The format must be FRACTAL_NZ.

*@par Outputs:
*two outputs: \n
*@li dw:A 4D Tensor. Must be one of the following types: float16.
*@li db:A 4D Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(BasicLSTMCellWeightGrad)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(h, TensorType({DT_FLOAT16}))
    .INPUT(dgate, TensorType({DT_FLOAT16}))
    .OUTPUT(dw, TensorType({DT_FLOAT16}))
    .OUTPUT(db, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OP_END_FACTORY_REG(BasicLSTMCellWeightGrad)

/**
*@brief: Basic LSTM Cell backward calculation.Calculate the gradient of gates and cell state.
*@par Inputs:
*eight inputs: \n
*@li c:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dht:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li dct:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li it:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li jt:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li ft:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li ot:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.
*@li tanhct:A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.

*@par Attributes:
*@li forget_bias:An integer identifying the forget bias in the op. Default to 1.
*@li activation:An string identifying the type of activation function in the op. Default to "tanh". Only tanh is currently supported.

*@par Outputs:
*two outputs: \n
*@li dgate:A 4D Tensor. Must be one of the following types: float16.
*@li dct_1:A 4D Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(BasicLSTMCellCStateGrad)
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dht, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(it, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(jt, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(ft, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(ot, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(tanhct, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dgate, TensorType({DT_FLOAT16}))
    .OUTPUT(dct_1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(forget_bias, Float, 1.0)
    .ATTR(activation, String, "tanh")
    .OP_END_FACTORY_REG(BasicLSTMCellCStateGrad)
}  // namespace ge

#endif  // GE_OP_BASIC_LSTM_CELL_H
