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

#ifndef GE_OP_NONLINEAR_FUC_OPS_H
#define GE_OP_NONLINEAR_FUC_OPS_H

#include "../graph/operator_reg.h"

namespace ge {
/**
*@brief Computes the for the gelu of "x".

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32

*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(Gelu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Gelu)

/**
*@brief Computes the gradient for the gelu of "x".

*@par Inputs:
*Two inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32
* @li x: A Tensor of the same type as "dy".
* @li y: A Tensor of the same type as "dy".

*@par Outputs:
*z: A Tensor. Has the same type as "dy".
*/
REG_OP(GeluGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GeluGrad)

/**
*@brief Computes the gradient for the tanh of "x".

*@par Inputs:
*Two inputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32,
*     double, complex64, complex128.
* @li dy: A Tensor of the same type as "y".

*@par Outputs:
*z: A Tensor. Has the same type as "y".
*/
REG_OP(TanhGrad)
    .INPUT(y, TensorType::UnaryDataType())
    .INPUT(dy, TensorType::UnaryDataType())
    .OUTPUT(z, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(TanhGrad)

REG_OP(Tanh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Tanh)

/**
* @brief Computes rectified linear: "max(x, 0)".
*
* @par Inputs:
* x: A tensor. Must be one of the following types: float32, float64, int32, uint8,\n
*     int16, int8, int64, uint16, float16, qint8.
*
* @par Outputs:
* y: A tensor. Has the same type as "x".
*
*/
REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                           DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_QINT8}))
    .OP_END_FACTORY_REG(Relu)

/**
* @brief Computes rectified linear 6.
* activations = min(max(features, 0), 6).

* @par Inputs:
* features: A Tensor of type RealNumberType.

* @par Outputs:
* activations: A Tensor of type RealNumberType.
*/
REG_OP(Relu6)
    .INPUT(features, TensorType::RealNumberType())
    .OUTPUT(activations, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(Relu6)

/**
* @brief Computes rectified linear 6 gradients for a Relu6 operation.
*     z = dy * (y > 0) * (y < 6).

* @par Inputs:
* @li y: A Tensor of type RealNumberType.
* @li dy: A Tensor of type RealNumberType.

* @par Outputs:
* z: A Tensor of type RealNumberType.
*/
REG_OP(Relu6Grad)
    .INPUT(y, TensorType::RealNumberType())
    .INPUT(dy, TensorType::RealNumberType())
    .OUTPUT(z, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(Relu6Grad)

/**
* @brief Compute sigmoid of "x" element-wise.

* @par Inputs:
* A Tensor of type UnaryDataType.

* @par Outputs:
* A Tensor. Has the same type as "x".

* @attention Constraints:
* @li "x" is with shape (D1, D2, ..., DK), where, D1 * D2... * Dn <= 2^31-1,
* Di <= 1000000, n <= 8.
* @li Ascend 310 provides only 1?? accuracy for the result.

* @see Relu()
*/
REG_OP(Sigmoid)
    .INPUT(x, TensorType(UnaryDataType))
    .OUTPUT(y, TensorType(UnaryDataType))
    .OP_END_FACTORY_REG(Sigmoid)

/**
* @brief Computes z = (y - y*y)*dy.

* @par Inputs:
* @li y: the input is tensor , dtype is UnaryDataType.
* @li dy the input is tensor , dtype is UnaryDataType.

* @par Outputs:
* z: the shape of output, dtype is UnaryDataType.
*/
REG_OP(SigmoidGrad)
    .INPUT(y, TensorType(UnaryDataType))
    .INPUT(dy, TensorType(UnaryDataType))
    .OUTPUT(z, TensorType(UnaryDataType))
    .OP_END_FACTORY_REG(SigmoidGrad)

REG_OP(Activation)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    /*
       0:sigmod, 1:relu, 2:tanh, 3:clipped ReLU, 4:Elu,
       5:leaky relu, 6:abs, 7:relu1, 8:softsign, 9:softplus
    */
    .ATTR(mode, Int, 1)
    .ATTR(coef, Float, 0)
    .OP_END_FACTORY_REG(Activation)

REG_OP(ActivationGrad)
    .INPUT(dy, TensorType{DT_FLOAT})
    .INPUT(x, TensorType{DT_FLOAT})
    .OUTPUT(dx, TensorType{DT_FLOAT})
    .ATTR(mode, Int, 1)
    .OP_END_FACTORY_REG(ActivationGrad)

REG_OP(Softplus)
    .INPUT(features, TensorType::FloatingDataType())
    .OUTPUT(activations, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Softplus)

REG_OP(SoftplusGrad)
    .INPUT(gradients, TensorType::FloatingDataType())
    .INPUT(features, TensorType::FloatingDataType())
    .OUTPUT(backprops, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(SoftplusGrad)

REG_OP(Softsign)
    .INPUT(features, TensorType::FloatingDataType())
    .OUTPUT(activations, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(Softsign)

REG_OP(Selu)
    .INPUT(features, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,
                                 DT_INT8,DT_INT32}))
    .OUTPUT(activations, TensorType({DT_FLOAT16,DT_FLOAT,DT_DOUBLE,
                                     DT_INT8,DT_INT32}))
    .OP_END_FACTORY_REG(Selu)

REG_OP(ReluGrad)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(features, TensorType::RealNumberType())
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGrad)

/**
*@brief Computes rectified linear gradients for a ReLU operation.

*@par Inputs:
* Two inputs, including:
*@li gradients: A Tensor. Must be one of the following types: float32, double, int32, int8, int16,\n int8, int64, uint16, float16, uint32, uint64
*@li mask: A Tensor. Must be the following types: uint8

*@par Outputs:
*backprops: A Tensor. Must have the same type as"gradients".

*@attention Constraints:
* The corresponding Relu operator needs to be called before using this operator on the network.

*@see Relu
*/
REG_OP(ReluGradV2)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGradV2)

/**
*@brief Computes rectified linear: `max(x, 0)`.
*
*@attention Constraints:\n
* The last dim must be mutiply of 8
* The second output `mask` is the result of `y` use 'gt' compare with 0.
*
*@par Inputs:
* x: A tensor. Must be one of the following types: float32, float64, int32, uint8,
*     int16, int8, int64, uint16, float16, qint8.
*
*@par Outputs:
*@li y : A `Tensor`. Has the same type as `x`.
*@li mask : A `Tensor`. Must be the type : `uint8`.
*
*/
REG_OP(ReluV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(mask, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(ReluV2)

REG_OP(PRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(PRelu)

REG_OP(PReluGrad)
    .INPUT(input_gradients, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_weights, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_backprops_dx, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_backprops_da, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(PReluGrad)

/**
*@brief Computes exponential linear: `exp(x) - 1` if < 0, `x` otherwise.
*
*@par Inputs:
* x : A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
*
*@par Outputs:
* y : A `Tensor`. Has the same type as `x`.
*
*/
REG_OP(Elu)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .ATTR(alpha, Float, 1.0)
    .OP_END_FACTORY_REG(Elu)

/**
*@brief Computes gradients for the exponential linear (Elu) operation.
*
*@par Inputs:
*@li grads : A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
*     The backpropagated gradients to the corresponding Elu operation.
*@li activations : A `Tensor`. Must have the same type as `grads`.
*     The outputs of the corresponding Elu operation.
*
*@par Outputs:
* y : A `Tensor`. Has the same type as `grads`.
*
*/
REG_OP(EluGrad)
    .INPUT(grads, TensorType::FloatingDataType())
    .INPUT(activations, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(EluGrad)

REG_OP(LeakyRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .ATTR(negative_slope, Float, 0.0)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OP_END_FACTORY_REG(LeakyRelu)

} // namespace ge

#endif // GE_OP_NONLINEAR_FUC_OPS_H
