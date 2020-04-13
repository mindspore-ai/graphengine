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

#ifndef GE_OP_NN_NORM_OPS_H
#define GE_OP_NN_NORM_OPS_H

#include "../graph/operator_reg.h"
namespace ge {

/**
*@brief Computes the gradient for log softmax activations.

*@par Inputs:
*@li grad: A Tensor. Must be one of the following types: float16, float32.
*@li x: A Tensor. Must be one of the following types: float16, float32.

*@par Attributes:
* axis: An optional list of ints. Defaults to "{-1}".

*@par Outputs:
* y: A Tensor. Has the same type as "grad".
*/

REG_OP(LogSoftmaxGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxGrad)

REG_OP(SparseSoftmaxCrossEntropyWithLogitsCCE)
    .INPUT(features, TensorType{DT_FLOAT})
    .INPUT(labels, TensorType{DT_FLOAT})
    .OUTPUT(out, TensorType{DT_FLOAT})
    .OUTPUT(non, TensorType{DT_FLOAT})
    .ATTR(cross_entropy_is_grad, Bool, 0)
    .ATTR(cross_entropy_mode, Int, 1)
    .ATTR(softmax_cross_entropy_lossscale_div_batch, Float, 1.0)
    .OP_END_FACTORY_REG(SparseSoftmaxCrossEntropyWithLogitsCCE)

/**
*@brief Computes sparse softmax cross entropy cost and gradients to backpropagate.

*@par Inputs:
*Two inputs, including:
* @li features: A Tensor. Must be one of the following types: half, float32, double.
*    A "batch_size * num_classes" matrix.
* @li labels: A Tensor of the same type as "features". batch_size vector with values in [0, num_classes).


*@par Outputs:
*loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). Has the same type as "features".
*/
REG_OP(SparseSoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(loss, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(SparseSoftmaxCrossEntropyWithLogits)

/**
*@brief Computes softmax cross entropy cost and gradients to backpropagate.

*@par Inputs:
*Two inputs, including:
* @li features: A Tensor. Must be one of the following types: half, float32, double.
*    A "batch_size * num_classes" matrix.
* @li labels: A Tensor of the same type as "features". A "batch_size * num_classes" matrix.

*@par Outputs:
*loss: A Tensor for per example loss (a "batch_size" vector). Has the same type as "features".
*backprop: A Tensor for the backpropagated gradients (a batch_size * num_classes matrix). Has the same type as "features".
*/
REG_OP(SoftmaxCrossEntropyWithLogits)
    .INPUT(features, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .INPUT(labels, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(backprop, TensorType({DT_DOUBLE,DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(SoftmaxCrossEntropyWithLogits)

REG_OP(SoftmaxGrad)
    .INPUT(softmax, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(grad_softmax, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OP_END_FACTORY_REG(SoftmaxGrad)

REG_OP(SigmoidCrossEntropyWithLogitsGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogitsGrad)

REG_OP(SigmoidCrossEntropyWithLogits)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogits)

REG_OP(SmoothL1Loss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1Loss)

REG_OP(SmoothL1LossGrad)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(sigma, Float, 1.0)
    .OP_END_FACTORY_REG(SmoothL1LossGrad)

REG_OP(BinaryCrossEntropy)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropy)

REG_OP(BinaryCrossEntropyGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad_output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(BinaryCrossEntropyGrad)

/**
*@brief Applies the Softmax function to an n-dimensional input Tensor rescaling them \n so 
that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.

*@par Inputs:
*One input:
*x: A mutable Tensor. Must be one of the following types: float16,
*float32, double. Should be a Variable Tensor.

*@par Attributes:
*axis: A list of ints. The dimension softmax would be performed on.

*@par Outputs:
*y: A Tensor. Has the same dimensionality and shape as the "x" with values in the range [0, 1]. Must be one of the following types: float16, float32, int32.
*/
REG_OP(Softmax)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(Softmax)

/**
*@brief Computes log softmax activations.

*@par Inputs:
*One input:
* logits: A Tensor. Must be one of the following types: double, float16, float32.

*@par Attributes:
* axis: An optional list of ints. Defaults to "{-1}".

*@par Outputs:
* logsoftmax: A Tensor. Has the same type as "logits".
*/
REG_OP(LogSoftmax)
    .INPUT(logits, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(logsoftmax, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT}))
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmax)

REG_OP(FusedBatchNormV2)
    .INPUT(x, TensorType{DT_FLOAT})                  /* Input data tensor from the previous operator"" */
    .INPUT(scale, TensorType{DT_FLOAT})              /* If spatial is true, the dimension of bias is (C) If spatial is false, the dimensions of scale are (C x D1 x ... x Dn)*/
    .INPUT(b, TensorType{DT_FLOAT})                  /* If spatial is true, the dimension of bias is (C) If spatial is false, the dimensions of scale are (C x D1 x ... x Dn)*/
    .OPTIONAL_INPUT(mean, TensorType{DT_FLOAT})               /* If spatial is true, the dimension of the running mean (training) or the estimated mean (testing) is (C).If spatial is false, the dimensions of the running mean (training) or the estimated mean (testing) are (C x D1 x ... x Dn)*/
    .OPTIONAL_INPUT(variance, TensorType{DT_FLOAT})           /* If spatial is true, the dimension of the running variance(training) or the estimated variance (testing) is (C). If spatial is false, the dimensions of the running variance(training) or the estimated variance (testing) are (C x D1 x ... x Dn).*/
    .OUTPUT(y, TensorType{DT_FLOAT})                 /* The output tensor of the same shape as X */
    .ATTR(momentum, Float, 0.9)            // Factor used in computing the running mean and variance.
    .ATTR(epsilon, Float, 1e-5f)           // The epsilon value to use to avoid division by zero
    .ATTR(mode, Int, 1)                    // 1 means using "CC_BATCHNORM_SPATIAL"; 0 means using "CC_BATCHNORM_PER_ACTIVATION"; only support 1 now
    .ATTR(use_global_stats, Bool, true)
    .ATTR(alpha, Float, 1)
    .ATTR(beta, Float, 0)
    .OP_END_FACTORY_REG(FusedBatchNormV2)

REG_OP(Scale)
    .INPUT(x, TensorType{DT_FLOAT})
    .OPTIONAL_INPUT(w, TensorType{DT_FLOAT})
    .OPTIONAL_INPUT(b, TensorType{DT_FLOAT})
    .OUTPUT(y, TensorType{DT_FLOAT})
    .ATTR(bias_term, Bool, false)
    .ATTR(axis, Int, 1)
    .ATTR(num_axis, Int, 1)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(Scale)

REG_OP(SoftmaxGradExt)
  .INPUT(grad, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x1, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x2, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
  .ATTR(axis, ListInt, {-1})
  .ATTR(keep_dims, Bool, false)
  .OP_END_FACTORY_REG(SoftmaxGradExt)

/**
*@brief Confuse mul, sum and sub.

*@par Inputs:
*Two inputs, including:
* @li grad: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.

*@par Outputs:
* y: A Tensor of the same type as "grad".

*/
REG_OP(ConfusionSoftmaxGrad)
  .INPUT(grad, TensorType({DT_FLOAT16,DT_FLOAT}))
  .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
  .OP_END_FACTORY_REG(ConfusionSoftmaxGrad)

/**
*@brief Layernorm operator interface implementation
*  calculating: x, gamma, beta
*  mean  = np.mean(x, reduce_axis, keepdims=True)
*  variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True)
*  y = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

*@par Inputs:
*Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.
* @li beta: A Tensor. Must be one of the following types: float16, float32.

*@par Attributes:
* @li begin_norm_axis: A required attribute, the type is int32.
* @li begin_params_axis: A required attribute,the type is int32.

*@par Outputs:
*Three outputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(LayerNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(begin_norm_axis, Int, 0)
    .ATTR(begin_params_axis, Int, 0)
    .ATTR(epsilon, Float, 0.0000001)
    .OP_END_FACTORY_REG(LayerNorm)

/**
*@brief LayerNormGrad operator interface implementation
*  calculating: dy, x, variance, mean, gamma
*  pd_xl = data_dy*data_gamma
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power((data_variance + EPSLON), (-1.5))),
*           reduce_axis, keepdims=True)
*  pd_mean = np.sum(((-1.0)*pd_xl
*            np.power((data_variance + EPSLON), (-0.5))),
*            reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True)
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) +
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True)
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.

*@par Outputs:
*Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(LayerNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(LayerNormGrad)

/**
*@brief LayerNormXBackprop operator interface implementation
*  calculating: dy, x, variance, mean, gamma
*  pd_xl = data_dy*data_gamma
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power((data_variance + EPSLON), (-1.5))),
*           reduce_axis, keepdims=True)
*  pd_mean = np.sum(((-1.0)*pd_xl
*            np.power((data_variance + EPSLON), (-0.5))),
*            reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True)
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) +
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True)
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.
* @li gamma: A Tensor. Must be one of the following types: float16, float32.

*@par Outputs:
*Three outputs, including:
* @li pd_x: A Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(LayerNormXBackprop)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(LayerNormXBackprop)

/**
*@brief LayerNormBetaGammaBackprop operator interface implementation
*  calculating: dy, x, variance, mean
*  pd_xl = data_dy*data_gamma
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power((data_variance + EPSLON), (-1.5))),
*           reduce_axis, keepdims=True)
*  pd_mean = np.sum(((-1.0)*pd_xl
*            np.power((data_variance + EPSLON), (-0.5))),
*            reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True)
*  pd_x = pd_xl*np.power((data_variance + EPSLON), (-0.5)) +
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)
*             np.power((data_variance + EPSLON), (-0.5))), param_axis, keepdims=True)
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor. Must be one of the following types: float16, float32.
* @li variance: A Tensor. Must be one of the following types: float16, float32.
* @li mean: A Tensor. Must be one of the following types: float16, float32.

*@par Outputs:
*Three outputs, including:
* @li pd_gamma: A Tensor. Must be one of the following types: float16, float32.
* @li pd_beta: A Tensor. Must be one of the following types: float16, float32.
*/
REG_OP(LayerNormBetaGammaBackprop)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(shape_gamma, ListInt)
    .OP_END_FACTORY_REG(LayerNormBetaGammaBackprop)

/**
*@brief Return "output" according to the algorithm of dropout_do_mask: \n
*  scale_x = x *(1 / keep_prob)
*  output = select(mask == 1, scale_x, 0)

*@par Inputs:
*Three inputs, including: \n
* @li x: A mutable Tensor. Must be one of the following types:
*     float16, float32
* @li mask: A mutable Tensor. Must met all of the following rules:
*     shape of mask should be 1D.
*     dtype of mask should be uint8.
*     value of shape should met the following algorithm:
*     value = (size(x) + 128 - 1) // 128 * 128 //8
* @li keep_prob: A mutable Tensor. Must met all of the following rules:
*     shape of "keep_prob" should be (1,) or [1,].
*     Has the same type as "x".

*@par Output:
*y: A mutable Tensor. Has the same type as "x".
*/
REG_OP(DropOutDoMask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(DropOutDoMask)

/**
*@brief Local Response Normalization.

*@par Inputs:
*One input, including:
*@li x: A Tensor. Must be 4-D shape, and only support the following types: float16, float32.

*@par Attributes:
*@li depth_radius: An optional int, specifying the half-width of the
* normalization window. Defaults to "5".
*@li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1".
*@li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1".
*@li beta: An optional float32. An exponent. Defaults to "0.5".
*@li norm_region: An optional string. A mode option. Defaults to "ACROSS_CHANNELS".

*@par Outputs:
*y: A Tensor. Has the same data type and shape as "x".
*/
REG_OP(LRN)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .ATTR(norm_region, String, "ACROSS_CHANNELS")
    .OP_END_FACTORY_REG(LRN)

/**
* @brief Computes the gradient for Local Response Normalization.

* @par Inputs:
* @li grads: A 4D Tensor of type float16 or float32.
* @li x: A 4D Tensor of type float16 or float32.
* @li y: A 4D Tensor of type float16 or float32.

* @par Attributes:
* @li depth_radius: An optional int, specifying the half-width of the
* normalization window. Defaults to "5".
* @li bias: An optional float32. An offset, usually > 0 to avoid dividing by 0.
* Defaults to "1".
* @li alpha: An optional float32. A scaling factor, usually positive.
* Defaults to "1".
* @li beta: An optional float32. An exponent. Defaults to "0.5".

* @par Outputs:
* z: A Tensor. Has the same type and shape as "grads".

* @attention Constraints:
* "x" and "y" must have the same shape and type as "grads".
*/
REG_OP(LRNGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .OP_END_FACTORY_REG(LRNGrad)

}  // namespace ge

#endif  //GE_OP_NN_NORM_OPS_H
