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

#ifndef GE_OP_REDUCE_OPS_H
#define GE_OP_REDUCE_OPS_H

#include "../graph/operator_reg.h"

namespace ge {
REG_OP(BNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingReduce)

REG_OP(BNTrainingReduceGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(diff_scale, TensorType({DT_FLOAT}))
    .INPUT(diff_offset, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BNTrainingReduceGrad)

REG_OP(BNTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(factor, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdate)

REG_OP(BNInfer)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(BNInfer)

REG_OP(BNTrainingUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateV2)

REG_OP(BNTrainingUpdateGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(diff_scale, TensorType({DT_FLOAT}))
    .OUTPUT(diff_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateGrad)

REG_OP(BNInferGrad)
    .INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(x_backprop, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BNInferGrad)

REG_OP(ReduceSum)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSum)

REG_OP(ReduceSumD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceSumD)

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension.

*@par Inputs:
*One input:
*x: A mutable Tensor. Must be one of the following types: float16,
* float32, double. Should be a Variable Tensor.

*@par Attributes:
*@li keep_dims: A bool. If true, retains reduced dimensions with length 1.
*@li axis: The dimensions to reduce. If None, reduces all dimensions.
*Must be in the range [- rank (input_sensor), rank (input_sensor)).

*@par Outputs:
*y: The reduced tensor.
*/
REG_OP(ReduceAllD)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAllD)

/**
*@brief Calculates the "logical sum" of elements of a tensor in a dimension.

*@par Inputs:
*Two inputs, including:
*@li x: A mutable Tensor. Must be one of the following types: float16, float32, double. Should be a Variable Tensor.
*@li axis: A mutable Tensor. The dimensions to reduce. If None, reduces all dimensions. Must be in the range [- rank (input_sensor), rank (input_sensor)).

*@par Attributes:
*keep_dims: A bool. If true, retains reduced dimensions with length 1.

*@par Outputs:
*y: The reduced tensor.
*/
REG_OP(ReduceAll)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAll)

/**
*@brief  Reduce a tensor on a certain axis based on product..

*@par Inputs:
*Two inputs, including:
*@li x: A mutable Tensor. Must be the type of NumberType.
*@li axis: A mutable Tensor. The dimensions to reduce.

*@par Attributes:
*@li keep_dims: A bool. If true, retains reduced dimensions with length 1. Defaults to "False".

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".
*/
REG_OP(ReduceProd)
    .INPUT(x,TensorType::NumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y,TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceProd)

/**
*@brief Computes the product of elements across dimensions of a tensor.

*@par Inputs:
* One input: \n
*x: A Tensor. Must be one of the following types: float16, float, int8, uint8.

*@par Attributes:
*@li axis: A required int8, int16, int32, or int64. Specifies the dimensions to reduce. No default value.
*@li keep_dims: An optional bool. If "True", retains reduced dimensions with length 1. Defaults to "False".

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".

*@attention Constraints:
* "keep_dims" is in the range [-rank(input_tensor), rank(input_tensor)].
*/
REG_OP(ReduceProdD)
    .INPUT(x,TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16}))
    .OUTPUT(y,TensorType({DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceProdD)

/**
*@brief Reduces "x" along the dimensions according to "axis".

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, int8, uint8.
* @li axis: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType.\n
*   - If None (the default), reduces all dimensions.\n
*   - Must be in the range [-rank(x), rank(x)).

*@par Attributes:
*keep_dims: A bool or NoneType. \n
* - If true, retains reduced dimensions with length 1. \n
* - If false, the rank of the tensor is reduced by 1 for each entry in axis.
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(ReduceMean)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMean)

/**
*@brief Reduces "x" along the dimensions according to "axis".

*@par Inputs:
*One input:
* @li x: A Tensor. Must be one of the following types: float16, float32, int8, uint8.

*@par Attributes:
*@li axis: The dimensions to reduce. Must be one of the following types: int, list, tuple, NoneType. \n
* If None (the default), reduces all dimensions. \n
* Must be in the range [-rank(x), rank(x)). \n
*@li keep_dims: A bool or NoneType. \n
* - If true, retains reduced dimensions with length 1. \n
* - If false, the rank of the tensor is reduced by 1 for each entry in axis.
*@par Outputs:
*y: A Tensor. Has the same type as "x".
*/
REG_OP(ReduceMeanD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_INT8, DT_UINT8}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMeanD)

REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMax)

/**
*@brief Returns the maximum of elements across dimensions of a Tensor.

*@par Inputs:
*x: A multi-dimensional Tensor of type float16, float32, or int16.

*@par Attributes:
* Two attributes, including: \n
*@li axis: A required listint, specifying the axis information of the index with the maximum value.
*@li keep_dims: A bool, specifying whether to keep dimensions for the output Tensor. Defaults to "false".

*@par Outputs:
*y: A multi-dimensional Tensor, specifying the maximum value of the corresponding axis in the tensor. Has the same type as "x". (If "keep_dims" is set to "false", the output dimensions are reduced by "dimension" compared with that of "x". Otherwise, the output has one fewer dimension than "x".)

*@attention Constraints:
* The value range of "axis" is [-dims, dims - 1]. "dims" indicates the dimension length of "x".
*/
REG_OP(ReduceMaxD)
    .INPUT(x, TensorType({DT_FLOAT, DT_UINT8, DT_INT8,
                          DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_UINT8, DT_INT8,
                           DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMaxD)

REG_OP(ReduceMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMin)

REG_OP(ReduceMinD)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT8,DT_UINT8}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMinD)
/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.\n
* Reduces `x` along the dimensions given in `axis`.
* Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
* entry in `axis`. If `keep_dims` is true, the reduced dimensions
* are retained with length 1.
*
* If `axis` is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:\n
* Only support bool
*
*@par Inputs:
*@li x : The boolean tensor to reduce.
*@li axis : The dimensions to reduce. If `None` (the default), reduces all
*          dimensions. Must be in the range `[-rank(x), rank(x))`.
*
*@par Attributes:
* keep_dims : If true, retains reduced dimensions with length 1.
*
*@par Outputs:
* y : The reduced tensor
*
*/
REG_OP(ReduceAny)
    .INPUT(x, TensorType({DT_BOOL}))
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAny)
/**
*@brief Computes the "logical or" of elements across dimensions of a tensor.\n
* Reduces `x` along the dimensions given in `axis`.
* Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
* entry in `axis`. If `keep_dims` is true, the reduced dimensions
* are retained with length 1.
*
* If `axis` is None, all dimensions are reduced, and a
* tensor with a single element is returned.
*
*@attention Constraints:\n
*  Only support bool
*
*@par Inputs:
* x : The boolean tensor to reduce.
*
*@par Attributes:
*@li axis : The dimensions to reduce. If `None` (the default), reduces all
*          dimensions. Must be in the range `[-rank(x), rank(x))`.
*@li keep_dims : If true, retains reduced dimensions with length 1.
*
*@par Outputs:
* y : The reduced tensor
*
*/
REG_OP(ReduceAnyD)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAnyD)

} //namespace ge


#endif /* GE_OP_REDUCE_OPS_H */
