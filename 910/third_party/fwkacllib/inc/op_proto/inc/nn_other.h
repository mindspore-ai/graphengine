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
 * \file nn_other.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Apply rotary position embedding.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r1: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r2: A tensor. Must be one of the following types: float16, float, bfloat16. 
 * @par Outputs:
 * y: A Tensor. Has the same shape as "x".
 */
REG_OP(RotaryMul)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(r1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(r2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryMul)

/**
 * @brief Calculate the inverse gradient of RotaryMul.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r1: A tensor. Must be one of the following types: float16, float, bfloat16.
 * r2: A tensor. Must be one of the following types: float16, float, bfloat16.
 * dy: A tensor. Data of grad increment.* 
 * @par Attributes:
 * @li need_backward: Optional. Control whether dr1 and dr2 need to be calculated. Defaults to "true".
 * @par Outputs:
 * dx: A Tensor. Has the same shape as "x".
 * dr1: A Tensor. Has the same shape as "r1".
 * dr2: A Tensor. Has the same shape as "r2".
 */
REG_OP(RotaryMulGrad)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(r1, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(r2, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .ATTR(need_backward, Bool, true)
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dr1, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OUTPUT(dr2, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16}))
    .OP_END_FACTORY_REG(RotaryMulGrad)

/**
* @brief init PartitionMap table. \n

* @par Inputs:
* @li ps_num: A Tensor, dtype is int32. 0-D. indicates ps number.
* @li ps_ids: A Tensor, dtype is int32. 1-D. indicates the id of ps. \n

* @par Attributes:
* @li partition_num: A Int, indicates the number of partition. \n
*/
REG_OP(InitPartitionMap)
    .INPUT(ps_num, TensorType({DT_INT32}))
    .INPUT(ps_ids, TensorType({DT_INT32}))
    .ATTR(partition_num, Int, 65537)
    .OP_END_FACTORY_REG(InitPartitionMap)

/**
* @brief uninit PartitionMap table. \n
*/
REG_OP(UninitPartitionMap)
    .OP_END_FACTORY_REG(UninitPartitionMap)

/**
* @brief init Embedding hashtable. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n

* @par Attributes:
* @li value_total_len: A Int, indicates the length of hashtable value. \n
* @li embedding_dim: A Int, indicates the length of embedding. \n
* @li bucket_size: A Int, Defaults to "0". \n
* @li dtype: A Type for data, Defaults to "DT_FLOAT". \n
* @li initializer_mode: A String of "random_uniform", "truncated_normal" , "constant" or "".
* indicates the algo of init method. Defaults to "".
* @li constant_value: A Float, used when initializer_mode is "constant", Defaults to "0". \n
* @li min: A Float, used when initializer_mode is "truncated_normal", the minimum value of the random number.
* Defaults to "-2".
* @li max: A Float, used when initializer_mode is "truncated_normal", the maximum value of the random number.
* Defaults to "2".
* @li mu: A Float, used when initializer_mode is "truncated_normal", The mean of the truncated_normal.
* Defaults to "0".
* @li sigma: A Float, used when initializer_mode is "truncated_normal", The variance of the truncated_normal.
* Defaults to "1".
* @li seed: A Int, Defaults to "0". \n
* @li seed2: A Int, Defaults to "0". \n
* @li filter_mode: A String of "no_filter" or "counter". indicates the type of the hashmap, Defaults to "no_filter". \n
* @li optimizer_mode: A String of "adam" or "adamw" or "adagrad" or "sgd" or "rmsprop". indicates the type of the optimizer_mode,
* Defaults to "".
* @li optimizer_params: Float list, when optimizer_mode is "adagrad", the initialize value of the optimizer. \n
*/
REG_OP(InitEmbeddingHashmap)
    .INPUT(table_id, TensorType({DT_INT32}))
    .ATTR(bucket_size, Int, 0)
    .REQUIRED_ATTR(value_total_len, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(dtype, Type, DT_FLOAT)
    .ATTR(initializer_mode, String, "")
    .ATTR(constant_value, Float, 0)
    .ATTR(min, Float, -2)
    .ATTR(max, Float, 2)
    .ATTR(mu, Float, 0)
    .ATTR(sigma, Float, 1)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(filter_mode, String, "no_filter")
    .ATTR(optimizer_mode, String, "")
    .ATTR(optimizer_params, ListFloat, {})
    .OP_END_FACTORY_REG(InitEmbeddingHashmap)

/**
* @brief embedding hsahtable data import. \n

* @par Inputs:
* @li file_path: A Tensor, dtype is string. 0-D. indicates embedding filepath.
* @li ps_id: A Tensor, dtype is int32. 0-D. indicates the id of ps.
* @li table_id: A Tensor, dtype is int32. 1-D. indicates the id of hashtable. \n

* @par Attributes:
* @li embedding_dim: A ListInt. indicates the hashtable value number.
* @li value_total_length: A ListInt. indicates the hashtable total length, inclue m+v or accum.
* @li only_var: A Bool. only import var.
* @li file_type: A String. indicates the import file .
* @li table_name: A List String. represents table name corresponding to table id . \n
*/
REG_OP(EmbeddingTableImport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .REQUIRED_ATTR(value_total_len, ListInt)
    .ATTR(only_var_flag, Bool, false)
    .ATTR(file_type, String, "bin")
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingTableImport)

/**
* @brief embedding hsahtable data lookup. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is uint32. 1-D. indicates the hashtable key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: A Int. indicates the hashtable value number.
* @li default_value: A Float, indicate the default value when can not find key. \n
*/
REG_OP(EmbeddingTableFind)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(default_value, Float, -1)
    .OP_END_FACTORY_REG(EmbeddingTableFind)

/**
* @brief uninit embedding hsahtable. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable. \n
*/
REG_OP(UninitEmbeddingHashmap)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(UninitEmbeddingHashmap)

/**
* @brief embedding hashtable lookup or init. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is DT_INT32. 0-D. indicates the id of hashtable.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key. \n

* @par Outputs:
* @li values: indicates the hashtable value. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding var value in hashtable.
* @li value_total_len: A Int, indicates the dim of embedding var+m+v or var+accum values in hashtable
* @li initializer_mode: A String of "random_uniform", "truncated_normal" or "constant".
* indicates the algo of init method, Defaults to "random_uniform".
* @li constant_value: A Float, used when initializer_mode is "constant", Defaults to "0".
* @li min: A Float, used when initializer_mode is "truncated_normal", the minimum value of the random number.
* Defaults to "-2".
* @li max: A Float, used when initializer_mode is "truncated_normal", the maximum value of the random number.
* Defaults to "2".
* @li mu: A Float, used when initializer_mode is "truncated_normal", The mean of the truncated_normal.
* Defaults to "0".
* @li sigma: A Float, used when initializer_mode is "truncated_normal", The variance of the truncated_normal.
* Defaults to "1".
* @li seed: An Int, Used to create a random seed, Defaults to "0".
* @li seed2: An Int, Used to create a random seed, Defaults to "0".
* @li filter_mode: A String of "no_filter" or "counter". indicates the type of the hashmap, Defaults to "no_filter".
* @li filter_freq: An Int, Used to set the threshold of the tal, Defaults to "0".
* @li default_key_or_value: A bool, indicates the default value get way.
* @li default_key: An Int, when default_key_or_value is true, use the default_key corresponding value as default value.
* @li default_value: An Int, when default_key_or_value is false, use the default_value as default value.
* @li optimizer_mode: A String of "adam" or "adamw" or "adagrad" or "sgd" or "rmsprop". indicates the type of the optimizer_mode,
* Defaults to "".
* @li optimizer_params: Float list, when optimizer_mode is "adagrad", the initialize value of the optimizer. \n
*/
REG_OP(EmbeddingTableFindAndInit)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .REQUIRED_ATTR(value_total_len, Int)
    .ATTR(initializer_mode, String, "random_uniform")
    .ATTR(constant_value, Float, 0)
    .ATTR(min, Float, -2)
    .ATTR(max, Float, 2)
    .ATTR(mu, Float, 0)
    .ATTR(sigma, Float, 1)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(filter_mode, String, "no_filter")
    .ATTR(filter_freq, Int, 0)
    .ATTR(default_key_or_value, Bool, false)
    .ATTR(default_key, Int, 0)
    .ATTR(default_value, Float, 0)
    .ATTR(optimizer_mode, String, "")
    .ATTR(optimizer_params, ListFloat, {})
    .OP_END_FACTORY_REG(EmbeddingTableFindAndInit)

/**
* @brief embedding hashtable embedding applyadam. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li beta1_power: A Scalar, dtype is DT_FLOAT16 or DT_FLOAT. 0-D. indicates the beta1's power.
* @li beta2_power: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2's power.
* @li lr: A Scalar, dtype is same as "beta1_power". 0-D. indicates the learning rate.
* @li beta1: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta1 param.
* @li beta2: A Scalar, dtype is same as "beta1_power". 0-D. indicates the beta2 param.
* @li epsilon: A Scalar, dtype is same as "beta1_power". 0-D. indicates the small value param.
* @li grad: A Tensor, dtype is same as "beta1_power". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable.
* @li mask_zero: An optional bool, whether to perform no-update interception when key==0. \n
*/
REG_OP(EmbeddingApplyAdam)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(beta1_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(mask_zero, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingApplyAdam)

/**
* @brief embedding hashtable embedding applyadamW. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li beta1_power: A Tensor, dtype is float16 or float. 0-D. indicates the beta1's power.
* @li beta2_power: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta2's power.
* @li lr: A Tensor, dtype is same as "beta1_power". 0-D. indicates the learning rate.
* @li weight_decay: A Tensor, dtype is same as "beta1_power". 0-D. indicates the weight decay.
* @li beta1: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta1 param.
* @li beta2: A Tensor, dtype is same as "beta1_power". 0-D. indicates the beta2 param.
* @li epsilon: A Tensor, dtype is same as "beta1_power". 0-D. indicates the small value param.
* @li grad: A Tensor, dtype is same as "beta1_power". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is int64. 1-D. indicates the hashtable key.
* @li max_grad_norm: A mutable Tensor of the same type as "beta1_power", an optional input. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable.
* @li amsgrad: An optional bool, indicates whether to use the AMSGrad variant of htis algorithm from
*     the paper On the Convergence of Adam and Beyond.
*     If "True", max_grad_norm input and output must be entered.
* @li maximize: An optional bool, maximize the params based on the objective.
* @li mask_zero: An optional bool, whether to perform no-update interception when key==0. \n
*/
REG_OP(EmbeddingApplyAdamW)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(beta1_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2_power, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight_decay, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(beta2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(epsilon, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(max_grad_norm, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .ATTR(mask_zero, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingApplyAdamW)

/**
* @brief embedding hashtable resource applyadagrad. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is the same as "grad". 0-D. indicates the learning rate.
* @li grad: A Tensor, dtype is the same as "grad". 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.
* @li global_step: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the train step. \n

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable.
* @li mask_zero: An optional bool, whether to perform no-update interception when key==0. \n
*/
REG_OP(EmbeddingApplyAdaGrad)
    .INPUT(var_handle, TensorType({DT_RESOURCE}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(global_step, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(mask_zero, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingApplyAdaGrad)

/**
* @brief embedding hashtable resource apply sgd. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is the same as "grad". 0-D. indicates the learning rate.
* @li grad: A Tensor, dtype is DT_FLOAT/DT_FLOAT16. 1-D. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable.
* @li mask_zero: An Optional Bool, whether to perfomr non-update interception when key==0. \n
*/
REG_OP(EmbeddingApplySgd)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(mask_zero, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingApplySgd)

/**
* @brief embedding hashtable resource apply rmsprop. \n

* @par Inputs:
* @li var_handle: The handle of embedding hashtable.
* @li lr: A Scalar, dtype is the same as "grad". indicates the learning rate.
* @li rho: A Scalar, dtype is the same as "grad". indicates the decay rate.
* @li momentum: A Scalar, dtype is the same as "grad". indicates the momentum.
* @li epsilon: A Scalar, dtype is the same as "grad". indicates the small value param.
* @li grad: A Tensor, dtype is NumberType. indicates the grad.
* @li keys: A Tensor, dtype is DT_INT64. 1-D. indicates the hashtable key.

* @par Outputs:
* @li var_handle: The handle of embedding hashtable. \n

* @par Attributes:
* @li embedding_dim: A Int, indicates the dim of embedding value in hashtable.
* @li mask_zero: An Optional Bool, whether to perfomr non-update interception when key==0. \n
*/
REG_OP(EmbeddingApplyRmsprop)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(lr, TensorType::NumberType())
    .INPUT(rho, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(mask_zero, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingApplyRmsprop)

/**
* @brief Exponential decay algorithm. \n

* @par Inputs:
* @li initial_learning_rate: A Scalar, dtype is DT_FLOAT/DT_FLOAT16. 0-D. indicates the learning rate.
* @li decay_rate: A Scalar, dtype is  the same as lr. 0-D. indicates the decay rate.
* @li decay_steps: A Scalar, dtype is DT_INT32/DT_INT64. 0-D. indicates the decay steps. \n

* @par Outputs:
* @li decayed_lr: Indicates the learning rate after updating. \n

* @par Attributes:
* @li staircase: A Scalar, dtype is DT_BOOL. 0-D. indicates the stratergy for updating lr.
* True indicates updating lr according to the decay_steps. False indicates updating lr each step. \n
*/
REG_OP(ExponentialDecayLR)
    .INPUT(var_handle, TensorType({DT_RESOURCE, DT_INT32}))
    .INPUT(initial_learning_rate, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(decay_rate, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(decay_steps, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(decayed_lr, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(staircase, Bool, false)
    .OP_END_FACTORY_REG(ExponentialDecayLR)

/**
* @brief embedding hashtable export. \n

* @par Inputs:
* @li file_path: A String, indicates the export file path.
* @li ps_id: A Int, dtype is DT_INT32, indicates the ps server id.
* @li table_id: A Tensor, 1D, dtype is DT_INT32, indicates the hashtable id.

* @par Attributes:
* @li embedding_dim: A ListInt. indicates the hashtable value number.
* @li value_total_length: A ListInt. indicates the hashtable total length, inclue m+v or accum.
* @li export_mode: A String. export mode, Defaults to "all".
* @li only_var: A Bool. only export var, Defaults to "false".
* @li file_type: A String. indicates the export file, Defaults to "bin".
* @li table_name: A List String. represents table name corresponding to table id .
* @li filter_export_flag: A Bool. represents filter export flag on counter filter scenario. \n
*/
REG_OP(EmbeddingTableExport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .REQUIRED_ATTR(embedding_dim, ListInt)
    .REQUIRED_ATTR(value_total_len, ListInt)
    .ATTR(export_mode, String, "all")
    .ATTR(only_var_flag, Bool, false)
    .ATTR(file_type, String, "bin")
    .ATTR(table_name, ListString, {})
    .ATTR(filter_export_flag, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingTableExport)

/**
* @brief embedding tableid trans to resource. \n

* @par Inputs:
* @li table_id: A Tensor, dtype is int32. 0-D. indicates the id of hashtable.

* @par Outputs:
* @li table_handle: indicates the resource_handle of tableid. \n
*/
REG_OP(TableToResource)
    .INPUT(table_id, TensorType({DT_INT32}))
    .OUTPUT(table_handle, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(TableToResource)

/**
* @brief embedding feature_id trans to offset_id. \n

* @par Inputs:
* @li feature_id: A Tensor, dtype is int64.

* @par Outputs:
* @li offset_id: A Tensor with same shape of feature_id, dtype is int32. \n
*/
REG_OP(EmbeddingFeatureMapping)
    .INPUT(feature_id, TensorType({DT_INT64}))
    .OUTPUT(offset_id, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(EmbeddingFeatureMapping)

/**
* @brief embedding compute var export. \n

* @par Inputs:
* @li file_path: A String, indicates the export file path.
* @li ps_id: A Int, dtype is int32, indicates the ps server id.
* @li table_id: A Int, dtype is int32, indicates the hashtable id. \n

* @par Attributes:
* @li table_name: A List String. represents table name corresponding to table id . \n
*/
REG_OP(EmbeddingComputeVarExport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingComputeVarExport)

/**
* @brief embedding compute var import. \n

* @par Inputs:
* @li file_path: A String, indicates the import file path.
* @li ps_id: A Int, dtype is int32, indicates the ps server id.
* @li table_id: A Int, dtype is int32, indicates the hashtable id.

* @par Attributes:
* @li table_name: A List String. represents table name corresponding to table id . \n
*/
REG_OP(EmbeddingComputeVarImport)
    .INPUT(file_path, TensorType({DT_STRING}))
    .INPUT(ps_id, TensorType({DT_INT32}))
    .INPUT(table_id, TensorType({DT_INT32}))
    .ATTR(table_name, ListString, {})
    .OP_END_FACTORY_REG(EmbeddingComputeVarImport)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
