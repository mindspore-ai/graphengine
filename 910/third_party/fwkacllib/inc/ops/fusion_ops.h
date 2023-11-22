/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file fusion_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Function FlashAttentionScore. \n

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, float32 .
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li real_shift: A matrix Tensor. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. The type support float16, bf16, float32.

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".
* @li inner_precise: A int. 0, fp16 high precision. 1, high performance.
*
* @par Outputs:
* softmax_max: A matrix Tensor. The type support float32.
* softmax_sum: A matrix Tensor. The type support float32.
* softmax_out: A matrix Tensor. The type support float16, bf16, float32.
* attention_out: A matrix Tensor. The type support float16, bf16, float32.
*/
REG_OP(FlashAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(real_shift, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT1, DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8, DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 1)
    .ATTR(sparse_mode, Int, 0)
    .OP_END_FACTORY_REG(FlashAttentionScore)

/**
* @brief Function FlashAttentionScoreGrad. \n

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, float32.
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li dy: A matrix Tensor. The type support float16, bf16, float32.
* @li real_shift: A scalar. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_in: A matrix Tensor. The type support float16, bf16, float32.
* @li attention_in: A matrix Tensor. The type support float16, bf16, float32.


* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".
* @li inner_precise: A int. 0, fp16 high precision. 1, high performance.

* @par Outputs:
* dq: A matrix Tensor. The type support float16, bf16, float32.
* dk: A matrix Tensor. The type support float16, bf16, float32.
* dv: A matrix Tensor. The type support float16, bf16, float32.
* dpse: A matrix Tensor. The type support float16, bf16, float32.
*/
REG_OP(FlashAttentionScoreGrad)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(dpse, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 65536)
    .ATTR(next_tockens, Int, 65536)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 1)
    .ATTR(sparse_mode, Int, 0)
    .OP_END_FACTORY_REG(FlashAttentionScoreGrad)

/**
* @brief Function IncreFlashAttention. \n

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, float32, int8.
* @li key: A matrix Tensor. The type support float16, bf16, float32, int8.
* @li value: A matrix Tensor. The type support float16, bf16, float32, int8.
* @li padding_mask: A matrix Tensor. The type support float16, float32.
* @li atten_mask: A matrix Tensor. The type support float16, bool, float32.
* @li actual_seq_lengths: A Tensor. The type support INT64.
* @li dequant_scale1: A Tensor. The type support INT64.
* @li quant_scale1: A Tensor. The type support float32.
* @li dequant_scale2: A Tensor. The type support INT64.
* @li quant_scale2: A Tensor. The type support float32.
* @li quant_offset2: A Tensor. The type support float32.

* @par Attributes:
* @li num_heads: A int. The number of the heads.
* @li scale_value: A float. The scale value. Default: 1.0.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD"]. Default: "BSH".
* @li num_key_value_heads: key value num heads.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, bf16, float32, int8. \n
*/
REG_OP(IncreFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_FLOAT32}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 1)
    .OP_END_FACTORY_REG(IncreFlashAttention)

/**
* @brief Function PromptFlashAttention. \n

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, float32 .
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li padding_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li actual_seq_lengths: A Tensor. The type support INT64.
* @li actual_seq_lengths_kv: A Tensor. The type support INT64.
* @li deq_scale1: A Tensor. The type support INT64.
* @li quant_scale1: A Tensor. The type support float32.
* @li deq_scale2: A Tensor. The type support INT64.
* @li quant_scale2: A Tensor. The type support float32.
* @li quant_offset2: A Tensor. The type support float32.

* @par Attributes:
* @li num_heads: A int. The number of the heads.
* @li scale_value: A float. The scale value. Default: 1.0.
* @li pre_tokens: A int. Previous tokens. Default: 214748647
* @li next_tokens: A int. Next tokens. Default: 0
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".
* @li num_key_value_heads: key value num heads. Default: 1
* @li sparse_mode: sparse mode. Default: 0

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, float32, int8. \n
*/
REG_OP(PromptFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_INT8}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_INT8}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(pre_tokens, Int, 214748647)
    .ATTR(next_tokens, Int, 0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .OP_END_FACTORY_REG(PromptFlashAttention)

/**
* @brief Fusion op for FFN.
* @par Inputs:
* @li x: A matrix Tensor. The type support int8, float16.
* @li weight1: A matrix Tensor. The type support int8, float16.
* @li weight2: A matrix Tensor. The type support int8, float16.
* @li expert_tokens: A matrix Tensor. The type support int64.
* @li bias1: A matrix Tensor. The type support int32, float16.
* @li bias2: A matrix Tensor. The type support int32, float16.
* @li scale: A matrix Tensor. The type support float32.
* @li offset: A matrix Tensor. The type support float32.
* @li deq_scale1: A matrix Tensor. The type support uint64.
* @li deq_scale2: A matrix Tensor. The type support uint64.

* @par Attributes:
* @li activation: A string. The type of activation.
* @li inner_precise: A int. 0, fp16 high precision. 1, high performance. Default value: 0
*
* @par Outputs:
* y: A matrix Tensor. The type support float16.
*/
REG_OP(FFN)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16}))
    .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16}))
    .INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16}))
    .OPTIONAL_INPUT(expert_tokens, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(activation, String)
    .ATTR(inner_precise, Int, 0)
    .OP_END_FACTORY_REG(FFN)

/**
* @brief Function AllGatherMatmul. \n

* @par Inputs:
* @li x1: A matrix Tensor. The type support float16, bf16.
* @li x2: A matrix Tensor. The type support float16, bf16.
* @li bias: A matrix Tensor. The type support float16, bf16. \n


* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: "false".
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: "false".
* @li gather_index: A int. Represents the input index for doing gather. Default: "0".
* @li comm_turn: A int. Number of communications with AICPU. Default: "0". \n

* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16.
* gatherOut: A matrix Tensor. The type support float16, bf16.
*/
REG_OP(AllGatherMatmul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(gatherOut, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(gather_index, Int, 0)
    .ATTR(comm_turn, Int, 0)
    .OP_END_FACTORY_REG(AllGatherMatmul)

/**
* @brief Function MatmulReduceScatter. \n

* @par Inputs:
* @li x1: A matrix Tensor. The type support float16, bf16.
* @li x2: A matrix Tensor. The type support float16, bf16.
* @li bias: A matrix Tensor. The type support float16, bf16. \n


* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
 perform.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: "false".
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: "false".
* @li comm_turn: A int. Number of communications with AICPU. Default: "0". \n

* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16.
*/
REG_OP(MatmulReduceScatter)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .OP_END_FACTORY_REG(MatmulReduceScatter)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_