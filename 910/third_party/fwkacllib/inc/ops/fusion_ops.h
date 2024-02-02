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
 * \file fusion_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fast and Memory-Efficient Exact Attention with IO-Awareness.

* @par Inputs:
* Eight inputs, including:
* @li query: A matrix Tensor. The type support float16, bf16.
* @li key: A matrix Tensor. The type support float16, bf16.
* @li value: A matrix Tensor. The type support float16, bf16.
* @li real_shift: A matrix Tensor. The type support float16, bf16.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. The type support bool, uint8.
* @li prefix: A matrix Tensor. The type support int64.

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH", "BNSD", "BSND"]. Default: "BSH".
* @li inner_precise: A int. 0, 1, reserved value. 2, support invalid lines.
* @li sparse_mode: A int. 0, defaultMsk. 1, allMask. 2, leftUpCasual. 3, rightDownCasual. 4, band. 5, prefix.
*
* @par Outputs:
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_out: A matrix Tensor. The type support float16, bf16.
* @li attention_out: A matrix Tensor. The type support float16, bf16.
*/
REG_OP(FlashAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(real_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
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
* @brief Function IncreFlashAttention.

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, float32, int8.
* @li key: A matrix Tensor. The type support float16, bf16, float32, int8.
* @li value: A matrix Tensor. The type support float16, bf16, float32, int8.
* @li pse_shift: A matrix Tensor. The type support float16, float32.
* @li atten_mask: A matrix Tensor. The type support float16, bool, float32.
* @li actual_seq_lengths: A Tensor. The type support INT64.
* @li dequant_scale1: A Tensor. The type support INT64.
* @li quant_scale1: A Tensor. The type support float32.
* @li dequant_scale2: A Tensor. The type support INT64.
* @li quant_scale2: A Tensor. The type support float32.
* @li quant_offset2: A Tensor. The type support float32.
* @li antiquant_scale: A Tensor. The type support float16.
* @li antiquant_offset: A Tensor. The type support float16.
* @li block_table: A Tensor. The type support int32.

* @par Attributes:
* @li num_heads: A int. The number of the heads.
* @li scale_value: A float. The scale value. Default: 1.0.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD"]. Default: "BSH".
* @li num_key_value_heads: key value num heads.
* @li block_size: A int. Max length in pageattention's kv block.
* @li inner_precise: A int. mode of precision in float16.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, bf16, float32, int8. \n
*/
REG_OP(IncreFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .DYNAMIC_INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_FLOAT32, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(IncreFlashAttention)

/**
* @brief Function PromptFlashAttention.

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, float32, bf16, int8.
* @li key: A matrix Tensor. The type support float16, float32, bf16, int8.
* @li value: A matrix Tensor. The type support float16, float32, bf16, int8.
* @li pse_shift: A matrix Tensor. The type support float16, float32, bf16.
* @li atten_mask: A matrix Tensor. The type support float16, bool, int8, uint8.
* @li actual_seq_lengths: A Tensor. The type support int64.
* @li actual_seq_lengths_kv: A Tensor. The type support int64.
* @li deq_scale1: A Tensor. The type support uint64.
* @li quant_scale1: A Tensor. The type support float32.
* @li deq_scale2: A Tensor. The type support uint64.
* @li quant_scale2: A Tensor. The type support float32.
* @li quant_offset2: A Tensor. The type support float32.

* @par Attributes:
* @li num_heads: A int. The number of the heads.
* @li scale_value: A float. The scale value. Default: 1.0.
* @li pre_tokens: A int. Previous tokens. Default: 214748647.
* @li next_tokens: A int. Next tokens. Default: 0.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD"]. Default: "BSH".
* @li num_key_value_heads: key value num heads. Default: 1.
* @li sparse_mode: sparse mode. Default: 0.
* @li inner_precise: A int. 0, float16 high precision. 1, high performance.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, float32, bf16, int8. \n
*/
REG_OP(PromptFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_INT8}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_INT8}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_INT8, DT_UINT8}))
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
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(PromptFlashAttention)


/**
* @brief Function FusedInferAttentionScore.

* @par Inputs:
* @li query: A matrix Tensor. The type support int8, float16, bf16.
* @li key: A matrix Tensor. The type support int8, float16, bf16.
* @li value: A matrix Tensor. The type support int8, float16, bf16.
* @li pse_shift: A matrix Tensor. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. The type support float16, bool, uint8, int8.
* @li actual_seq_lengths: A Tensor. The type support INT64.
* @li actual_seq_lengths_kv: A Tensor. The type support INT64.
* @li dequant_scale1: A Tensor. The type support UINT64.
* @li quant_scale1: A Tensor. The type support float32.
* @li dequant_scale2: A Tensor. The type support UINT64.
* @li quant_scale2: A Tensor. The type support float32.
* @li quant_offset2: A Tensor. The type support float32.
* @li antiquant_scale: A Tensor. The type support float16, bf16.
* @li antiquant_offset: A Tensor. The type support float16, bf16.
* @li block_table: An int.

* @par Attributes:
* @li num_heads: An int. The number of the heads.
* @li scale: A float. The scale value. Default: 1.0.
* @li pre_tokens: An int. Previous tokens. Default: 2147483647.
* @li next_tokens: An int. Next tokens. Default: 0.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".
* @li num_key_value_heads: key value num heads. Default: 0.
* @li sparse_mode: sparse mode. Default: 0.
* @li inner_precise: An int. 0, float16 high precision. 1, high performance. Default: 1.
* @li block_size: An int. Default: 0.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, float32, int8, bf16. \n
*/
REG_OP(FusedInferAttentionScore)
    .INPUT(query, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .DYNAMIC_INPUT(key, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .DYNAMIC_INPUT(value, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_UINT8, DT_INT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_BF16}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .ATTR(block_size, Int, 0)
    .OP_END_FACTORY_REG(FusedInferAttentionScore)


/**
* @brief Backwards calculation of FlashAttentionScore.

* @par Inputs:
* Thirteen inputs, including:
* @li query: A matrix Tensor. The type support float16, bf16.
* @li key: A matrix Tensor. The type support float16, bf16.
* @li value: A matrix Tensor. The type support float16, bf16.
* @li dy: A matrix Tensor. The type support float16, bf16.
* @li pse_shift: A scalar. The type support float16, bf16.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. The type support uint8, bool.
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_in: A matrix Tensor. The type support float16, bf16.
* @li attention_in: A matrix Tensor. The type support float16, bf16.
* @li prefix: A matrix Tensor. The type support int64.


* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".
* @li inner_precise: A int. 0, float16 high precision. 1, high performance.
* @li sparse_mode: A int. 0, defaultMsk. 1, allMask. 2, leftUpCasual. 3, rightDownCasual. 4, band. 5, prefix.

* @par Outputs:
* @li dq: A matrix Tensor. The type support float16, bf16.
* @li dk: A matrix Tensor. The type support float16, bf16.
* @li dv: A matrix Tensor. The type support float16, bf16.
* @li dpse: A matrix Tensor. The type support float16, bf16.
*/
REG_OP(FlashAttentionScoreGrad)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dpse, TensorType({DT_FLOAT16, DT_BF16}))
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
* @brief Fusion op for FFN.
* @par Inputs:
* fourteen inputs, including:
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
* @li antiquant_scale1: A matrix Tensor. The type support float16.
* @li antiquant_scale2: A matrix Tensor. The type support float16.
* @li antiquant_offset1: A matrix Tensor. The type support float16.
* @li antiquant_offset2: A matrix Tensor. The type support float16.

* @par Attributes:
* @li activation: A string. The type of activation.
* @li inner_precise: A int. 0, fp16 high precision. 1, high performance. Default value: 0
*
* @par Outputs:
* y: A matrix Tensor. The type support float16. \n
*/
REG_OP(FFN)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))
    .INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))
    .OPTIONAL_INPUT(expert_tokens, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(antiquant_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(activation, String)
    .ATTR(inner_precise, Int, 0)
    .OP_END_FACTORY_REG(FFN)


/**
* @brief Fusion op of allgather and matmul.
* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bfloat16.
* @li x2: A matrix Tensor. The type support float16, bfloat16.
* @li bias: A matrix Tensor. The type support float16, bfloat16. \n
*
* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: "false".
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: "false".
* @li gather_index: A int. Represents the input index for doing gather.
  Default: "0".
* @li comm_turn: A int. Number of communications with AICPU. Default: "0". \n
*
* @par Outputs:
* @li y: A matrix Tensor. The type support float16, bfloat16.
* @li gather_out: A matrix Tensor. The type support float16, bfloat16. \n
*/
REG_OP(AllGatherMatmul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(gather_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(gather_index, Int, 0)
    .ATTR(comm_turn, Int, 0)
    .OP_END_FACTORY_REG(AllGatherMatmul)

/**
* @brief Fusion op of matmul and reduce scatter.
* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bfloat16.
* @li x2: A matrix Tensor. The type support float16, bfloat16.
* @li bias: A matrix Tensor. The type support float16, bfloat16. \n
*
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
*
* @par Outputs:
* y: A matrix Tensor. The type support float16, bfloat16. \n
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

/**
* @brief Function MatmulAllReduce.

* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bf16.
* @li x2: A matrix Tensor. The type support float16, bf16.
* @li bias: A matrix Tensor. The type support float16, bf16. \n


* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
 perform. support "sum", "min", "max" ,"prod" .
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: false.
* @li comm_turn: A int. Number of communications with AICPU. Default: 0. \n

* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16.
*/
REG_OP(MatmulAllReduce)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .OP_END_FACTORY_REG(MatmulAllReduce)


/**
* @brief Function WeightQuantBatchMatmulV2. \n

* @par Inputs:
* @li x: A matrix Tensor. Shape supports (m,k)/(k,m), Format supports ND.
* @li weight: A matrix Tensor of quantized weight. Shape supports (n,k)/(k,n), Format supports ND.
* @li antiquant_scale: A Tensor for antiquant scale.
* Shape supports (1)/(1,n)/(n,1)/(ceil(k/antiquant_group_size),n)/(n,ceil(k/antiquant_group_size)),
* Format supports ND.
* @li antiquant_offset: A Tensor for antiquant offset. Shape and Format is same with antiquant_scale.
* @li quant_scale: A Tensor for quantization parameters. Shape supports (1)/(1,n), Format supports ND.
* @li quant_offset: A Tensor for quantization parameters. Shape and Format is same with quant_scale.
* @li bias: A Tensor. Shape supports (n)/(1,n), Format supports ND.\n

* @par Attributes:
* @li transpose_x: A bool. x is transposed if true.
* @li transpose_weight: A bool. weight is transposed if true.
* when transpose_weight is true, weight's shape is (n, k), antiquant_scale's shape should be (n, 1).
* @li antiquant_group_size: int, when weight's dtype is int8, antiquant_group_size can only be 0,
* weight's dtype is int4, antiquant_group_size must in [32, max(k-1, int_max_value)]
* and antiquant_group_size % 32 == 0. \n

* @par Outputs:
* y: A matrix Tensor.
*/
REG_OP(WeightQuantBatchMatmulV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_INT8, DT_INT4}))
    .INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT, DT_UINT64}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(antiquant_group_size, Int, 0)
    .OP_END_FACTORY_REG(WeightQuantBatchMatmulV2)


/**
* @brief Function GroupedMatmul. \n

* @par Inputs:
* @li x: A Tensor List.
* @li weight: A Tensor List of weight.
* @li bias: A Tensor List of bias.
* @li group_list: a Tensor.

* @par Attributes:
* @li split_item: A int.

* @par Outputs:
* y: A Tensor List.
*/
  REG_OP(GroupedMatmul)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(split_item, Int, 0)
    .OP_END_FACTORY_REG(GroupedMatmul)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
