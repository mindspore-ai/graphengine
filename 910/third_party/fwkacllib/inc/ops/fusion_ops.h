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
* @li query: A matrix Tensor. The type support float16, bf16, float32.
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li real_shift: A matrix Tensor. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. The type support bool, uint8.
* @li prefix: A matrix Tensor. The type support int64.
* @li actual_seq_qlen: A matrix Tensor. The type support int64. If used, layout need to be setted TND. ex. If the q seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10]
* @li actual_seq_kvlen: A matrix Tensor. The type support int64. If used, layout need to be setted TND. ex. If the kv seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10]

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH", "BNSD", "BSND", "TND"]. Default: "BSH".
* @li inner_precise: A int. 0, 1, reserved value. 2, support invalid lines.
* @li sparse_mode: A int. 0, defaultMsk. 1, allMask. 2, leftUpCasual. 3, rightDownCasual. 4, band. 5, prefix.
*
* @par Outputs:
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_out: A matrix Tensor. The type support float16, bf16, float32.
* @li attention_out: A matrix Tensor. The type support float16, bf16, float32.
*/
REG_OP(FlashAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(real_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .OP_END_FACTORY_REG(FlashAttentionScore)

/**
* @brief Function IncreFlashAttention.

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, int8.
* @li key: A matrix Tensor. The type support float16, bf16, int8.
* @li value: A matrix Tensor. The type support float16, bf16, int8.
* @li pse_shift: A matrix Tensor. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. The type support bool, int8, uint8.
* @li actual_seq_lengths: A Tensor. The type support INT64.
* @li dequant_scale1: A Tensor. The type support uint64, float32.
* @li quant_scale1: A Tensor. The type support float32.
* @li dequant_scale2: A Tensor. The type support uint64, float32.
* @li quant_scale2: A Tensor. The type support float32, bf16.
* @li quant_offset2: A Tensor. The type support float32, bf16.
* @li antiquant_scale: A Tensor. The type support float16, bf16.
* @li antiquant_offset: A Tensor. The type support float16, bf16.
* @li block_table: A Tensor. The type support int32.
* @li kv_padding_size: A Tensor. The type support int64.

* @par Attributes:
* @li num_heads: A int. The number of the heads.
* @li scale_value: A float. The scale value. Default: 1.0.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD", "BSND"]. Default: "BSH".
* @li num_key_value_heads: key value num heads.
* @li block_size: A int. Max length in pageattention's kv block.
* @li inner_precise: A int. mode of precision in float16.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, bf16, int8. \n
*/
REG_OP(IncreFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(IncreFlashAttention)

/**
* @brief Compute the GeGluV2,
* where the activations function in GLU is Gelu.

* @par Inputs:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32. \n

* @par Outputs:
* two outputs, including:
* @li y: A Tensor. Must be one of the following types: bfloat16, float16, float32.
* @li gelu: A Tensor. Must be one of the following types: bfloat16, float16, float32. \n

* @par Attributes:
* two attributes, including:
* @li dim: A optional int. The dimension to be split, default is -1.
* @li approximate: A optional int.
* The gelu approximation algorithm to use: 'none'(0) or 'tanh'(1), default is 'tanh'(1).
* @li activate_left: A optional bool.
* The gelu activate_left algorithm to use: 
*     'false'(activate right) or 'true'(activate left), defalut is 'false'(activate right). \n

* @par Third-party framework compatibility:
* New operator GeGluV2.
*/
REG_OP(GeGluV2)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .OUTPUT(gelu, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(approximate, Int, 1)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(GeGluV2)

/**
* @brief Computes the gradient for the GeGluV2 of "x" .
*
* @par Inputs:
* Three inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, bfloat16, float32.
* @li x: A Tensor of the same type as "dy".
* @li gelu: A Tensor of the same type as "dy".
*
* @par Outputs:
* @li dx: A Tensor. Has the same type as "dy".
*
* @par Attributes:
* @li dim: A optional Int.  default is -1.
* @li approximate: A optional Int.
* The gelu grad approximation algorithm to use: 0 or 1, default is 1('tanh').
* @li activate_left: A optional Bool.
* Whether the left side of x is used as an input parameter to the activation function,
* default is false, use the right side. \n
*
* @par Third-party framework compatibility
* Compatible with the Pytorch operator GeGluGradV2.
*
*/
REG_OP(GeGluGradV2)
    .INPUT(dy, "T")
    .INPUT(x, "T")
    .INPUT(gelu, "T")
    .OUTPUT(dx, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(approximate, Int, 1)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(GeGluGradV2)

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
* @li deq_scale1: A Tensor. The type support uint64,float32.
* @li quant_scale1: A Tensor. The type support float32.
* @li deq_scale2: A Tensor. The type support uint64,float32.
* @li quant_scale2: A Tensor. The type support float32.
* @li quant_offset2: A Tensor. The type support float32.

* @par Attributes:
* @li num_heads: A int. The number of the heads.
* @li scale_value: A float. The scale value. Default: 1.0.
* @li pre_tokens: A int. Previous tokens. Default: 214748647.
* @li next_tokens: A int. Next tokens. Default: 0.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD", "BSND", "NSD", "SH"]. Default: "BSH".
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
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
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
* @li key_antiquant_scale: A Tensor. The type support float16, bf16, float32.
* @li key_antiquant_offset: A Tensor. The type support float16, bf16, float32.
* @li value_antiquant_scale: A Tensor. The type support float16, bf16, float32.
* @li value_antiquant_offset: A Tensor. The type support float16, bf16, float32.
* @li key_shared_prefix: A matrix Tensor. The type support int8, float16, bf16.
* @li value_shared_prefix: A matrix Tensor. The type support int8, float16, bf16.
* @li actual_shared_prefix_len: A Tensor. The type support INT64.

* @par Attributes:
* @li num_heads: An int. The number of the heads.
* @li scale: A float. The scale value. Default: 1.0.
* @li pre_tokens: An int. Previous tokens. Default: 2147483647.
* @li next_tokens: An int. Next tokens. Default: 2147483647.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD", "BSND", "NSD", "SH"]. Default: "BSH".
* @li num_key_value_heads: key value num heads. Default: 0.
* @li sparse_mode: sparse mode. Default: 0.
* @li inner_precise: An int. 0, float16 high precision. 1, high performance. Default: 1.
* @li block_size: An int. Default: 0.
* @li antiquant_mode: An int. Default: 0.
* @li softmax_lse_flag: An bool. Default: false.
* @li key_antiquant_mode: An int. Default: 0.
* @li value_antiquant_mode: An int. Default: 0.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, float32, int8, bf16. \n
* softmax_lse: A matrix Tensor. The type support float32. \n
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
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(key_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_shared_prefix, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .OPTIONAL_INPUT(value_shared_prefix, TensorType({DT_INT8, DT_FLOAT16,DT_BF16}))
    .OPTIONAL_INPUT(actual_shared_prefix_len, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_BF16}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT32}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(antiquant_mode, Int, 0)
    .ATTR(softmax_lse_flag, Bool, false)
    .ATTR(key_antiquant_mode, Int, 0)
    .ATTR(value_antiquant_mode, Int, 0)
    .OP_END_FACTORY_REG(FusedInferAttentionScore)


/**
* @brief Backwards calculation of FlashAttentionScore.

* @par Inputs:
* Thirteen inputs, including:
* @li query: A matrix Tensor. The type support float16, bf16, float32.
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li dy: A matrix Tensor. The type support float16, bf16, float32.
* @li pse_shift: A scalar. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. The type support uint8, bool.
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_in: A matrix Tensor. The type support float16, bf16, float32.
* @li attention_in: A matrix Tensor. The type support float16, bf16, float32.
* @li prefix: A matrix Tensor. The type support int64.
* @li actual_seq_qlen: A matrix Tensor. The type support int64. If used, layout need to be setted TND. ex. If the q seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10]
* @li actual_seq_kvlen: A matrix Tensor. The type support int64. If used, layout need to be setted TND. ex. If the kv seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10]

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH", "BNSD", "BSND", "TND"]. Default: "BSH".
* @li inner_precise: A int. 0, 1, reserved value. 2, support invalid lines.
* @li sparse_mode: A int. 0, defaultMask. 1, allMask. 2, leftUpCasual. 3, rightDownCasual. 4, band. 5, prefix.

* @par Outputs:
* @li dq: A matrix Tensor. The type support float16, bf16, float32.
* @li dk: A matrix Tensor. The type support float16, bf16, float32.
* @li dv: A matrix Tensor. The type support float16, bf16, float32.
* @li dpse: A matrix Tensor. The type support float16, bf16, float32.
*/
REG_OP(FlashAttentionScoreGrad)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dpse, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 65536)
    .ATTR(next_tockens, Int, 65536)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .OP_END_FACTORY_REG(FlashAttentionScoreGrad)


/**
* @brief Fusion op for FFN.
* @par Inputs:
* fourteen inputs, including:
* @li x: A matrix Tensor. The type support int8, float16, bfloat16.
* @li weight1: A matrix Tensor. The type support int4, int8, float16, bfloat16.
* @li weight2: A matrix Tensor. The type support int4, int8, float16, bfloat16.
* @li expert_tokens: A matrix Tensor. The type support int64.
* @li bias1: A matrix Tensor. The type support int32, float16, float32.
* @li bias2: A matrix Tensor. The type support int32, float16, float32.
* @li scale: A matrix Tensor. The type support float32.
* @li offset: A matrix Tensor. The type support float32.
* @li deq_scale1: A matrix Tensor. The type support uint64, int64, float32, bfloat16.
* @li deq_scale2: A matrix Tensor. The type support uint64, int64, float32, bfloat16.
* @li antiquant_scale1: A matrix Tensor. The type support float16, bfloat16.
* @li antiquant_scale2: A matrix Tensor. The type support float16, bfloat16.
* @li antiquant_offset1: A matrix Tensor. The type support float16, bfloat16.
* @li antiquant_offset2: A matrix Tensor. The type support float16, bfloat16.

* @par Attributes:
* @li activation: A string. The type of activation.
* @li inner_precise: A int. 0, fp16 high precision. 1, high performance. Default value: 0
* @li output_dtype: A int. -1, output data type is float16. 0, quant and output data type is float16. 1, quant and output data type is bfloat16. Default -1.
* @li tokens_index_flag: A bool. false, values in expert tokens are values. true, values therein are indices. Default value: false
*
* @par Outputs:
* y: A matrix Tensor. The type support float16, bfloat16. \n
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
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_BF16, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_BF16, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(activation, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(output_dtype, Int, -1)
    .ATTR(tokens_index_flag, Bool, false)
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
* @li comm_turn: A int. Number of communications with AICPU. Default: "0".
* @li rank_size: A int. Number of rank num. Default: "0".
* @li is_gather_out: A bool. If True, output gather_out matrix. Default: "true". \n
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
    .ATTR(rank_size, Int, 0)
    .ATTR(is_gather_out, Bool, true)
    .OP_END_FACTORY_REG(AllGatherMatmul)

/**
* @brief Combine similar tokens using the matching algorithm.
* @par Inputs:
* @li token_a: A Tensor. Type is:DT_FLOAT16. Shape is (B, S1, H).
* @li token_b: A Tensor. Type is:DT_FLOAT16. Shape is (B, S2, H).
* @li topk_indice: A Tensor. Type is:DT_INT64. Shape is (B, S1, H), S1 must equal with token_a. Value range is [0, S1), no dup.
* @li arg_max: A Tensor. Type is:DT_INT64. Shape is (B, S1, H), S1 must equal with token_a. Value range is [0, S2), can dup.
* @par Outputs:
* @li unmerge_token_a: A Tensor. Type is:DT_FLOAT16.
* @li unmerge_token_b: A Tensor. Type is:DT_FLOAT16.
* @li unreduce_count: A Tensor. Type is:DT_FLOAT.
* @par Attributes:
* @li top_rate: Type is:Float. rate to calculate how many rows of token_a merge to token_b. default is "0.5".
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TomeMerge)
    .INPUT(token_a, TensorType({DT_FLOAT16}))
    .INPUT(token_b, TensorType({DT_FLOAT16}))
    .INPUT(topk_indice, TensorType({DT_INT64}))
    .INPUT(arg_max, TensorType({DT_INT64}))
    .OUTPUT(unmerge_token_a, TensorType({DT_FLOAT16}))
    .OUTPUT(unreduce_token_b, TensorType({DT_FLOAT16}))
    .OUTPUT(unreduce_count, TensorType({DT_FLOAT}))
    .ATTR(top_rate, Float, 0.5)
    .OP_END_FACTORY_REG(TomeMerge)

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
    .ATTR(rank_size, Int, 0)
    .OP_END_FACTORY_REG(MatmulReduceScatter)

/**
* @brief Function MatmulAllReduce.

* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bf16, int8.
* @li x2: A matrix Tensor. The type support float16, bf16, int8, int4.
* @li bias: A matrix Tensor. The type support float16, bf16, int32.
* @li x3: A matrix Tensor. The type support float16, bf16.
* @li antiquant_scale: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. The type support float16, bf16.
* @li dequant_scale: A matrix Tensor. The type support float16, bf16, uint64, int64, float32.
* @li pertoken_scale: A matrix Tensor. The type support float32. \n


* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
 perform. support "sum", "min", "max" ,"prod" .
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: false.
* @li comm_turn: A int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: A int. Number of per-group for quant. Default: 0. \n

* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16.
*/
REG_OP(MatmulAllReduce)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .OP_END_FACTORY_REG(MatmulAllReduce)

/**
* @brief Function MatmulAllReduceAddRmsNorm.

* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bf16, int8.
* @li x2: A matrix Tensor. The type support float16, bf16, int8, int4.
* @li bias: A matrix Tensor. The type support float16, bf16, int32.
* @li residual: A matrix Tensor. The type support float16, bf16.
* @li gamma: A matrix Tensor. The type support float16, bf16.
* @li antiquant_scale: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. The type support float16, bf16.
* @li dequant_scale: A matrix Tensor. The type support float16, bf16, uint64. \n


* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
 perform. support "sum", "min", "max" ,"prod" .
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: false.
* @li comm_turn: A int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: A int. Number of per-group for quant. Default: 0. \n
* @li epsilon: A float32. Default: 1e-06. \n

* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16.
* norm_out: A matrix Tensor. The type support float16, bf16.
*/
REG_OP(MatmulAllReduceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(residual, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(norm_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(epsilon, Float, 1e-06)
    .OP_END_FACTORY_REG(MatmulAllReduceAddRmsNorm)

/**
* @brief Function InplaceMatmulAllReduceAddRmsNorm.

* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bf16, int8.
* @li x2: A matrix Tensor. The type support float16, bf16, int8, int4.
* @li bias: A matrix Tensor. The type support float16, bf16, int32.
* @li residual: A matrix Tensor. The type support float16, bf16.
* @li gamma: A matrix Tensor. The type support float16, bf16.
* @li antiquant_scale: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. The type support float16, bf16.
* @li dequant_scale: A matrix Tensor. The type support float16, bf16, uint64. \n


* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
 perform. support "sum", "min", "max" ,"prod" .
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication. Default: false.
* @li comm_turn: A int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: A int. Number of per-group for quant. Default: 0. \n
* @li epsilon: A float32. Default: 1e-06. \n

* @par Outputs:
* residual: A matrix Tensor. The type support float16, bf16.
* norm_out: A matrix Tensor. The type support float16, bf16.
*/
REG_OP(InplaceMatmulAllReduceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(residual, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64}))
    .OUTPUT(residual, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(norm_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(epsilon, Float, 1e-06)
    .OP_END_FACTORY_REG(InplaceMatmulAllReduceAddRmsNorm)

/**
* @brief matmul layer norm reduce.
*
* @par Inputs:
* @li x1: A Tensor. Must be one of the following types: float16.
* @li x2: A Tensor. Must be one of the following types: float16.
* @li bias: A Tensor. Must be one of the following types: float16.
*
* @par Outputs:
* y: A Tensor. Must be one of the following types: float16.
* sum: A Tensor. Must be one of the following types: float16.
* square_sum: A Tensor. Must be one of the following types: float16.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(MatmulLayerNormReduce)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16}))
    .OUTPUT(x2, TensorType({DT_FLOAT16}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(MatmulLayerNormReduce)


/**
* @brief Function AGLU. \n

* @par Inputs:
* four inputs, including:
* @li x: A required matrix Tensor. The type support float16.
* @li weight1: A required matrix Tensor. The type support float16.
* @li bias1: A optional matrix Tensor. The type support float16.
* @li weight2: A optional matrix Tensor. The type support float16.
* @li bias2: A optional matrix Tensor. The type support float16.

* @par Attributes:
* @li activate_func: A required string. The type of activation.
* @li activate_left: A optional bool. Default: false.

* @par Outputs:
* y: A matrix Tensor. The type support float16. \n
*/
REG_OP(AGLU)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(weight1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(weight2, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(activate_func, String)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(AGLU)


/**
* @brief Function WeightQuantBatchMatmulV2. \n

* @par Inputs:
* @li x: A matrix Tensor. Shape supports (m,k)/(k,m), Format supports ND. The type support float16, bf16.
* @li weight: A matrix Tensor of quantized weight. Shape supports (n,k)/(k,n), Format supports ND/NZ.
* The type support int8, int4.
* @li antiquant_scale: A Tensor for antiquant scale.
* Shape supports (1)/(1,n)/(n,1)/(ceil(k/antiquant_group_size),n)/(n,ceil(k/antiquant_group_size)),
* Format supports ND. The type support float16, bf16.
* @li antiquant_offset: A Tensor for antiquant offset. Shape and Format is same with antiquant_scale.
* The type support float16, bf16.
* @li quant_scale: A Tensor for quantization parameters. Shape supports (1)/(1,n), Format supports ND.
* The type support float32, uint64.
* @li quant_offset: A Tensor for quantization parameters. Shape and Format is same with quant_scale.
* The type support float32.
* @li bias: A Tensor. Shape supports (n)/(1,n), Format supports ND.
* Specifically, these optional inputs support the shape (0,). At this point,
* it means that the optional input doesn't exist. The type support float16, float32. \n

* @par Attributes:
* @li transpose_x: A bool. x is transposed if true. Default: false.
* @li transpose_weight: A bool. weight is transposed if true. Default: false.
* when transpose_weight is true, weight's shape is (n, k),
* antiquant_scale's shape should be (1)/(n,1)/(n,ceil(k/antiquant_group_size)).
* @li antiquant_group_size: int, antiquant_group_size must in [0, k-1] and antiquant_group_size % 32 == 0.
* When the antiquant_group_size is 0, it means that the per-group mode is not used. Default: 0. \n

* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16, int8.
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
* @li scale: A Tensor List of scale.
* @li offset: A Tensor List of offset.
* @li antiquant_scale: A Tensor List of antiquant_scale.
* @li antiquant_offset: A Tensor List of antiquant_offset.
* @li group_list: a Tensor.
* @li per_token_scale: A Tensor of per_token_scale.

* @par Attributes:
* @li split_item: A int.
* @li dtype: A int. only invalid for quant case. -1, output data type is int8. 0, not supported. 1, output data type is bfloat16. Default -1.
* @li transpose_weight: A bool. Reserved parameter, indicate wether input weight is transposed, not enabled.
* @li transpose_x: A bool. Reserved parameter, indicate wether input x is transposed, not enabled.
* @li group_type: A int. Indicates the splited dimension.

* @par Outputs:
* y: A Tensor List.
*/
  REG_OP(GroupedMatmul)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .DYNAMIC_INPUT(scale, TensorType({DT_UINT64, DT_BF16}))
    .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))
    .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(per_token_scale, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))
    .ATTR(split_item, Int, 0)
    .ATTR(dtype, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(transpose_x, Bool, false)
    .ATTR(group_type, Int, -1)
    .OP_END_FACTORY_REG(GroupedMatmul)

  /**
  * @brief Function TomeUnmerge. \n

  * @par Inputs:
  * @li attention: A Tensor List, attention out. Shape is (B, S, H). S = S2 + S1 - (S2 + S1) * top_rate
  * @li ori_index_a: A Tensor List of origin index A. Shape is (B, S1, H), Value range [0, S1 + S2), no dup and cant dup with ori_index_b.
  * @li ori_index_b: A Tensor List of origin index B. Shape is (B, S2, H), Value range [0, S1 + S2), no dup and cant dup with ori_index_a.
  * @li topk_indice: A Tensor List of topK indice. Shape is (B, S1, H), S1 must equal with ori_index_a.
  * @li arg_max: A Tensor List of ArgMax. Shape is (B, S1, H), S1 must equal with ori_index_a.

  * @par Attributes:
  * @li top_rate: A Float. rate to calculate how many rows of token_a merge to token_b. default is "0.5".

  * @par Outputs:
  * unzip_token: A Tensor List, restore by ori_index_a and ori_index_b.
  * @par Restrictions:
  * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
  */
  REG_OP(TomeUnmerge)
      .INPUT(attention, TensorType({DT_FLOAT16}))
      .INPUT(ori_index_a, TensorType({DT_INT64}))
      .INPUT(ori_index_b, TensorType({DT_INT64}))
      .INPUT(topk_indice, TensorType({DT_INT64}))
      .INPUT(arg_max, TensorType({DT_INT64}))
      .OUTPUT(unzip_token, TensorType({DT_FLOAT16}))
      .ATTR(top_rate, Float, 0.5)
      .OP_END_FACTORY_REG(TomeUnmerge)

/**
* @brief Function GroupedMatMulAllReduce. \n

* @par Inputs:
* @li x: A Tensor List.
* @li weight: A Tensor List of weight.
* @li bias: A Tensor List of bias.
* @li group_list: a Tensor.

* @par Attributes:
* @li splitItem: A int64.
* @li group: A string. A required String identifying the group of ranks
* @li reduceOp: A string. A required string identifying the reduction operation to
 perform. support "sum".
* @li commTurn: A int64. Number of communications with AICPU. Default: 0.

* @par Outputs:
* y: A Tensor List.
*/
  REG_OP(GroupedMatMulAllReduce)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(splitItem, Int, 0)
    .REQUIRED_ATTR(group, String)
    .ATTR(reduceOp, String, "sum")
    .ATTR(commTurn, Int, 0)
    .OP_END_FACTORY_REG(GroupedMatMulAllReduce)

  /**
   * @brief compute init routing for moe input.
   * @par Inputs:
   * @li x: A Tensor. Type is:BFloat16, Float16 or Float32.
   * @li row_idx: A Tensor. Type is:Int32.
   * @li expert_idx: A Tensor. Type is:Int32.
   * @par Outputs:
   * @li expanded_x: A Tensor. Type is:BFloat16, Float16 or Float32.
   * @li expanded_row_idx: A Tensor. Type is:Int32.
   * @li expanded_expert_idx: A Tensor. Type is:Int32.
   * @par Attributes:
   * @li active_num: Required parameter. Type is:Int32.
   */
    REG_OP(MoeInitRouting)
    .INPUT(x, "T1")
    .INPUT(row_idx, "T2")
    .INPUT(expert_idx, "T2")
    .OUTPUT(expanded_x, "T1")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expanded_expert_idx, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .REQUIRED_ATTR(active_num, Int)
    .OP_END_FACTORY_REG(MoeInitRouting)

  /**
   * @brief compute init routing quant for moe input.
   * @par Inputs:
   * @li x: A Tensor. Type is:BFloat16, Float16 or Float32.
   * @li row_idx: A Tensor. Type is:Int32.
   * @li expert_idx: A Tensor. Type is:Int32.
   * @par Outputs:
   * @li expanded_x: A Tensor. Type is:Int8.
   * @li expanded_row_idx: A Tensor. Type is:Int32.
   * @li expanded_expert_idx: A Tensor. Type is:Int32.
   * @par Attributes:
   * @li active_num: Required parameter. Type is:Int32.
   * @li scale: Required parameter. Type is:Float.
   * @li offset: Required parameter. Type is:Float.
   */
    REG_OP(MoeInitRoutingQuant)
    .INPUT(x, "T1")
    .INPUT(row_idx, "T2")
    .INPUT(expert_idx, "T2")
    .OUTPUT(expanded_x, "T3")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expanded_expert_idx, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .DATATYPE(T3, TensorType({DT_INT8}))
    .REQUIRED_ATTR(active_num, Int)
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .OP_END_FACTORY_REG(MoeInitRoutingQuant)

  /**
   * @brief compute softmax and topk for moe input.
   * @par Inputs:
   * @li x: A Tensor. Type is:BFloat16, Float16 or Float32.
   * @li finished: A Tensor. Type is:Bool.
   * @par Outputs:
   * @li y: A Tensor. Type is:BFloat16, Float16 or Float32.
   * @li expert_idx: A Tensor. Type is:Int32.
   * @li row_idx: A Tensor. Type is:Int32.
   * @par Attributes:
   * @li k: Required parameter. Type is:Int32.
   */
    REG_OP(MoeGatingTopKSoftmax)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(finished, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(row_idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(k, Int)
    .OP_END_FACTORY_REG(MoeGatingTopKSoftmax)

/**
* @brief In MoE computation, the final step involves processing and merging the output results of the MoE FNN.
* @par Inputs:
* @li expanded_x: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li x1: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li x2: An optional Tensor. Type is:BFloat16, Float16 or Float32.
* @li bias: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li scales: A Tensor. Type is:BFloat16, Float16 or Float32.
* @li expanded_row_idx: A Tensor. Type is:Int32.
* @li expanded_expert_idx: A Tensor. Type is:Int32.
* @par Outputs:
* @li y: A Tensor. Type is:BFloat16, Float16 or Float32.
*/
REG_OP(MoeFinalizeRouting)
    .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(expanded_row_idx, TensorType({DT_INT32, DT_INT32, DT_INT32}))
    .INPUT(expanded_expert_idx, TensorType({DT_INT32, DT_INT32, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeFinalizeRouting)

/**
* @brief Binary finds the position of the last row processed by each expert in the sorted_experts array.
* @par Inputs:
* @li sorted_experts: A Tensor. Type is:Int32.
* @par Outputs:
* @li total_rows_before_expert: A Tensor. Type is:Int32.
* @par Attributes:
* @li num_experts: Required parameter. Type is:Int. The value must be more than 0 and less than 2147483647.
*/
REG_OP(MoeComputeExpertTokens)
    .INPUT(sorted_experts, "T")
    .OUTPUT(total_rows_before_expert, "T")
    .REQUIRED_ATTR(num_experts, Int)
    .DATATYPE(T, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(MoeComputeExpertTokens)

/**
* @brief The fusion operator of Gelu activation function and quantum quantization.
* @par Inputs:
* @li x: A Tensor. Type is DT_FLOAT32, DT_FLOAT16, DT_BF16.
* @li input_scale: An optional Tensor. When quant_mode is "static",it is a required Tensor.
*     Type is DT_FLOAT32, DT_FLOAT16, DT_BF16.The type is consistent with x or has higher accuracy
* @li input_offset: An optional Tensor. Type is DT_FLOAT32, DT_FLOAT16, DT_BF16.
*     The shape and type should be the same as input_scale.It can also be null.
* @par Outputs:
* @li y: A Tensor. Type is DT_INT8.
* @li out_scale: A Tensor. Type is DT_FLOAT32.
* @par Attributes:
* @li approximate: Required parameter. Type is String. The value must be none or tanh.
* @li quant_mode: Required parameter. Type is String. The value must be dynamic or static.
*/
REG_OP(GeluQuant)
    .INPUT(x, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(input_scale, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(input_offset, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .OUTPUT(out_scale, TensorType({DT_FLOAT32}))
    .ATTR(approximate, String, "none")
    .ATTR(quant_mode, String, "dynamic")
    .OP_END_FACTORY_REG(GeluQuant)

/**
* @brief
   swin_transformer model specific structure.Operator only supports swin_transformer. \n
* @par Inputs:
* Eight inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li gamma: A Tensor. Must be one of the following types: float16.
* @li beta: A Tensor. Must be one of the following types: float16.
* @li weight: A Tensor. Must be one of the following types: int8.
* @li bias: A Tensor. Must be one of the following types: float16.
* @li quant_scale: A Tensor. Must be one of the following types: float16.
* @li quant_offset: A Tensor. Must be one of the following types: float16.
* @li dequant_scale: A Tensor. Must be one of the following types: uint64. \n

* @par Attributes:
* @li head_num: A required attribute, the type is int. Defaults to 1.
* @li seq_length: A required attribute, the type is int. Defaults to 32.
* @li epsilon: A required attribute, the type is float. Defaults to 0.000001.
* @li ori_height: A required attribute, the type is int. Defaults to 7
* @li ori_weight: A required attribute, the type is int. Defaults to 7. \n
* @li h_win_szie: A required attribute, the type is int. Defaults to 7. \n
* @li w_win_size: A required attribute, the type is int. Defaults to 7. \n
* @li weight_transpose: A required attribute, the type is bool. Defaults to true. \n

* @par Outputs:
* Three outputs, including:
* @li query_output: A Tensor. Must be one of the following types: float16.
* @li key_output: A Tensor. Must be one of the following types: float16.
* @li value_output: A Tensor. Must be one of the following types: float16. \n
*/
REG_OP(SwinTransformerLnQkvQuant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .INPUT(quant_scale, TensorType({DT_FLOAT16}))
    .INPUT(quant_offset, TensorType({DT_FLOAT16}))
    .INPUT(dequant_scale, TensorType({DT_UINT64}))
    .OUTPUT(query_output, TensorType({DT_FLOAT16}))
    .OUTPUT(key_output, TensorType({DT_FLOAT16}))
    .OUTPUT(value_output, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(seq_length, Int)
    .REQUIRED_ATTR(epsilon, Float)
    .REQUIRED_ATTR(ori_height, Int)
    .REQUIRED_ATTR(ori_weight, Int)
    .REQUIRED_ATTR(h_win_szie, Int)
    .REQUIRED_ATTR(w_win_size, Int)
    .REQUIRED_ATTR(weight_transpose, Bool)
    .OP_END_FACTORY_REG(SwinTransformerLnQkvQuant)

/**
* @brief The quant fusion operator of SwinAttentionScoreQuant.

* @par Inputs:
* @li query: A matrix Tensor. The type support int8.
* @li key: A matrix Tensor. The type support int8.
* @li value: A matrix Tensor. The type support int8.
* @li scale_quant: A Tensor. The type support fp16.
* @li scale_dequant1: A Tensor. The type support uint64.
* @li scale_dequant2: A Tensor. The type support uint64.
* @li bias_quant: A Tensor. The type support fp16.
* @li bias_dequant1: A Tensor. The type support int32.
* @li bias_dequant2: A Tensor. The type support int32.
* @li padding_mask1: A matrix Tensor. The type support fp16.
* @li padding_mask2: A matrix Tensor. The type support fp16.
* @li attention_score: A matrix Tensor. The type support fp16.

* @par Attributes:
* @li query_transpose: A bool. Whether query is transposed. Default: false.
* @li key_transpose: A bool. Whether key is transposed. Default: false.
* @li value_transpose: A bool. Whether value is transposed. Default: false.
* @li softmax_axes: A int. Which axes to calculate softmax. Default: -1.

* @par Outputs:
* @li attention_score: A matrix Tensor. The type support fp16. \n
*/
REG_OP(SwinAttentionScoreQuant)
    .INPUT(query, TensorType({DT_INT8}))
    .INPUT(key, TensorType({DT_INT8}))
    .INPUT(value, TensorType({DT_INT8}))
    .INPUT(scale_quant, TensorType({DT_FLOAT16}))
    .INPUT(scale_dequant1, TensorType({DT_UINT64}))
    .INPUT(scale_dequant2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(bias_quant, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_dequant1, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(bias_dequant2, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(padding_mask1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(padding_mask2, TensorType({DT_FLOAT16}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(value_transpose, Bool, false)
    .ATTR(softmax_axes, Int, -1)
    .OP_END_FACTORY_REG(SwinAttentionScoreQuant)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
