/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_TYPES_H
#define LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_TYPES_H

#include "ge/ge_api_types.h"

namespace llm {
constexpr uint64_t kInvalidReqId = UINT64_MAX;
constexpr uint64_t kInvalidPrefixId = UINT64_MAX;

struct LLMEngineStatus {
  uint64_t empty_max_prompt_kv;  // 全量集群可容纳的KV
  int32_t num_free_blocks;
};

struct LLMModelStatus {
  int32_t num_free_blocks;
  int32_t reserved[128];
};

class LLMReq {
 public:
  LLMReq() = default;
  ~LLMReq() = default;

  void SetReqId(const uint64_t req_id) {
    req_id_ = req_id;
  }

  uint64_t GetReqId() const {
    return req_id_;
  }

  void SetPromptLength(const uint64_t prompt_length) {
    prompt_length_ = prompt_length;
  }

  uint64_t GetPromptLength() const {
    return prompt_length_;
  }

  void SetPromptClusterId(const uint64_t prompt_cluster_id) {
    prompt_cluster_id_ = prompt_cluster_id;
  }

  uint64_t GetPromptClusterId() const {
    return prompt_cluster_id_;
  }

  void SetDecoderClusterId(const uint64_t decoder_cluster_id) {
    decoder_cluster_id_ = decoder_cluster_id;
  }

  uint64_t GetDecoderClusterId() const {
    return decoder_cluster_id_;
  }

  void SetPrefixId(const uint64_t prefix_id) {
    prefix_id_ = prefix_id;
  }

  uint64_t GetPrefixId() const {
    return prefix_id_;
  }

  void SetSequenceLen(const uint64_t sequence_length) {
    sequence_length_ = sequence_length;
  }

  uint64_t GetSequenceLen() const {
    return sequence_length_;
  }

 private:
  uint64_t req_id_{kInvalidReqId};
  // 请求prompt的句子长度，做完padding的值， 用于申请prompt的KVCache
  uint64_t prompt_length_{0UL};
  uint64_t prompt_cluster_id_{0UL};  // in/out， runPrompt的输出， runecoder的输入
  uint64_t decoder_cluster_id_{0UL};
  uint64_t prefix_id_{kInvalidPrefixId};
  uint64_t sequence_length_{0UL};
  int8_t reserved_[120];
};

constexpr const char LLM_OPTION_INPUTS_BATCH_DIM_INDEX[] = "llm.InputsBatchSizeDimIndex";
constexpr const char LLM_OPTION_INPUT_WAIT_TIME[] = "llm.InputWaitTime";
constexpr const char kPrompt[] = "Prompt";
constexpr const char kDecoder[] = "Decoder";
constexpr const char LLM_BATCH_MODE_MANUAL[] = "manual";
constexpr const char LLM_BATCH_MODE_AUTO[] = "auto";
constexpr const char LLM_MODEL_TYPE_INFERENCE[] = "inference";
constexpr const char LLM_MODEL_TYPE_POSTPROCESS[] = "postprocess";
constexpr const char LLM_OPTION_OM_CACHE_PATH[] = "llm.OmCachePath";
// 用于区分加载的方式
constexpr const char LLM_OPTION_LOAD_TYPE_FLOWMODEL[] = "llm.IsFlowModel";
constexpr const char LLM_OPTION_CLUSTER_INFO[] = "llm.ClusterInfo";
constexpr const char LLM_OPTION_ROLE[] = "llm.Role";
constexpr const char LLM_OPTION_MODEL_INPUTS_SHAPES[] = "llm.InputShapes";
constexpr const char LLM_OPTION_MODEL_INPUTS_DTYPES[] = "llm.InputDtypes";
constexpr const char LLM_OPTION_MODEL_INPUTS_PADDING[] = "llm.InputPaddings";
constexpr const char LLM_OPTION_MODEL_KV_CACHE_SHAPES[] = "llm.RefInputShapes";
constexpr const char LLM_OPTION_MODEL_KV_CACHE_DTYPES[] = "llm.RefInputDtypes";
constexpr const char LLM_OPTION_OUTPUT_NUM[] = "llm.OutputNums";
constexpr const char LLM_OPTION_SYNC_KV_CACHE_WAIT_TIME[] = "llm.SyncKvCacheWaitTime";
constexpr const char LLM_OPTION_NN_EXECUTE_WAIT_TIME[] = "llm.NnExecuteWaitTime";
constexpr const char LLM_OPTION_PROCESS_REQUEST_WAIT_TIME[] = "llm.ProcessRequestWaitTime";

constexpr const char LLM_OPTION_RUN_MODE[] = "llm.RunMode";
constexpr const char LLM_OPTION_PROMPT_AND_DOCODER_INTERLEAVED_STEP[] = "llm.PromptAndDecoderInterleavedStep";
constexpr const char LLM_OPTION_INPUTS_BATCH_PADDING[] = "llm.InputsBatchPadding";
constexpr const char LLM_OPTION_OUTPUT_MAX_SIZE[] = "llm.OutputMaxSize";
constexpr const char LLM_OPTION_GRAPH_COMPILER_CACHE_DIR[] = "llm.graph_compiler_cache_dir";
constexpr const char LLM_OPTION_GRAPH_KEYS[] = "llm.graph_keys";
constexpr const char LLM_OPTION_BATCH_MODE[] = "llm.batch_mode";
constexpr const char LLM_OPTION_GRAPH_PATH[] = "llm.GraphPath";
constexpr const char LLM_OPTION_INPUT_NODE_DEPLOYMENT[] = "llm.InputNodeDeployment";

constexpr const char LLM_OPTION_POSTPROCESS_MODEL_INPUTS_SHAPES[] = "llm.PostProcessInputShapes";
constexpr const char LLM_OPTION_POSTPROCESS_MODEL_INPUTS_DTYPES[] = "llm.PostProcessInputDtypes";
constexpr const char LLM_OPTION_POSTPROCESS_MODEL_OM_CACHE_PATH[] = "llm.PostProcessOmCachePath";
constexpr const char LLM_OPTION_POSTPROCESS_OUTPUT_NUM[] = "llm.PostProcessOutputNums";
// optional, "postprocess:1;postprocess:2"
constexpr const char LLM_OPTION_OUTPUT_MAPPING[] = "llm.OutputMapping";

// options for pipeline stage execution, example: "40;40"
constexpr const char LLM_OPTION_KV_CACHE_COUNTS[] = "llm.KvCacheCounts";
constexpr const char LLM_OPTION_HCOM_CLUSTER_CONFIG[] = "llm.HcomClusterConfig";
// "enable" or "disable", default value is "enable" if option not set
constexpr const char LLM_OPTION_PIPELINE_EXECUTION[] = "llm.PipelineExecution";
// example, 2 stage, "0,1,2,3,4,5,6;"
constexpr const char LLM_OPTION_PIPELINE_INPUT_INDICES[] = "llm.PipelineInputIndices";
// PagedAttention options
constexpr const char LLM_OPTION_ENABLE_PAGED_ATTENTION[] = "llm.EnablePagedAttention";
constexpr const char LLM_OPTION_PAGED_ATTENTION_BLOCK_SIZE[] = "llm.PagedAttentionBlockSize";
constexpr const char LLM_OPTION_PAGED_ATTENTION_BLOCKS_NUM[] = "llm.PagedAttentionBlocksNum";
constexpr const char LLM_OPTION_PAGED_ATTENTION_MAX_SEQ_LEN[] = "llm.PagedAttentionMaxSeqLen";
constexpr const char LLM_OPTION_PAGED_ATTENTION_MAX_SEQS_NUM[] = "llm.PagedAttentionMaxSeqsNum";
constexpr const char LLM_OPTION_PAGED_ATTENTION_MAX_PROMPT_LEN[] = "llm.PagedAttentionMaxPromptLen";

constexpr const char LLM_OPTION_ENABLE_SWITCH_ROLE[] = "llm.EnableSwitchRole";

struct IpInfo {
  uint32_t ip = 0U;
  uint16_t port = 0U;
};

struct ClusterInfo {
  uint64_t remote_cluster_id = 0U;
  int32_t remote_role_type = 0;
  std::vector<IpInfo> local_ip_infos;
  std::vector<IpInfo> remote_ip_infos;
};
}  // namespace llm

#endif  // LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_TYPES_H
