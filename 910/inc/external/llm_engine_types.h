/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_TYPES_H
#define LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_TYPES_H

#include "ge/ge_api_types.h"

namespace llm {
constexpr uint64_t kInvalidReqId = UINT64_MAX;
constexpr uint64_t kInvalidPrefixId = UINT64_MAX;

struct LLMEngineStatus {
  uint64_t empty_max_prompt_kv;  // 全量集群可容纳的KV
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

 private:
  uint64_t req_id_{kInvalidReqId};
  // 请求prompt的句子长度，做完padding的值， 用于申请prompt的KVCache
  uint64_t prompt_length_{0UL};
  uint64_t prompt_cluster_id_{0UL};  // in/out， runPrompt的输出， runecoder的输入
  uint64_t decoder_cluster_id_{0UL};
  uint64_t prefix_id_{kInvalidPrefixId};
  int8_t reserved_[128];
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
constexpr const char LLM_OPTION_CLUSTER_DEPLOYMENT_CONFIG[] = "llm.ClusterDeploymentConfig";
constexpr const char LLM_OPTION_CLUSTER_INFO[] = "llm.ClusterInfo";
constexpr const char LLM_OPTION_ROLE[] = "llm.Role";
constexpr const char LLM_OPTION_MODEL_INPUTS_SHAPES[] = "llm.InputShapes";
constexpr const char LLM_OPTION_MODEL_INPUTS_DTYPES[] = "llm.InputDtypes";
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

constexpr const char LLM_OPTION_POSTPROCESS_MODEL_INPUTS_SHAPES[] = "llm.PostProcessInputShapes";
constexpr const char LLM_OPTION_POSTPROCESS_MODEL_INPUTS_DTYPES[] = "llm.PostProcessInputDtypes";
constexpr const char LLM_OPTION_POSTPROCESS_MODEL_OM_CACHE_PATH[] = "llm.PostProcessOmCachePath";
constexpr const char LLM_OPTION_POSTPROCESS_OUTPUT_NUM[] = "llm.PostProcessOutputNums";
// optional, "postprocess:1;postprocess:2"
constexpr const char LLM_OPTION_OUTPUT_MAPPING[] = "llm.OutputMapping";

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
