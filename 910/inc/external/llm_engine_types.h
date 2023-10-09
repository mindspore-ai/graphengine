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

 private:
  uint64_t req_id_{0UL};
  // 请求prompt的句子长度，做完padding的值， 用于申请prompt的KVCache
  uint64_t prompt_length_{0UL};
  uint64_t prompt_cluster_id_{0UL};  // in/out， runPrompt的输出， runecoder的输入
  uint64_t decoder_cluster_id_{0UL};
};

constexpr char LLM_OPTION_INPUTS_BATCH_DIM_INDEX[] = "llm.InputsBatchSizeDimIndex";
constexpr char LLM_OPTION_WAIT_TIME[] = "llm.WaitTime";
constexpr char kPrompt[] = "Prompt";
constexpr char kDecoder[] = "Decoder";
constexpr const char LLM_OPTION_OM_CACHE_PATH[] = "llm.OmCachePath";
constexpr const char LLM_OPTION_CLUSTER_DEPLOYMENT_CONFIG[] = "llm.ClusterDeploymentConfig";
constexpr const char LLM_OPTION_ROLE[] = "llm.Role";
constexpr const char LLM_OPTION_MODEL_INPUTS_SHAPES[] = "llm.InputShapes";
constexpr const char LLM_OPTION_MODEL_INPUTS_DTYPES[] = "llm.InputDtypes";
constexpr const char LLM_OPTION_MODEL_KV_CACHE_SHAPES[] = "llm.RefInputShapes";
constexpr const char LLM_OPTION_MODEL_KV_CACHE_DTYPES[] = "llm.RefInputDtypes";
constexpr const char LLM_OPTION_OUTPUT_NUM[] = "llm.OutputNums";
}  // namespace llm

#endif  // LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_TYPES_H
