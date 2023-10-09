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

#ifndef LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_H
#define LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_H

#include <map>
#include <string>
#include <thread>
#include <vector>
#include "ge/ge_ir_build.h"
#include "llm_engine_types.h"

namespace llm {
class DecoderManager;
class PromptManager;
class LLMEngine {
 public:
  explicit LLMEngine(uint64_t cluster_id) : cluster_id_(cluster_id) {}
  ~LLMEngine();
  ge::Status LLMEngineInitialize(const std::vector<ge::ModelBufferData> &model_buffer_datas,
                                 const std::map<ge::AscendString, ge::AscendString> &options);
  static LLMEngineStatus fetchLLMEngineStatus();
  int64_t FetchLlmEngineQueueStatus();
  // API2：execute prompt
  ge::Status RunPromptAsync(const LLMReq &req, const std::vector<ge::Tensor> &inputs, ge::RunAsyncCallback callback);
  ge::Status RunPrompt(const LLMReq &req, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &output);

  // API3: Execute the Decoder calculation
  // a. Assign an idle index for the request
  // b. Fetch KVCache from the specified prompt cluster based on the request and write it to the corresponding idle
  // index c. Perform Decoder computation and asynchronously return the calculation result
  ge::Status RunDecoderAsync(const LLMReq &req, const std::vector<ge::Tensor> &inputs, ge::RunAsyncCallback callback);
  ge::Status RunDecoder(const LLMReq &req, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &output);

  // Externally notifies that the request has ended. If the request has already started execution, release the
  // placeholders associated with incremental inference. If the request has not yet started execution, remove it from
  // the queue.
  ge::Status LLMReqComplete(const LLMReq &req);
  ge::Status LLMEngineFinalize();

 private:
  std::shared_ptr<PromptManager> prompt_manager_;
  std::shared_ptr<DecoderManager> decoder_manager_;
  uint64_t cluster_id_;
  std::string role_;
  std::atomic<bool> is_initialized_{false};
  std::atomic<bool> is_finalized_{false};
};
}  // namespace llm

#endif  // LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_H
