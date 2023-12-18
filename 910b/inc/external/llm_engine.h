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
class PriorityScheduleManager;
enum class RunMode : uint32_t {
  kSeparateSchedule = 0,
  kPrioritySchedule
};
class LLMEngine {
 public:
  explicit LLMEngine(uint64_t cluster_id) : cluster_id_(cluster_id) {}
  ~LLMEngine();
  ge::Status LLMEngineInitialize(const std::vector<ge::ModelBufferData> &model_buffer_datas,
                                 const std::map<ge::AscendString, ge::AscendString> &options);
  ge::Status LLMEngineInitializeV2(
      const std::map<ge::AscendString, std::vector<ge::ModelBufferData>> &model_type_to_buffer_datas,
      const std::map<ge::AscendString, ge::AscendString> &options);
  static LLMEngineStatus FetchLLMEngineStatus();
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

  // Preload prompt prefix model to generate kv cache
  ge::Status PreloadPromptPrefix(const LLMReq &req, const std::vector<ge::Tensor> &inputs);

  // Release kv cache of prompt prefix model
  ge::Status ReleasePromptPrefix(const LLMReq &req);

  // @brief 从Prompt cluster拉取该request对应的KV到本Decoder cluster的暂存区中，每次调用成功后都会覆盖暂存区之前的kv
  // @param [in] req instance of LLMReq
  // @return Status result of function
  //         SUCCESS: 成功
  //         LLM_PARAM_INVALID: 参数错误, 如cluster id校验错误, 当前非manual batching模式等
  //         LLM_KV_CACHE_NOT_EXIST: prompt中不存在该request对应的KV
  //         FAILED: 拉取KV失败
  ge::Status PullKv(const LLMReq &req);

  // @brief 将KV从本暂存区中merge到batch中, 该接口会释放暂存区中的kv
  // @param [in] req_id
  // @param [in] batch_index
  // @return Status result of function
  //         SUCCESS: 成功
  //         LLM_PARAM_INVALID: 参数错误, 如cluster id校验错, 当前非manual batching模式, batch_index越界等
  //         LLM_KV_CACHE_NOT_EXIST: KV不在暂存区
  //         FAILED: 合并KV失败
  ge::Status MergeKv(const uint64_t req_id, const int32_t batch_index);

  // @brief 执行Decoder推理
  // @param [in] req_ids 每个batch对应的request_id, 如果一个batch无效，其对应的req_id需要设置为UINT64_MAX
  // @param [in] inputs 输入tensor
  // @param [out] outputs 输出tensor
  // @return Status result of function
  //         SUCCESS: 成功
  //         LLM_PARAM_INVALID: 参数错误, 如当前非manual batching模式, request_id与batch不对应等
  //         FAILED: 执行推理失败
  ge::Status RunDecoder(const std::vector<uint64_t> &req_ids,
                        const std::vector<ge::Tensor> &inputs,
                        std::vector<ge::Tensor> &outputs);

  ge::Status LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                          const int32_t timeout = -1);

  ge::Status UnlinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                            const int32_t timeout = -1);

 private:
  std::shared_ptr<PromptManager> prompt_manager_;
  std::shared_ptr<DecoderManager> decoder_manager_;
  std::shared_ptr<PriorityScheduleManager> priority_schedule_manager_;
  uint64_t cluster_id_;
  std::string role_;
  std::atomic<bool> is_initialized_{false};
  std::atomic<bool> is_finalized_{false};
  RunMode run_mode_{RunMode::kSeparateSchedule};
};
}  // namespace llm

#endif  // LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_H
