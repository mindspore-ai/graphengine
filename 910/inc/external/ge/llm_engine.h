/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_H
#define LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_H

#include <map>
#include <string>
#include <vector>
#include "ge/ge_ir_build.h"
#include "llm_engine_types.h"

namespace llm {
class LLMEngine {
 public:
  explicit LLMEngine(const uint64_t cluster_id);
  ~LLMEngine();
  ge::Status LLMEngineInitialize(const std::vector<ge::ModelBufferData> &model_buffer_datas,
                                 const std::map<ge::AscendString, ge::AscendString> &options);
  // @brief 添加模型
  // @param [in] model_type_to_buffer_datas
  // @param [in] options 模型options
  // @param [out] model_id 模型ID
  // @return 返回状态 SUCCESS: 成功
  ge::Status AddLLMModel(const std::map<ge::AscendString, std::vector<ge::ModelBufferData>> &model_type_to_buffer_datas,
                         const std::map<ge::AscendString, ge::AscendString> &options, uint64_t &model_id);
  ge::Status LLMEngineInitializeV2(
      const std::map<ge::AscendString, std::vector<ge::ModelBufferData>> &model_type_to_buffer_datas,
      const std::map<ge::AscendString, ge::AscendString> &options);
  LLMEngineStatus FetchLLMEngineStatus();
  // API2：execute prompt
  ge::Status RunPromptAsync(const LLMReq &req, const std::vector<ge::Tensor> &inputs, ge::RunAsyncCallback callback,
                            uint64_t model_id = 0UL);

  // @brief 运行prompt模型
  // @param [in] llm请求
  // @param [in] 模型输入
  // @param [in] 模型输出
  // @param [out] model_id 模型ID
  // @return 返回状态 SUCCESS: 成功
  ge::Status RunPrompt(const LLMReq &req, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &output,
                       uint64_t model_id = 0UL);

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
  ge::Status PreloadPromptPrefix(const LLMReq &req, const std::vector<ge::Tensor> &inputs, uint64_t model_id = 0UL);

  // Release kv cache of prompt prefix model
  ge::Status ReleasePromptPrefix(const LLMReq &req, uint64_t model_id = 0UL);

  // @brief 从Prompt cluster拉取该request对应的KV到本Decoder cluster的暂存区中，每次调用成功后都会覆盖暂存区之前的kv
  // @param [in] req instance of LLMReq
  // @return Status result of function
  //         SUCCESS: 成功
  //         LLM_PARAM_INVALID: 参数错误, 如cluster id校验错误, 当前非manual batching模式等
  //         LLM_KV_CACHE_NOT_EXIST: prompt中不存在该request对应的KV
  //         FAILED: 拉取KV失败
  ge::Status PullKv(const LLMReq &req, uint64_t model_id = 0UL);

  // @brief 将KV从本暂存区中merge到batch中, 该接口会释放暂存区中的kv
  // @param [in] req_id 请求id, 需要与PullKv的匹配
  // @param [in] batch_index 目标batch的index
  // @param [in] batch_id 目标batch的id, 使能交错式调度时可以设置，默认为0
  // @return Status result of function
  //         SUCCESS: 成功
  //         LLM_PARAM_INVALID: 参数错误, 如cluster id校验错, 当前非manual batching模式, batch_index越界等
  //         LLM_KV_CACHE_NOT_EXIST: KV不在暂存区
  //         FAILED: 合并KV失败
  ge::Status MergeKv(const uint64_t req_id, const int32_t batch_index, const int32_t batch_id = 0,
                     uint64_t model_id = 0UL);

  // @brief 执行Decoder推理
  // @param [in] req_ids 每个batch对应的request_id, 如果一个batch无效，其对应的req_id需要设置为UINT64_MAX
  // @param [in] inputs 输入tensor
  // @param [out] outputs 输出tensor
  // @return Status result of function
  //         SUCCESS: 成功
  //         LLM_PARAM_INVALID: 参数错误, 如当前非manual batching模式, request_id与batch不对应等
  //         FAILED: 执行推理失败
  ge::Status RunDecoder(const std::vector<uint64_t> &req_ids, const std::vector<ge::Tensor> &inputs,
                        std::vector<ge::Tensor> &outputs);

  // @brief 执行Decoder推理
  // @param [in] requests 每个batch对应的request, 如果一个batch无效，其对应的req_id需要设置为UINT64_MAX
  // @param [in] inputs 输入tensor
  // @param [out] outputs 输出tensor
  // @param [in] model_id 模型ID
  // @return Status result of function
  //         SUCCESS: 成功
  //         LLM_PARAM_INVALID: 参数错误, 如当前非manual batching模式, request_id与batch不对应等
  //         FAILED: 执行推理失败
  ge::Status RunDecoder(const std::vector<LLMReq> &requests, const std::vector<ge::Tensor> &inputs,
                        std::vector<ge::Tensor> &outputs, uint64_t model_id = 0UL);

  // @brief 进行device间建链
  // @param [in] clusters 需要建链的cluster信息
  // @param [in] timeout 超时时间，单位ms
  // @param [out] rets 每个cluster建链结果
  // @return Status result of function
  //         SUCCESS: 成功
  //         其他做错误码: 执行推理失败
  ge::Status LinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                          const int32_t timeout = -1);

  // @brief 进行device间断链
  // @param [in] clusters 需要建链的cluster信息
  // @param [in] timeout 超时时间，单位ms
  // @param [out] rets 每个cluster断链结果
  // @return Status result of function
  //         SUCCESS: 成功
  //         其他做错误码: 执行推理失败
  ge::Status UnlinkClusters(const std::vector<ClusterInfo> &clusters, std::vector<ge::Status> &rets,
                            const int32_t timeout = -1);

  // @brief 执行Prompt推理
  // @param [in] reqs 请求, 按实际请求设置，不须要使用req_id为UINT64_MAX的req补齐到batch size
  // @param [in] inputs 输入tensor
  // @param [out] outputs 输出tensor
  // @param [in] model_id 模型ID
  // @return Status result of function
  //         SUCCESS: 成功
  //         LLM_PARAM_INVALID: 参数错误, 如当前非manual batching模式
  //         FAILED: 执行推理失败
  ge::Status RunPrompt(const std::vector<LLMReq> &reqs, const std::vector<ge::Tensor> &inputs,
                       std::vector<ge::Tensor> &output, uint64_t model_id = 0UL);

  LLMModelStatus FetchLLMModelStatus(uint64_t model_id);
 private:
  class LlmEngineImpl;
  std::unique_ptr<LlmEngineImpl> impl_;
};
}  // namespace llm

#endif  // LLM_ENGINE_INC_EXTERNAL_LLM_ENGINE_H
