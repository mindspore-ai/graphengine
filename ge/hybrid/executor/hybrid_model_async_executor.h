/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef GE_HYBRID_EXECUTOR_MODEL_HYBRID_MODEL_ASYNC_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_MODEL_HYBRID_MODEL_ASYNC_EXECUTOR_H_
#include <atomic>
#include <mutex>
#include <future>
#include "external/ge/ge_api_error_codes.h"
#include "external/ge/ge_api_types.h"
#include "graph/load/new_model_manager/data_inputer.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "runtime/stream.h"

namespace ge {
namespace hybrid {
class HybridModel;
class HybridModelAsyncExecutor {
 public:
  explicit HybridModelAsyncExecutor(HybridModel *model);
  ~HybridModelAsyncExecutor();

  Status Init();

  Status Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs);

  Status Start(const std::shared_ptr<ModelListener> &listener);

  void SetDeviceId(uint32_t device_id);

  void SetModelId(uint32_t model_id);

  Status Stop();

  Status EnqueueData(const std::shared_ptr<InputDataWrapper> &data);

 private:
  Status InitInputTensors();

  Status RunInternal();

  Status SyncVarData();

  Status HandleResult(Status exec_ret,
                      uint32_t data_id,
                      HybridModelExecutor::ExecuteArgs &args,
                      OutputData *output_data);

  Status CopyOutputs(HybridModelExecutor::ExecuteArgs &args,
                     OutputData *output_data,
                     std::vector<ge::OutputTensorInfo> &outputs);

  Status OnComputeDone(uint32_t data_index, uint32_t result_code, std::vector<ge::OutputTensorInfo> &outputs);

  Status PreRun(InputData &current_data);

  Status CopyInputData(const InputData &current_data);

  std::mutex mu_;
  HybridModel *model_;
  uint32_t device_id_ = 0U;
  uint32_t model_id_ = 0U;
  std::atomic_bool run_flag_;
  std::unique_ptr<DataInputer> data_inputer_;
  std::unique_ptr<HybridModelExecutor> executor_;
  std::future<Status> future_;
  uint64_t iterator_count_ = 0;

  rtStream_t stream_ = nullptr;
  std::map<uint32_t, TensorValue> input_tensors_;
  std::shared_ptr<ModelListener> listener_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_MODEL_HYBRID_MODEL_ASYNC_EXECUTOR_H_
