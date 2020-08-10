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

#ifndef INC_FRAMEWORK_GE_RUNTIME_MODEL_RUNNER_H_
#define INC_FRAMEWORK_GE_RUNTIME_MODEL_RUNNER_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "common/ge_inner_error_codes.h"
#include "common/ge_types.h"
#include "ge_runtime/davinci_model.h"

namespace ge {
namespace model_runner {
class RuntimeModel;
using RuntimeInfo = std::tuple<uint32_t, uint32_t, void *>;
class ModelRunner {
 public:
  static ModelRunner &Instance();

  bool LoadDavinciModel(uint32_t device_id, uint64_t session_id, uint32_t model_id,
                        std::shared_ptr<DavinciModel> davinci_model, std::shared_ptr<ModelListener> listener);

  bool DistributeTask(uint32_t model_id);

  bool LoadModelComplete(uint32_t model_id);

  const std::vector<uint32_t> &GetTaskIdList(uint32_t model_id) const;

  const std::vector<uint32_t> &GetStreamIdList(uint32_t model_id) const;

  const std::map<std::string, std::shared_ptr<RuntimeInfo>> &GetRuntimeInfoMap(uint32_t model_id) const;

  void *GetModelHandle(uint32_t model_id) const;

  bool UnloadModel(uint32_t model_id);

  bool RunModel(uint32_t model_id, const InputData &input_data, OutputData *output_data);

  bool GetInputOutputDescInfo(uint32_t model_id, bool zero_copy, std::vector<InputOutputDescInfo> *input_desc,
                              std::vector<InputOutputDescInfo> *output_desc, std::vector<uint32_t> *input_format,
                              std::vector<uint32_t> *output_format);

 private:
  ModelRunner() = default;
  ~ModelRunner() = default;

  std::unordered_map<uint32_t, std::shared_ptr<RuntimeModel>> runtime_models_;
};
}  // namespace model_runner
}  // namespace ge

#endif  // INC_FRAMEWORK_GE_RUNTIME_MODEL_RUNNER_H_
