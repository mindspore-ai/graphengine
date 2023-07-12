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

#ifndef INC_FRAMEWORK_COMMON_HELPER_NANO_MODEL_SAVE_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_NANO_MODEL_SAVE_HELPER_H_

#include "framework/common/helper/model_helper.h"

namespace ge {
class GE_FUNC_VISIBILITY NanoModelSaveHelper : public ModelHelper {
 public:
  NanoModelSaveHelper() = default;
  virtual ~NanoModelSaveHelper() override = default;
  NanoModelSaveHelper(const NanoModelSaveHelper &) = default;
  NanoModelSaveHelper &operator=(const NanoModelSaveHelper &) & = default;

  virtual Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                           ModelBufferData &model, const bool is_unknown_shape) override;

  virtual void SetSaveMode(const bool val) override {
    is_offline_ = val;
  }

 private:
  Status SaveToDbg(const GeModelPtr &ge_model, const std::string &output_file) const;
  Status SaveToExeOmModel(const GeModelPtr &ge_model, const std::string &output_file,
                       ModelBufferData &model);
  Status SaveAllModelPartiton(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                              const size_t model_index = 0UL);
  Status SaveModelDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                       const GeModelPtr &ge_model, const size_t model_index = 0UL);
  Status SaveStaticTaskDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                       const GeModelPtr &ge_model, const size_t model_index = 0UL);
  Status SaveDynamicTaskDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                       const GeModelPtr &ge_model, const size_t model_index = 0UL);
  Status SaveTaskParam(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                       const GeModelPtr &ge_model, const size_t model_index = 0UL);
  Status SaveModelTbeKernel(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0UL) const;

 private:
  void SetModelDescInfo(const std::shared_ptr<uint8_t> &buff) { model_desc_info_ = buff; }
  void SetStaticTaskInfo(const std::shared_ptr<uint8_t> &buff) { static_task_info_ = buff; }
  void SetDynamicTaskInfo(const std::shared_ptr<uint8_t> &buff) { dynamic_task_info_ = buff; }
  void SetTaskParamInfo(const std::shared_ptr<uint8_t> &buff) { task_param_info_ = buff; }

  std::shared_ptr<uint8_t> model_desc_info_ = nullptr;
  std::shared_ptr<uint8_t> static_task_info_ = nullptr;
  std::shared_ptr<uint8_t> dynamic_task_info_ = nullptr;
  std::shared_ptr<uint8_t> task_param_info_ = nullptr;
  bool is_offline_ = true;
  std::unordered_map<int64_t, uint32_t> search_ids_;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_NANO_MODEL_SAVE_HELPER_H_
