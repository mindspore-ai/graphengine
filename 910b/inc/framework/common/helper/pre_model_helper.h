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

#ifndef INC_FRAMEWORK_COMMON_HELPER_PRE_MODEL_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_PRE_MODEL_HELPER_H_

#include "framework/common/helper/model_helper.h"

namespace ge {
class GE_FUNC_VISIBILITY PreModelHelper : public ModelHelper {
 public:
  PreModelHelper() = default;
  virtual ~PreModelHelper() override = default;
  PreModelHelper(const PreModelHelper &) = default;
  PreModelHelper &operator=(const PreModelHelper &) & = default;

  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                                   ModelBufferData &model, const bool is_unknown_shape) override;

 private:
  Status SaveToExeOmModel(const GeModelPtr &ge_model, const std::string &output_file, ModelBufferData &model,
                          const GeRootModelPtr &ge_root_model = nullptr);
  Status SaveAllModelPartiton(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                              const size_t model_index = 0UL);
  Status SavePreModelHeader(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model) const;
  Status SaveModelDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                       const size_t model_index = 0UL);
  Status SaveTaskDesc(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                      const size_t model_index = 0UL);
  Status SaveKernelArgs(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                        const size_t model_index = 0UL);
  Status SaveKernelBin(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                       const size_t model_index = 0UL);
  Status SaveModelCustAICPU(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0UL) const;

 private:
  void SetKernelArgsInfo(const std::shared_ptr<uint8_t> &buff) {
    kernel_args_info_ = buff;
  }
  void SetTaskDescInfo(const std::shared_ptr<uint8_t> &buff) {
    task_desc_info_ = buff;
  }
  void SetModelDescInfo(const std::shared_ptr<uint8_t> &buff) {
    model_desc_info_ = buff;
  }
  void SetKernelBinInfo(const std::shared_ptr<uint8_t> &buff) {
    kernel_bin_info_ = buff;
  }

  std::shared_ptr<uint8_t> kernel_args_info_ = nullptr;
  std::shared_ptr<uint8_t> task_desc_info_ = nullptr;
  std::shared_ptr<uint8_t> model_desc_info_ = nullptr;
  std::shared_ptr<uint8_t> kernel_bin_info_ = nullptr;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_PRE_MODEL_HELPER_H_
