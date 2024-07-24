/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
  Status SaveModelWeights(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          const size_t model_index = 0UL) const;
  Status SaveTaskParam(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                       const GeModelPtr &ge_model, const size_t model_index = 0UL);
  Status SaveModelTbeKernel(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0UL) const;
  Status SaveModelDescExtend(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                             const size_t model_index = 0UL);

 private:
  void SetModelDescInfo(const std::shared_ptr<uint8_t> &buff) { model_desc_info_ = buff; }
  void SetStaticTaskInfo(const std::shared_ptr<uint8_t> &buff) { static_task_info_ = buff; }
  void SetDynamicTaskInfo(const std::shared_ptr<uint8_t> &buff) { dynamic_task_info_ = buff; }
  void SetTaskParamInfo(const std::shared_ptr<uint8_t> &buff) { task_param_info_ = buff; }
  void SetModelExtendInfo(const std::shared_ptr<uint8_t> &buff) { model_extend_info_ = buff; }

  std::shared_ptr<uint8_t> model_desc_info_ = nullptr;
  std::shared_ptr<uint8_t> static_task_info_ = nullptr;
  std::shared_ptr<uint8_t> dynamic_task_info_ = nullptr;
  std::shared_ptr<uint8_t> task_param_info_ = nullptr;
  std::shared_ptr<uint8_t> model_extend_info_ = nullptr;
  bool is_offline_ = true;
  std::unordered_map<int64_t, uint32_t> search_ids_;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_NANO_MODEL_SAVE_HELPER_H_
