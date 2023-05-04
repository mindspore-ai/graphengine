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

#include "framework/common/helper/model_save_helper.h"

namespace ge {
class GE_FUNC_VISIBILITY NanoModelSaveHelper : public ModelSaveHelper {
 public:
  NanoModelSaveHelper() = default;
  ~NanoModelSaveHelper() override = default;

  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                           ModelBufferData &model, const bool is_unknown_shape) override;

  Status SaveToOmModel(const GeModelPtr &ge_model, const std::string &output_file,
                       ModelBufferData &model, const GeRootModelPtr &ge_root_model = nullptr) override;
  void SetSaveMode(const bool val) override {
    is_offline_ = val;
  }

 private:
  bool is_offline_ = true;
  NanoModelSaveHelper(const NanoModelSaveHelper &) = default;
  NanoModelSaveHelper &operator=(const NanoModelSaveHelper &) & = default;
  Status SaveToDbg(const GeModelPtr &ge_model, const std::string &output_file) const;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_NANO_MODEL_SAVE_HELPER_H_
