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

#ifndef INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_H_

#include <memory>
#include <string>

#include "framework/common/helper/model_save_helper_factory.h"
#include "framework/common/helper/om_file_helper.h"
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "framework/common/types.h"
#include "graph/model.h"
#include "platform/platform_info.h"
#include "common/op_so_store/op_so_store.h"

namespace ge {
class GE_FUNC_VISIBILITY ModelSaveHelper {
 public:
  ModelSaveHelper() = default;

  virtual ~ModelSaveHelper() = default;

  virtual Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model,
                                   const std::string &output_file,
                                   ModelBufferData &model,
                                   const bool is_unknown_shape) = 0;

  virtual Status SaveToOmModel(const GeModelPtr &ge_model,
                               const std::string &output_file,
                               ModelBufferData &model,
                               const GeRootModelPtr &ge_root_model = nullptr) = 0;

  virtual void SetSaveMode(const bool val) = 0;
 protected:
  ModelSaveHelper(const ModelSaveHelper &) = default;
  ModelSaveHelper &operator=(const ModelSaveHelper &) & = default;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_H_