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

#ifndef INC_FRAMEWORK_OMG_MODEL_TOOL_H_
#define INC_FRAMEWORK_OMG_MODEL_TOOL_H_

#include <memory>
#include <string>

#include "framework/common/debug/ge_log.h"
#include "proto/ge_ir.pb.h"

namespace ge {
class GE_FUNC_VISIBILITY ModelTool {
 public:
  static Status GetModelInfoFromOm(const char *model_file, ge::proto::ModelDef &model_def, uint32_t &modeldef_size);

  static Status GetModelInfoFromPbtxt(const char *model_file, ge::proto::ModelDef &model_def);
};
}  // namespace ge

#endif  // INC_FRAMEWORK_OMG_MODEL_TOOL_H_
