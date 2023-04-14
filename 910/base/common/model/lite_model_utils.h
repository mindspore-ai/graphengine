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
#ifndef GE_COMMON_MODEL_LITE_MODEL_UTILS_H_
#define GE_COMMON_MODEL_LITE_MODEL_UTILS_H_

#include "common/model/ge_root_model.h"

namespace ge {
class LiteModelUtils {
 public:
  LiteModelUtils() = default;
  ~LiteModelUtils() = default;

  static std::vector<int64_t> GetInputSize(const ConstOpDescPtr &op_desc);
  static std::vector<int64_t> GetOutputSize(const ConstOpDescPtr &op_desc);
  static std::vector<int64_t> GetWorkspaceSize(const ConstOpDescPtr &op_desc);
  static std::vector<int64_t> GetWeightSize(const ConstOpDescPtr &op_desc);
};
}  // namespace ge
#endif  // GE_COMMON_MODEL_LITE_MODEL_UTILS_H_