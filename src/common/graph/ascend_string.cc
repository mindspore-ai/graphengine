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

#include "external/graph/ascend_string.h"

namespace ge {
AscendString::AscendString(const char* name) {
  if (name != nullptr) {
    name_ = std::shared_ptr<std::string>(new (std::nothrow) std::string(name));
  }
}

const char* AscendString::GetString() const {
  if (name_ == nullptr) {
    return nullptr;
  }

  return (*name_).c_str();
}
}  // namespace ge
