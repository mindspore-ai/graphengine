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

#ifndef AIR_CXX_EXECUTOR_OPTION_H
#define AIR_CXX_EXECUTOR_OPTION_H

#include <string>

namespace gert {
class VISIBILITY_EXPORT ExecutorOption {
 public:
  ExecutorOption() : executor_type_("") {}
  explicit ExecutorOption(std::string executor_type) : executor_type_(executor_type) {}
  const std::string &GetExecutorType() const {
    return executor_type_;
  }
  virtual ~ExecutorOption() = default;

 private:
  std::string executor_type_;
};
}  // namespace gert

#endif  // AIR_CXX_EXECUTOR_OPTION_H
