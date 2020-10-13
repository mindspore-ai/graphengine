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

#ifndef INC_FRAMEWORK_ENGINE_DNNENGINE_H_
#define INC_FRAMEWORK_ENGINE_DNNENGINE_H_

#include <map>
#include <string>
#include <vector>

#include "common/ge_inner_error_codes.h"
#include "common/ge_types.h"
#include "graph/types.h"

namespace ge {
enum PriorityEnum {
  COST_0 = 0,
  COST_1,
  COST_2,
  COST_3,
  COST_9 = 9,
  COST_10 = 10,
};

struct DNNEngineAttribute {
  std::string engine_name;
  std::vector<std::string> mem_type;
  uint32_t compute_cost;
  enum RuntimeType runtime_type;  // HOST, DEVICE
  // If engine input format must be specific, set this attribute, else set FORMAT_RESERVED
  Format engine_input_format;
  Format engine_output_format;
};

class DNNEngine {
 public:
  virtual ~DNNEngine() = default;
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;
  virtual Status Finalize() = 0;
  virtual void GetAttributes(DNNEngineAttribute &attr) const = 0;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_ENGINE_DNNENGINE_H_
