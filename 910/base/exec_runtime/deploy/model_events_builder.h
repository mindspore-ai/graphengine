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

#ifndef BASE_EXEC_RUNTIME_DEPLOY_MODEL_EVENTS_BUILDER_H
#define BASE_EXEC_RUNTIME_DEPLOY_MODEL_EVENTS_BUILDER_H

#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "external/ge/ge_api_types.h"

namespace ge {
struct EventNode {
  std::string type;
  std::string group_name;
  std::string event_name;
  int32_t group_tag;
  int32_t model_peer_rank;
  int32_t logic_peer_rank;
};

class ModelEventsBuilder {
 public:
  explicit ModelEventsBuilder(const std::string &event_table) : event_table_(event_table) {};
  ~ModelEventsBuilder() = default;
  Status Build();
  const std::vector<EventNode> &GetModelEvents() const;
 private:
  std::vector<EventNode> event_nodes_;
  std::string event_table_;
};

class HcomExecUtils {
 public:
  static std::string ToJson(const std::vector<EventNode> &event_nodes);
};
} // namespace ge

#endif  // BASE_EXEC_RUNTIME_DEPLOY_MODEL_EVENTS_BUILDER_H
