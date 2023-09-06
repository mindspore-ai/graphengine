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

#ifndef D_AIR_INC_FRAMEWORK_PNE_DATA_FLOW_GRAPH_DATA_FLOW_OPERATORS_H_
#define D_AIR_INC_FRAMEWORK_PNE_DATA_FLOW_GRAPH_DATA_FLOW_OPERATORS_H_

#include "flow_graph/flow_graph.h"

namespace ge {
namespace dflow {
constexpr const char_t *kAttrNameTagNames = "tag_names";

class FlowSend : public FlowOperator {
 public:
  FlowSend(const char *name, const char *tag_name);
  FlowSend(const char *name, const std::vector<AscendString> &tag_names);
  ~FlowSend() override;
};

class FlowRecv : public FlowOperator {
 public:
  FlowRecv(const char *name, const char *tag_name);
  FlowRecv(const char *name, const std::vector<AscendString> &tag_names);
  ~FlowRecv() override;
};
}  // namespace dflow
}  // namespace ge

#endif  // D_AIR_INC_FRAMEWORK_PNE_DATA_FLOW_GRAPH_DATA_FLOW_OPERATORS_H_
