/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef GE_GRAPH_COMMON_OME_CONTEXT_H_
#define GE_GRAPH_COMMON_OME_CONTEXT_H_

#include "framework/omg/omg_inner_types.h"

namespace ge {
struct OmeContext {
  std::string dynamic_node_type;
  std::vector<NodePtr> data_nodes;
  std::vector<NodePtr> getnext_nosink_nodes;
  std::vector<std::vector<int32_t>> dynamic_shape_dims;
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  bool is_subgraph_multi_batch = false;
};
}  // namespace ge
#endif  // GE_GRAPH_COMMON_OME_CONTEXT_H_
