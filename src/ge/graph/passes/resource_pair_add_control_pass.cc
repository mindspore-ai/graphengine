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

#include "graph/passes/resource_pair_add_control_pass.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include "framework/common/debug/ge_log.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_adapter.h"

namespace {
const char *const kSeparate = "/";
const std::map<std::string, std::string> kResourcePairType = {{"StackPush", "StackPop"}};
const std::set<std::string> kResourceTypes = {"StackPush", "StackPop"};
}  // namespace

namespace ge {
Status ResourcePairAddControlPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  GELOGD("ResourcePairAddControlPass pass start.");
  std::map<std::string, std::map<std::string, NodePtr>> prefix_2_node_per_type;
  // find all node of condition type, store with type and scope prefix key
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (kResourceTypes.find(node->GetType()) != kResourceTypes.end()) {
      std::string node_name = node->GetName();
      std::string node_prefix;
      size_t last_separate_index = node_name.find_last_of(kSeparate);
      if (last_separate_index != std::string::npos) {
        node_prefix = node_name.substr(0, last_separate_index);
      }
      prefix_2_node_per_type[node->GetType()][node_prefix] = node;
      GELOGD("ResourcePairAddControlPass insert prefix:%s, op_name:%s, op_type:%s", node_prefix.c_str(),
             node_name.c_str(), node->GetType().c_str());
    }
  }

  // according type pair, find same prefix node, add control edge
  for (auto &resource_type_pair : kResourcePairType) {
    auto from_item_prefix_2_node = prefix_2_node_per_type.find(resource_type_pair.first);
    if (from_item_prefix_2_node != prefix_2_node_per_type.end()) {
      for (auto &prefix_2_node : from_item_prefix_2_node->second) {
        const std::string &prefix = prefix_2_node.first;
        NodePtr from_node = prefix_2_node.second;
        GE_CHECK_NOTNULL(from_node);
        auto to_item_prefix_2_node = prefix_2_node_per_type.find(resource_type_pair.second);
        if (to_item_prefix_2_node == prefix_2_node_per_type.end()) {
          GELOGE(PARAM_INVALID, "find peer type node fail, suffix:%s, from_type:%s, to_type:%s", prefix.c_str(),
                 resource_type_pair.first.c_str(), resource_type_pair.second.c_str());
          return PARAM_INVALID;
        }
        auto to_prefix_2_node = to_item_prefix_2_node->second.find(prefix);
        if (to_prefix_2_node == to_item_prefix_2_node->second.end()) {
          GELOGE(PARAM_INVALID, "find peer prefix node fail, suffix:%s, from_type:%s, to_type:%s", prefix.c_str(),
                 resource_type_pair.first.c_str(), resource_type_pair.second.c_str());
          return PARAM_INVALID;
        }
        NodePtr to_node = to_prefix_2_node->second;
        GE_CHECK_NOTNULL(to_node);
        auto from_anchor = from_node->GetOutControlAnchor();
        auto to_anchor = to_node->GetInControlAnchor();
        GE_CHECK_NOTNULL(from_anchor);
        GE_CHECK_NOTNULL(to_anchor);
        graphStatus ret = from_anchor->LinkTo(to_anchor);
        if (ret != GRAPH_SUCCESS) {
          GELOGE(PARAM_INVALID, "link fail, from_node:%s, to_node:%s, from_type:%s, to_type:%s",
                 from_node->GetName().c_str(), to_node->GetName().c_str(), resource_type_pair.first.c_str(),
                 resource_type_pair.second.c_str());
          return PARAM_INVALID;
        }
        GELOGD("link success, from_node:%s, to_node:%s, from_type:%s, to_type:%s", from_node->GetName().c_str(),
               to_node->GetName().c_str(), resource_type_pair.first.c_str(), resource_type_pair.second.c_str());
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
