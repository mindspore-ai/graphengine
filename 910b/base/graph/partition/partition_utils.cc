/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "graph/partition/partition_utils.h"
#include "graph/compute_graph.h"
#include "graph/utils/node_utils.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"

namespace ge {
bool PartitionUtils::IsDataLike(const NodePtr &node) {
  return (node->GetType() == CONSTANT) ||
         (node->GetType() == DATA) ||
         (node->GetType() == AIPPDATA) ||
         (node->GetType() == CONSTANTOP) ||
         (node->GetType() == VARIABLE);
}

graphStatus PartitionUtils::GetStageNames(const NodePtr &node, vector<std::string> &stage_names) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc, "[Get][OpDesc] OpDesc is nullptr");
  GE_ASSERT_TRUE(AttrUtils::GetListStr(op_desc, ATTR_NAME_PIPELINE_STAGE, stage_names),
                 "[Get][Attr] stage names on node: %s failed.", node->GetName().c_str());
  return SUCCESS;
}

graphStatus PartitionUtils::CheckWritableVarNode(const ComputeGraphPtr &root_graph) {
  constexpr int32_t kInvalidOutIndex = -1;
  for (const auto &node : root_graph->GetDirectNode()) {
    if (node->GetType() == VARIABLE) {
      const auto &op_desc = node->GetOpDesc();
      const auto &input_name_index = op_desc->GetAllInputName();
      for (const auto &name_index : input_name_index) {
        const int32_t out_index = op_desc->GetOutputIndexByName(name_index.first);
        GE_ASSERT_TRUE(out_index == kInvalidOutIndex, "[Check][Variable] node %s is ref of index %d.",
                       node->GetName().c_str(), out_index);
      }
    }
  }
  return SUCCESS;
}

bool PartitionUtils::CheckSameStageName(const vector<std::string> &name1, const vector<std::string> &name2) {
  if (name1.size() != name2.size()) {
    GELOGD("Stage names is not same, name1 size %zu, name2 size %zu.", name1.size(), name2.size());
    return false;
  }
  for (size_t idx = 0; idx < name1.size(); ++idx) {
    if (name1[idx] != name2[idx]) {
      GELOGD("Stage name1 %s is not same as stage name2 %s.", name1[idx].c_str(), name2[idx].c_str());
      return false;
    }
  }
  return true;
}

graphStatus PartitionUtils::CheckDtResourceNodes(const ComputeGraphPtr &root_graph) {
  std::unordered_set<std::string> unique_stage;
  for (const auto &node : root_graph->GetDirectNode()) {
    if (!NodeUtils::IsDtResourceNode(node)) {
      continue;
    }
    vector<std::string> stage_names;
    GE_ASSERT_GRAPH_SUCCESS(GetStageNames(node, stage_names), "[Call][GetStageNames] failed, node %s.",
                            node->GetName().c_str());
    GE_ASSERT_TRUE(stage_names.size() <= kUniqueStageNameNum,
                   "[Check][StageNum] of node %s failed, stage size %zu should < %zu", node->GetName().c_str(),
                   stage_names.size(), kUniqueStageNameNum);
    if (!stage_names.empty()) {
      unique_stage.insert(stage_names[0]);
    }
    GE_ASSERT_TRUE(unique_stage.size() <= kUniqueStageNameNum,
                   "[Check][StageNum] of node %s failed, got different stage names[size(%zu)]", node->GetName().c_str(),
                   unique_stage.size());
  }
  return SUCCESS;
}

graphStatus PartitionUtils::SetSubgraphGraphId(const ComputeGraphPtr &root_graph,
                                               const ComputeGraphPtr &subgraph) {
  std::string session_graph_id;
  GE_ASSERT_TRUE(AttrUtils::GetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                 "[Get][Attr] %s on root graph:%s failed.", ATTR_NAME_SESSION_GRAPH_ID.c_str(),
                 root_graph->GetName().c_str());
  GE_ASSERT_TRUE(AttrUtils::SetStr(*subgraph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                 "[Set][Attr] %s on subgraph:%s failed.", ATTR_NAME_SESSION_GRAPH_ID.c_str(),
                 subgraph->GetName().c_str());
  return GRAPH_SUCCESS;
}

bool PartitionUtils::IsOutNode(const NodePtr &node) {
  if ((node->GetType() == NETOUTPUT) ||
      (node->GetType() == NOOP)) {
    return true;
  }
  return false;
}
} // namespace ge