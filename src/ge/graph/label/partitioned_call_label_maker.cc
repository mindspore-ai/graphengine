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

#include "partitioned_call_label_maker.h"

#include "common/util.h"
#include "common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace ge {
constexpr int32_t kSubGraphIndex = 0;

/**
 * @ingroup ge
 * @brief Make label node to functional call.
 * @param [in/out] label_index: serial id for whole graph.
 * @return: 0 for success / others for fail
 */
Status PartitionedCallLabelMaker::Run(uint32_t &label_index) {
  GE_CHECK_NOTNULL(parent_node_);
  GE_CHECK_NOTNULL(parent_graph_);

  OpDescPtr call_desc = parent_node_->GetOpDesc();
  GE_CHECK_NOTNULL(call_desc);

  std::string sub_graph_name = call_desc->GetSubgraphInstanceName(kSubGraphIndex);
  if (sub_graph_name.empty()) {
    GELOGE(INTERNAL_ERROR, "Node: %s has no subgraph name.", sub_graph_name.c_str());
    return FAILED;
  }

  ComputeGraphPtr sub_graph = parent_graph_->GetSubgraph(sub_graph_name);
  if (sub_graph == nullptr) {
    GELOGE(INTERNAL_ERROR, "Node: %s has no subgraph.", sub_graph_name.c_str());
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_LABEL_MAKER(PARTITIONEDCALL, PartitionedCallLabelMaker);
REGISTER_LABEL_MAKER(STATEFULPARTITIONEDCALL, PartitionedCallLabelMaker);
}  // namespace ge
