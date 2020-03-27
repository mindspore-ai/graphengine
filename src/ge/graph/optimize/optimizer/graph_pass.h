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

#ifndef GE_GRAPH_OPTIMIZE_OPTIMIZER_GRAPH_PASS_H_
#define GE_GRAPH_OPTIMIZE_OPTIMIZER_GRAPH_PASS_H_

#include <string>
#include <vector>

#include "./pass.h"
#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"

namespace ge {
///
/// @ingroup domi
/// @brief
/// @author
///
class GraphPass : public Pass<ge::ComputeGraph> {
 public:
  ///
  /// @param [in] graph  graph to optimize
  /// @return SUCCESS optimize success
  /// @return NOT_CHANGED not optimized
  /// @return other optimize fail
  /// @author
  ///
  virtual Status Run(ge::ComputeGraphPtr graph) = 0;
  static void RecordOriginalNames(std::vector<ge::NodePtr> originalNodes, const ge::NodePtr &node) {
    GE_CHECK_NOTNULL_JUST_RETURN(node);
    std::vector<std::string> originalNames;
    for (ge::NodePtr nodeTmp : originalNodes) {
      GE_IF_BOOL_EXEC(nodeTmp == nullptr, return;)
      std::vector<std::string> namesTmp;
      ge::OpDescPtr opdescTmp = nodeTmp->GetOpDesc();
      if (!ge::AttrUtils::GetListStr(opdescTmp, "original_op_names", namesTmp)) {
        GELOGW("Get original_op_names failed");
      }
      if (namesTmp.size() != 0) {
        originalNames.insert(originalNames.end(), namesTmp.begin(), namesTmp.end());
      } else {
        originalNames.emplace_back(opdescTmp->GetName());
      }
    }

    if (originalNames.size() == 0) {
      std::string tmp;
      originalNames.emplace_back(tmp);
    }

    GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(node->GetOpDesc(), "_datadump_original_op_names", originalNames),
                       return, "Set original_op_names fail.")
  }

  static bool IsConstNode(const ge::NodePtr &node) {
    if (node == nullptr) {
      GELOGE(PARAM_INVALID, "Input param node is nullptr.");
      return false;
    }
    if (node->GetOpDesc()->GetType() == CONSTANTOP) {
      return true;
    } else if (node->GetOpDesc()->GetType() == FRAMEWORKOP) {
      string type;
      GE_CHK_BOOL_EXEC(ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type),
                         return false, "Get original_type for op %s fail!", node->GetName().c_str());
      GE_IF_BOOL_EXEC(type == CONSTANT, GELOGI("Is const op"); return true);
      return false;
    } else {
      return false;
    }
  }
};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_OPTIMIZER_GRAPH_PASS_H_
