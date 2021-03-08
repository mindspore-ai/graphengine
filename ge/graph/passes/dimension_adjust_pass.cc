/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "graph/passes/dimension_adjust_pass.h"

#include <memory>
#include <string>
#include <vector>
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
const int kDataInputIndex = 0;
const int kRemoveInputIndex = 1;
}  // namespace

Status DimensionAdjustPass::Run(ge::NodePtr &node) {
  if (node == nullptr) {
    GELOGE(PARAM_INVALID, "node is nullptr.");
    return PARAM_INVALID;
  }

  OpDescPtr op_desc_ptr = node->GetOpDesc();
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "GetOpDesc return nullptr.");
    return PARAM_INVALID;
  }

  string type;
  Status ret = GetOriginalType(node, type);
  if (ret != SUCCESS) {
    GELOGE(ret, "DimensionAdjustPass get originnal type fail.");
    return ret;
  }

  KernelFactory &factory = KernelFactory::Instance();
  shared_ptr<Kernel> op_kernel = factory.Create(type);
  if (op_kernel == nullptr) {
    return SUCCESS;
  }
  bool is_unknown = false;
  auto ret_status = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
  if (ret_status != GRAPH_SUCCESS) {
    GELOGW("Get node unknown status failed, node name:%s, type:%s.", node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (is_unknown) {
    GELOGI("Current node %s, type %s is unknown shape which should be skip.",
           node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  // call compute function
  ret = op_kernel->Compute(node);
  if (ret != SUCCESS) {
    if (ret == NOT_CHANGED) {
      return SUCCESS;
    }
    GELOGE(ret, "DimensionAdjustPass compute failed");
    return ret;
  }
  if (node->GetAllInDataAnchors().size() > static_cast<size_t>(kRemoveInputIndex)) {
    ret = PassUtils::UnlinkNodeWithControlCopy(node, kRemoveInputIndex);
    if (ret != SUCCESS) {
      GELOGE(ret, "DimensionAdjustPass unlink node with control copy fail.");
      return ret;
    }
  }

  ret = DealWithInNodes(node);
  if (ret != SUCCESS) {
    GELOGE(ret, "DealWithInNodes of %s failed.", node->GetName().c_str());
    return ret;
  }

  std::vector<int> data_relink_io_map = {kDataInputIndex};
  return IsolateAndDeleteNode(node, data_relink_io_map);
}

Status DimensionAdjustPass::DealWithInNodes(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  auto graph = node->GetOwnerComputeGraph();
  auto in_data_anchors = node->GetAllInDataAnchors();
  for (auto &in_data_anchor : in_data_anchors) {
    if (in_data_anchor == nullptr) {
      continue;
    }
    auto in_node_anchor = in_data_anchor->GetPeerOutAnchor();
    if (in_node_anchor == nullptr) {
      continue;
    }
    auto in_node = in_node_anchor->GetOwnerNode();
    if (in_node->GetType() == SWITCHN) {
      auto identity_name = node->GetName() + "_ctrl_identity_" + std::to_string(in_data_anchor->GetIdx());
      auto identity =
          AddIdentityNodeToGraph(identity_name, node->GetOpDesc()->GetInputDesc(in_data_anchor->GetIdx()), graph);
      GE_CHECK_NOTNULL(identity);
      GELOGI("Create new identity node[%s] after node %s[type: %s] success.", identity->GetName().c_str(),
             in_node->GetName().c_str(), in_node->GetType().c_str());
      GE_CHK_STATUS_RET(GraphUtils::AddEdge(in_node_anchor, identity->GetInDataAnchor(0)))
      GE_CHECK_NOTNULL(identity->GetOutControlAnchor());
      if (identity->GetOutControlAnchor()->IsLinkedWith(node->GetInControlAnchor())) {
        continue;
      }
      GE_CHK_STATUS_RET(GraphUtils::AddEdge(identity->GetOutControlAnchor(), node->GetInControlAnchor()))
    }
  }

  return SUCCESS;
}

NodePtr DimensionAdjustPass::AddIdentityNodeToGraph(const string &name, const GeTensorDesc &tensor,
                                                    ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    GELOGE(INTERNAL_ERROR, "Comput graph ptr is null in creating identity node.");
    return nullptr;
  }

  OpDescPtr desc = MakeShared<OpDesc>("", "");
  if (desc == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Failed to create op desc.");
    return nullptr;
  }

  desc->SetName(name);
  desc->SetType(IDENTITY);
  auto ret = desc->AddInputDesc(tensor);
  auto ret2 = desc->AddOutputDesc(tensor);
  if ((ret != GRAPH_SUCCESS) || (ret2 != GRAPH_SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "Failed to add input/output desc in creating identity.");
    return nullptr;
  }

  return graph->AddNodeFront(desc);
}
}  // namespace ge
