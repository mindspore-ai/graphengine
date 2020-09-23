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

#include "graph/passes/reshape_recovery_pass.h"
#include "common/ge/ge_util.h"

namespace ge {
namespace {
NodePtr CreateReshape(const ConstGeTensorDescPtr &src, const ConstGeTensorDescPtr &dst, const ComputeGraphPtr &graph) {
  static std::atomic_long reshape_num(0);
  auto next_num = reshape_num.fetch_add(1);
  auto reshape = MakeShared<OpDesc>("Reshape_ReshapeRecoveryPass_" + std::to_string(next_num), RESHAPE);
  if (reshape == nullptr) {
    return nullptr;
  }
  auto ret = reshape->AddInputDesc("x", *src);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }
  ret = reshape->AddInputDesc("shape", GeTensorDesc(GeShape(), Format(), DT_INT32));
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }
  ret = reshape->AddOutputDesc("y", *dst);
  if (ret != GRAPH_SUCCESS) {
    return nullptr;
  }

  return graph->AddNode(reshape);
}

Status InsertReshapeIfNeed(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  for (auto src_anchor : node->GetAllOutDataAnchors()) {
    auto src_tensor = node->GetOpDesc()->GetOutputDescPtr(src_anchor->GetIdx());
    GE_CHECK_NOTNULL(src_tensor);
    for (auto dst_anchor : src_anchor->GetPeerInDataAnchors()) {
      auto dst_node = dst_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(dst_node);
      GE_CHECK_NOTNULL(dst_node->GetOpDesc());
      auto dst_tensor = dst_node->GetOpDesc()->GetInputDescPtr(dst_anchor->GetIdx());
      bool is_need_insert_reshape = src_tensor->GetShape().GetDims() != UNKNOWN_RANK &&
                                    dst_tensor->GetShape().GetDims() != UNKNOWN_RANK &&
                                    src_tensor->GetShape().GetDims() != dst_tensor->GetShape().GetDims();
      if (is_need_insert_reshape) {
        auto reshape = CreateReshape(src_tensor, dst_tensor, node->GetOwnerComputeGraph());
        GE_CHECK_NOTNULL(reshape);
        auto ret = GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, reshape);
        if (ret != GRAPH_SUCCESS) {
          GELOGE(INTERNAL_ERROR, "Failed to insert reshape between node %s and %s", node->GetName().c_str(),
                 dst_node->GetName().c_str());
          return INTERNAL_ERROR;
        }
        GELOGI("Insert reshape between %s and %s to keep the shape continues", node->GetName().c_str(),
               dst_node->GetName().c_str());
      }
    }
  }
  return SUCCESS;
}
}  // namespace

Status ReshapeRecoveryPass::Run(ComputeGraphPtr graph) {
  for (const auto &node : graph->GetDirectNode()) {
    auto ret = InsertReshapeIfNeed(node);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}
}  // namespace ge
