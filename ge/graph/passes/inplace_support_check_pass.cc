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

#include "graph/passes/inplace_support_check_pass.h"
#include "framework/common/debug/log.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace {
const uint32_t kInplaceSupportOutputIndex = 0;
const uint32_t kInplaceSupportOutputNum = 1;
static const std::set<std::string> src_node_types = { ge::DATA, ge::ANN_DATA, ge::AIPPDATA,
                                                      ge::CONSTANT, ge::CONSTANTOP,
                                                      ge::VARIABLE, ge::VARIABLEV2 };
}

namespace ge {
Status InplaceSupportCheckPass::Run(NodePtr &node) {
  GELOGD("InplaceSupportCheckPass running");
  if (src_node_types.count(node->GetType()) > 0) {
    GELOGD("meet src_node %s, skip InplaceSupportCheckPass", node->GetName().c_str());
    return SUCCESS;
  }
  if (node->GetAllOutDataAnchorsSize() != kInplaceSupportOutputNum) {
    GELOGD("output num of node %s is not %u, skip InplaceSupportCheckPass",
           node->GetName().c_str(), kInplaceSupportOutputNum);
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(node->GetOpDesc());
  const DataType &output_type = node->GetOpDesc()->GetOutputDesc(kInplaceSupportOutputIndex).GetDataType();
  const GeShape &output_shape = node->GetOpDesc()->GetOutputDesc(kInplaceSupportOutputIndex).GetShape();
  GELOGD("process InplaceSupportCheckPass on node %s", node->GetName().c_str());
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const auto &peer_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_data_anchor == nullptr) {
      continue;
    }
    auto in_node = peer_data_anchor->GetOwnerNode();
    if (src_node_types.count(in_node->GetType()) > 0) {
      GELOGD("meet src_node %s", in_node->GetName().c_str());
      continue;
    }
    if (peer_data_anchor->GetPeerInDataNodesSize() != kInplaceSupportOutputNum) {
      GELOGD("peer_data_anchor links with multi in_data_anchors");
      continue;
    }

    int32_t inplace_input_idx = in_data_anchor->GetIdx();
    const DataType &input_type = node->GetOpDesc()->GetInputDesc(inplace_input_idx).GetDataType();
    const GeShape &input_shape = node->GetOpDesc()->GetInputDesc(inplace_input_idx).GetShape();
    if (input_type !=  output_type) {
      GELOGD("DataType mismatch, in_idx=%d, input_type=%u, output_type=%u", inplace_input_idx, input_type, output_type);
      continue;
    }
    if (input_shape.GetDims() != output_shape.GetDims()) {
      GELOGD("Shape mismatch, in_idx=%d, input_shape=[%s], output_shape=[%s]",
             inplace_input_idx, input_shape.ToString().c_str(), output_shape.ToString().c_str());
      continue;
    }

    GELOGD("add attr INPLACE_SUPPORT_INPUT_INDEX on node %s, input_idx=%d", node->GetName().c_str(), inplace_input_idx);
    if (!AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(kInplaceSupportOutputIndex),
                           INPLACE_SUPPORT_INPUT_INDEX, inplace_input_idx)) {
      GELOGE(FAILED, "Set attr INPLACE_SUPPORT_INPUT_INDEX on node %s failed.", node->GetName().c_str());
      return FAILED;
    }
    AddRePassNode(node);
    break;
  }

  GELOGD("InplaceSupportCheckPass success");
  return SUCCESS;
}
}  // namespace ge
