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

#include "graph/optimize/optimizer/allreduce_fusion_pass.h"
#include <string>
#include "common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "hccl/base.h"
#include "hccl/hcom.h"

namespace ge {
Status AllReducePass::Run(ge::ComputeGraphPtr graph) {
  GELOGI("FusionAllReducePass: start");
  std::vector<NodePtr> fusionOps;
  std::vector<float> inputGradientSize;
  std::vector<float> inputGradientTime;

  static const float inputGradientSizeTemp = 0.0;
  static const float inputGradientTimeTemp = 0.0;

  // Get all nodes
  for (auto nodePtr : graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(nullptr == nodePtr, GELOGW("FusionAllReducePass: null node exists"); continue;);

    ge::OpDescPtr opDescPtr = nodePtr->GetOpDesc();
    GE_IF_BOOL_EXEC(nullptr == opDescPtr,
                    GELOGW("FusionAllReducePass: desc of node %s is null", nodePtr->GetName().c_str());
                    continue;)
    GE_IF_BOOL_EXEC(HCOMALLREDUCE == opDescPtr->GetType(),
                    // the op is allreduce and fusion > 0, then run fusion
                    std::int64_t hcom_fusion = 1;
                    GE_IF_BOOL_EXEC(!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_FUSION, hcom_fusion),
                                    GELOGW("FusionAllReducePass: not get hcom_fusion from opDescPtr "
                                           "by HCOM_ATTR_FUSION"));
                    GELOGI("after GetInt, hcom_fusion is :%ld", hcom_fusion); GE_IF_BOOL_EXEC(
                      hcom_fusion > 0, fusionOps.push_back(nodePtr); inputGradientSize.push_back(inputGradientSizeTemp);
                      inputGradientTime.push_back(inputGradientTimeTemp);))
  }
  // The number of allredecue operator must be more than 1
  GE_IF_BOOL_EXEC(1 >= fusionOps.size(), GELOGW("FusionAllReducePass NOT_CHANGED: the graph has "
                                                "%lu allreduce operator",
                                                fusionOps.size());
                  return NOT_CHANGED;);

  string group = "group";
  u32 gradientNum = fusionOps.size();
  string model_name_str = graph->GetName();
  const char *model_name = model_name_str.c_str();
  model_feature modelFeature{model_name, gradientNum, inputGradientSize.data(), inputGradientTime.data()};

  u32 segmentNum = 0;
  u32 segmentIndex[HCCL_MAX_SEGMENT_NUM] = {};

  // Call HCCL function: hcom_gradient_segment
  GELOGI("FusionAllReducePass: invoking hcom_get_split_strategy");
  GE_IF_BOOL_EXEC(HCCL_SUCCESS != hcom_get_split_strategy(group.c_str(), &modelFeature, HCCL_MAX_SEGMENT_NUM,
                                                          &segmentNum, segmentIndex),
                  GELOGE(FAILED, "FusionAllReducePass FAILED: the graph has %lu allreduce operator", fusionOps.size());
                  return FAILED;)
  GELOGI("FusionAllReducePass: invoke hcom_get_split_strategy successfully");

  // check whether segmentNum is legal or not
  GE_IF_BOOL_EXEC((HCCL_MAX_SEGMENT_NUM < segmentNum || 1 > segmentNum || segmentNum > gradientNum),
                  GELOGE(FAILED,
                         "FusionAllReducePass FAILED: illegal segmentNum=%u, "
                         "HCCL_MAX_SEGMENT_NUM=%u, gradientNum=%u",
                         segmentNum, HCCL_MAX_SEGMENT_NUM, gradientNum);
                  return FAILED;);

  // check whether segmentIndex is legal or not
  GE_IF_BOOL_EXEC((segmentIndex[segmentNum - 1] != gradientNum - 1),
                  GELOGE(FAILED,
                         "FusionAllReducePass FAILED: illegal segmentIndex[0]=%u, "
                         "segmentIndex[segmentNum-1]=%u, gradientNum=%u",
                         segmentIndex[0], segmentIndex[(segmentNum)-1], gradientNum);
                  return FAILED;);

  for (uint32_t i = 0; i < segmentNum - 1; i++) {
    GE_IF_BOOL_EXEC(segmentIndex[i] >= segmentIndex[i + 1], GELOGE(FAILED,
                                                                   "FusionAllReducePass FAILED: illegal "
                                                                   "segmentIndex[%u]=%u, segmentIndex[%u]=%u",
                                                                   i, segmentIndex[i], i + 1, segmentIndex[i + 1]);
                    return FAILED;);
  }

  // check whether fusion is needed or not
  GE_IF_BOOL_EXEC(
    segmentNum == gradientNum,
    GELOGE(NOT_CHANGED, "FusionAllReducePass NOT_CHANGED: segmentNum=%u, gradientNum=%u", segmentNum, gradientNum);
    return NOT_CHANGED;)

  std::unordered_set<void *> anchorPtrSet;
  std::vector<ge::OutDataAnchorPtr> fusionOpPeerOutDataAnchor;
  std::vector<ge::OutDataAnchorPtr> fusionOpPeerOutDataToInControl;
  std::vector<ge::OutControlAnchorPtr> fusionOpPeerOutControlAnchor;
  std::vector<std::pair<int, ge::InDataAnchorPtr>> fusionOpPeerInDataAnchor;
  std::vector<std::pair<int, ge::InControlAnchorPtr>> fusionOpPeerInControlFromOutData;
  std::vector<ge::InControlAnchorPtr> fusionOpPeerInControlAnchor;
  ge::OutControlAnchorPtr previousNewAllreduceOutControlAnchor = nullptr;

  // Traversing the segmentNum
  uint32_t start = 0;
  uint32_t end = 0;
  for (uint32_t segmentIdx = 0; segmentIdx < segmentNum; segmentIdx++) {
    end = segmentIndex[segmentIdx];
    GE_IF_BOOL_EXEC(end - start < 1,
                    GELOGI("FusionAllReducePass: segmentIndex[%u]=%u", segmentIdx, segmentIndex[segmentIdx]);
                    start = end + 1; continue;);

    ge::OpDescPtr originDescPtr = fusionOps[start]->GetOpDesc();
    GE_CHECK_NOTNULL(originDescPtr);
    ge::OpDescPtr newAllreduceDesc = AttrUtils::CloneOpDesc(originDescPtr);
    GE_CHECK_NOTNULL(newAllreduceDesc);

    // Cleat buffer
    anchorPtrSet.clear();
    fusionOpPeerOutDataAnchor.clear();
    fusionOpPeerOutDataToInControl.clear();
    fusionOpPeerOutControlAnchor.clear();
    fusionOpPeerInDataAnchor.clear();
    fusionOpPeerInControlFromOutData.clear();
    fusionOpPeerInControlAnchor.clear();

    // Traversing the Allreduce operators of each group
    int outDataAnchorIndex = 0;
    GE_CHK_STATUS_RET(GetPeerOutDataToInData(anchorPtrSet, fusionOpPeerOutDataAnchor, fusionOps[start]),
                      "Get peer outDataAnchor to inDataAnchor failed");

    GE_CHK_STATUS_RET(GetPeerInAnchorToOutData(anchorPtrSet, fusionOpPeerInDataAnchor, fusionOpPeerInControlFromOutData,
                                               fusionOps[start]),
                      "Get peer inDataAnchor and inControlAnchor to outDataAnchor failed");

    GE_CHK_STATUS_RET(GetPeerOutDataToInControl(anchorPtrSet, fusionOpPeerOutDataToInControl, fusionOps[start]),
                      "Get peer outDataAnchor to inControlAnchor failed");
    GE_CHK_STATUS_RET(GetPeerOutControlToInControl(anchorPtrSet, fusionOpPeerOutControlAnchor, fusionOps[start]),
                      "Get peer outControlAnchor to inControlAnchor failed");
    GE_CHK_STATUS_RET(GetPeerInControlFromOutControl(anchorPtrSet, fusionOpPeerInControlAnchor, fusionOps[start]),
                      "Get peer outControlAnchor from inControlAnchor failed");
    GE_CHK_STATUS_RET(graph->RemoveNode(fusionOps[start]), "FusionAllReducePass FAILED: remove node %s\n.",
                      fusionOps[start]->GetName().c_str());

    for (uint32_t idx = start + 1; idx <= end; idx++) {
      GE_CHK_STATUS_RET(
        GetPeerOutDataToInData(anchorPtrSet, fusionOpPeerOutDataAnchor, fusionOps[idx], newAllreduceDesc),
        "Get peer outDataAnchor to inDataAnchor failed");
      GE_CHK_STATUS_RET(GetPeerOutDataToInControl(anchorPtrSet, fusionOpPeerOutDataToInControl, fusionOps[idx]),
                        "Get peer outDataAnchor to inControlAnchor failed");
      GE_CHK_STATUS_RET(GetPeerOutControlToInControl(anchorPtrSet, fusionOpPeerOutControlAnchor, fusionOps[idx]),
                        "Get peer outControlAnchor to inControlAnchor failed");
      GE_CHK_STATUS_RET(
        GetPeerAnchorFromOutData(anchorPtrSet, fusionOpPeerInDataAnchor, fusionOpPeerInControlFromOutData,
                                 fusionOps[idx], newAllreduceDesc, outDataAnchorIndex),
        "Get peerAnchor from outDataAnchor failed");
      GE_CHK_STATUS_RET(GetPeerInControlFromOutControl(anchorPtrSet, fusionOpPeerInControlAnchor, fusionOps[idx]),
                        "Get peer outControlAnchor from inControlAnchor failed");

      // Delete the node
      GE_CHK_STATUS_RET(graph->RemoveNode(fusionOps[idx]), "FusionAllReducePass FAILED: remove node %s\n.",
                        fusionOps[idx]->GetName().c_str());
    }

    NodePtr newAllReducePtr = graph->AddNode(newAllreduceDesc);
    GE_CHECK_NOTNULL(newAllReducePtr);
    // Link the inputDataAnchor
    for (uint32_t i = 0; i < fusionOpPeerOutDataAnchor.size(); i++) {
      GE_CHK_STATUS_RET(
        GraphUtils::AddEdge(fusionOpPeerOutDataAnchor[i], newAllReducePtr->GetInDataAnchor(static_cast<int>(i))),
        "FusionAllReducePass FAILED: add input data edge failed");
    }

    // Link the inputControlAnchor
    for (uint32_t i = 0; i < fusionOpPeerOutControlAnchor.size(); i++) {
      GE_CHK_STATUS_RET(GraphUtils::AddEdge(fusionOpPeerOutControlAnchor[i], newAllReducePtr->GetInControlAnchor()),
                        "FusionAllReducePass FAILED: add input control edge failed");
    }

    for (uint32_t i = 0; i < fusionOpPeerOutDataToInControl.size(); i++) {
      GE_CHK_STATUS_RET(GraphUtils::AddEdge(fusionOpPeerOutDataToInControl[i], newAllReducePtr->GetInControlAnchor()),
                        "FusionAllReducePass FAILED: add edge from out data to incontrol "
                        "failed");
    }

    // Link the outputDataAnchor
    for (uint32_t i = 0; i < fusionOpPeerInDataAnchor.size(); i++) {
      auto peerInDataAnchor = fusionOpPeerInDataAnchor[i].second;
      GE_CHK_STATUS_RET(
        GraphUtils::AddEdge(newAllReducePtr->GetOutDataAnchor(fusionOpPeerInDataAnchor[i].first), peerInDataAnchor),
        "FusionAllReducePass FAILED: add output data edge failed");
    }
    for (uint32_t i = 0; i < fusionOpPeerInControlFromOutData.size(); i++) {
      auto peerInControlAnchor = fusionOpPeerInControlFromOutData[i].second;
      GE_CHK_STATUS_RET(
        GraphUtils::AddEdge(newAllReducePtr->GetOutDataAnchor(fusionOpPeerInControlFromOutData[i].first),
                            peerInControlAnchor),
        "FusionAllReducePass FAILED: add edge from out data to in control "
        "failed");
    }

    // Link the outputControlAnchor
    for (uint32_t i = 0; i < fusionOpPeerInControlAnchor.size(); i++) {
      GE_CHK_STATUS_RET(GraphUtils::AddEdge(newAllReducePtr->GetOutControlAnchor(), fusionOpPeerInControlAnchor[i]),
                        "FusionAllReducePass FAILED: add output control edge failed");
    }

    // Link the newAllreduce
    if (segmentIdx > 0 && previousNewAllreduceOutControlAnchor != nullptr) {
      GE_CHK_STATUS_RET(
        GraphUtils::AddEdge(previousNewAllreduceOutControlAnchor, newAllReducePtr->GetInControlAnchor()),
        "FusionAllReducePass FAILED: add input previous control edge failed");
    }

    previousNewAllreduceOutControlAnchor = newAllReducePtr->GetOutControlAnchor();
    start = end + 1;
  }

  return SUCCESS;
}

Status AllReducePass::GetPeerOutDataToInData(std::unordered_set<void *> &anchorSet,
                                             vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec,
                                             ge::NodePtr &srcNodePtr) {
  for (auto inDataAnchor : srcNodePtr->GetAllInDataAnchors()) {
    GE_IF_BOOL_EXEC(inDataAnchor == nullptr, continue;);
    OutDataAnchorPtr peerOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peerOutDataAnchor == nullptr, continue;);
    if (anchorSet.count(peerOutDataAnchor.get()) == 0) {
      peerOutDataAnchorVec.push_back(peerOutDataAnchor);
      anchorSet.insert(peerOutDataAnchor.get());
      GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(peerOutDataAnchor, inDataAnchor));
    }
  }
  return SUCCESS;
}

Status AllReducePass::GetPeerInAnchorToOutData(
  std::unordered_set<void *> &anchorSet, std::vector<std::pair<int, ge::InDataAnchorPtr>> &fusionOpPeerInDataAnchor,
  std::vector<std::pair<int, ge::InControlAnchorPtr>> &fusionOpPeerInControlFromOutData, ge::NodePtr &srcNodePtr) {
  for (auto outDataAnchor : srcNodePtr->GetAllOutDataAnchors()) {
    GE_IF_BOOL_EXEC(outDataAnchor == nullptr, continue;);
    for (auto peerInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(peerInDataAnchor == nullptr, continue;);
      if (anchorSet.count(peerInDataAnchor.get()) == 0) {
        std::pair<int, ge::InDataAnchorPtr> pairPeerInDataAnchor;
        pairPeerInDataAnchor.first = 0;
        pairPeerInDataAnchor.second = peerInDataAnchor;
        fusionOpPeerInDataAnchor.push_back(pairPeerInDataAnchor);
        anchorSet.insert(peerInDataAnchor.get());
        GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(outDataAnchor, peerInDataAnchor));
      }
    }

    for (auto peerInControlAnchorFromData : outDataAnchor->GetPeerInControlAnchors()) {
      GE_IF_BOOL_EXEC(peerInControlAnchorFromData == nullptr, continue;);
      if (anchorSet.count(peerInControlAnchorFromData.get()) == 0) {
        std::pair<uint32_t, ge::InControlAnchorPtr> pairPeerInControlAnchorFromData;
        pairPeerInControlAnchorFromData.first = 0;
        pairPeerInControlAnchorFromData.second = peerInControlAnchorFromData;
        fusionOpPeerInControlFromOutData.push_back(pairPeerInControlAnchorFromData);
        anchorSet.insert(peerInControlAnchorFromData.get());
        GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(outDataAnchor, peerInControlAnchorFromData));
      }
    }
  }
  return SUCCESS;
}

Status AllReducePass::GetPeerOutDataToInData(std::unordered_set<void *> &anchorSet,
                                             vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec,
                                             ge::NodePtr &srcNodePtr, ge::OpDescPtr &dstOpDescPtr) {
  for (auto inDataAnchor : srcNodePtr->GetAllInDataAnchors()) {
    GE_IF_BOOL_EXEC(inDataAnchor == nullptr, continue;);
    OutDataAnchorPtr peerOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peerOutDataAnchor == nullptr, continue;);
    if (anchorSet.count(peerOutDataAnchor.get()) == 0) {
      peerOutDataAnchorVec.push_back(peerOutDataAnchor);
      anchorSet.insert(peerOutDataAnchor.get());
      if (dstOpDescPtr->AddInputDesc(inDataAnchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(inDataAnchor->GetIdx())) !=
          ge::GRAPH_SUCCESS) {
        GELOGW("GetPeerOutDataToInData: AddInputDesc failed");
      }
      GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(peerOutDataAnchor, inDataAnchor));
    }
  }
  return SUCCESS;
}

Status AllReducePass::GetPeerOutDataToInControl(std::unordered_set<void *> &anchorSet,
                                                vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec,
                                                ge::NodePtr &srcNodePtr) {
  InControlAnchorPtr inControlAnchor = srcNodePtr->GetInControlAnchor();
  GE_CHECK_NOTNULL(inControlAnchor);
  for (auto peerOutDataToInControl : inControlAnchor->GetPeerOutDataAnchors()) {
    GE_IF_BOOL_EXEC(peerOutDataToInControl == nullptr, continue;);
    if (anchorSet.count(peerOutDataToInControl.get()) == 0) {
      peerOutDataToInControlVec.push_back(peerOutDataToInControl);
      anchorSet.insert(peerOutDataToInControl.get());
      GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(peerOutDataToInControl, inControlAnchor));
    }
  }
  return SUCCESS;
}

Status AllReducePass::GetPeerOutControlToInControl(std::unordered_set<void *> &anchorSet,
                                                   vector<ge::OutControlAnchorPtr> &peerOutControlToInControlVec,
                                                   ge::NodePtr &srcNodePtr) {
  InControlAnchorPtr inControlAnchor = srcNodePtr->GetInControlAnchor();
  GE_CHECK_NOTNULL(inControlAnchor);
  for (auto peerOutControlAnchor : inControlAnchor->GetPeerOutControlAnchors()) {
    GE_IF_BOOL_EXEC(peerOutControlAnchor == nullptr, continue;);
    if (anchorSet.count(peerOutControlAnchor.get()) == 0) {
      peerOutControlToInControlVec.push_back(peerOutControlAnchor);
      anchorSet.insert(peerOutControlAnchor.get());
      GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(peerOutControlAnchor, inControlAnchor));
    }
  }
  return SUCCESS;
}

Status AllReducePass::GetPeerAnchorFromOutData(
  std::unordered_set<void *> &anchorSet, vector<std::pair<int, ge::InDataAnchorPtr>> &peerInDataFromOutDataVec,
  vector<std::pair<int, ge::InControlAnchorPtr>> &peerInControlFromOutDataVec, ge::NodePtr &srcNodePtr,
  ge::OpDescPtr &dstOpDescPtr, int &index) {
  for (auto outDataAnchor : srcNodePtr->GetAllOutDataAnchors()) {
    GE_IF_BOOL_EXEC(outDataAnchor == nullptr, continue;)
    if (outDataAnchor->GetPeerInDataAnchors().size() > 0 || outDataAnchor->GetPeerInControlAnchors().size() > 0) {
      if (dstOpDescPtr->AddOutputDesc(
            outDataAnchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(outDataAnchor->GetIdx())) != ge::GRAPH_SUCCESS) {
        GELOGW("GetPeerAnchorFromOutData: AddOutputDesc failed");
      }
      index++;
    }

    for (auto peerInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(peerInDataAnchor == nullptr, continue;)
      if (anchorSet.count(peerInDataAnchor.get()) == 0) {
        std::pair<int, ge::InDataAnchorPtr> pairPeerInDataAnchor;
        pairPeerInDataAnchor.first = index;
        pairPeerInDataAnchor.second = peerInDataAnchor;
        peerInDataFromOutDataVec.push_back(pairPeerInDataAnchor);
        anchorSet.insert(peerInDataAnchor.get());
        GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(outDataAnchor, peerInDataAnchor))
      }
    }

    for (auto peerInControlAnchorFromData : outDataAnchor->GetPeerInControlAnchors()) {
      GE_IF_BOOL_EXEC(peerInControlAnchorFromData == nullptr, continue;)
      if (anchorSet.count(peerInControlAnchorFromData.get()) == 0) {
        std::pair<int, ge::InControlAnchorPtr> pairPeerInControlAnchorFromData;
        pairPeerInControlAnchorFromData.first = index;
        pairPeerInControlAnchorFromData.second = peerInControlAnchorFromData;
        peerInControlFromOutDataVec.push_back(pairPeerInControlAnchorFromData);
        anchorSet.insert(peerInControlAnchorFromData.get());
        GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(outDataAnchor, peerInControlAnchorFromData))
      }
    }
  }
  return SUCCESS;
}

Status AllReducePass::GetPeerInControlFromOutControl(std::unordered_set<void *> &anchorSet,
                                                     vector<ge::InControlAnchorPtr> &peerInControlFromOutControlVec,
                                                     ge::NodePtr &srcNodePtr) {
  OutControlAnchorPtr outControlAnchor = srcNodePtr->GetOutControlAnchor();
  GE_CHECK_NOTNULL(outControlAnchor);
  for (auto peerInControlAnchor : outControlAnchor->GetPeerInControlAnchors()) {
    GE_IF_BOOL_EXEC(peerInControlAnchor == nullptr, continue;)
    if (anchorSet.count(peerInControlAnchor.get()) == 0) {
      peerInControlFromOutControlVec.push_back(peerInControlAnchor);
      anchorSet.insert(peerInControlAnchor.get());
      GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(outControlAnchor, peerInControlAnchor))
    }
  }
  return SUCCESS;
}
}  // namespace ge
