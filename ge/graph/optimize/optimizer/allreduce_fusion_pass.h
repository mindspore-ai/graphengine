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

#ifndef GE_GRAPH_OPTIMIZE_OPTIMIZER_ALLREDUCE_FUSION_PASS_H_
#define GE_GRAPH_OPTIMIZE_OPTIMIZER_ALLREDUCE_FUSION_PASS_H_

#include <unordered_set>
#include <utility>
#include <vector>
#include "inc/graph_pass.h"

namespace ge {
//
class AllReducePass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  Status GetPeerOutDataToInData(std::unordered_set<void *> &anchorSet,
                                vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec, ge::NodePtr &srcNodePtr,
                                ge::OpDescPtr &dstOpDescPtr);
  Status GetPeerOutDataToInControl(std::unordered_set<void *> &anchorSet,
                                   vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec, ge::NodePtr &srcNodePtr);
  Status GetPeerOutControlToInControl(std::unordered_set<void *> &anchorSet,
                                      vector<ge::OutControlAnchorPtr> &peerOutControlToInControlVec,
                                      ge::NodePtr &srcNodePtr);
  Status GetPeerAnchorFromOutData(std::unordered_set<void *> &anchorSet,
                                  vector<std::pair<int, ge::InDataAnchorPtr>> &peerInDataFromOutDataVec,
                                  vector<std::pair<int, ge::InControlAnchorPtr>> &peerInControlFromOutDataVec,
                                  ge::NodePtr &srcNodePtr, ge::OpDescPtr &dstOpDescPtr, int &index);
  Status GetPeerInControlFromOutControl(std::unordered_set<void *> &anchorSet,
                                        vector<ge::InControlAnchorPtr> &peerInControlFromOutControlVec,
                                        ge::NodePtr &srcNodePtr);
  Status GetPeerOutDataToInData(std::unordered_set<void *> &anchorSet,
                                std::vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec,
                                ge::NodePtr &srcNodePtr);
  Status GetPeerInAnchorToOutData(std::unordered_set<void *> &anchorSet,
                                  std::vector<std::pair<int, ge::InDataAnchorPtr>> &fusionOpPeerInDataAnchor,
                                  std::vector<std::pair<int, ge::InControlAnchorPtr>>&fusionOpPeerInControlFromOutData,
                                  ge::NodePtr &srcNodePtr);
};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_OPTIMIZER_ALLREDUCE_FUSION_PASS_H_
