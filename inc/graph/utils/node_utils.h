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

#ifndef INC_GRAPH_UTILS_NODE_UTILS_H_
#define INC_GRAPH_UTILS_NODE_UTILS_H_

#include <map>
#include <vector>
#include "graph/node.h"
namespace ge {
class NodeUtils {
 public:
  static graphStatus AddSendEventId(const NodePtr &node, const uint32_t &event_id);
  static graphStatus AddRecvEventId(const NodePtr &node, const uint32_t &event_id);
  static graphStatus GetSendEventIdList(const NodePtr &node, std::vector<uint32_t> &vec_send);
  static graphStatus GetRecvEventIdList(const NodePtr &node, std::vector<uint32_t> &vec_recv);

  static graphStatus ClearSendInfo();
  static graphStatus ClearRecvInfo();

  static graphStatus GetSingleOutputNodeOfNthLayer(const NodePtr &src, int depth, NodePtr &dst);

  static graphStatus GetDataOutAnchorAndControlInAnchor(const NodePtr &node_ptr, OutDataAnchorPtr &out_data,
                                                        InControlAnchorPtr &in_control);

  static graphStatus ClearInDataAnchor(const NodePtr &node_ptr, const InDataAnchorPtr &in_data_anchor);
  static graphStatus SetAllAnchorStatus(const NodePtr &nodePtr);
  static graphStatus SetAllAnchorStatus(Node &node);
  static bool IsAnchorStatusSet(const NodePtr &nodePtr);
  static bool IsAnchorStatusSet(const Node &node);

  static graphStatus MoveOutputEdges(const NodePtr &origin_node, const NodePtr &new_node);

  static void UpdateIsInputConst(const NodePtr &nodePtr);
  static void UpdateIsInputConst(Node &node);
  static bool IsConst(const Node &node);
  static void UnlinkAll(const Node &node);
  static graphStatus UpdatePeerNodeInputDesc(const NodePtr &node_ptr);

  static bool IsInNodesEmpty(const Node &node);
  static GeTensorDesc GetOutputDesc(const Node &node, uint32_t index);
  static GeTensorDesc GetInputDesc(const Node &node, uint32_t index);
  static graphStatus UpdateOutputShape(const Node &node, uint32_t index, const GeShape &shape);
  static graphStatus UpdateInputShape(const Node &node, uint32_t index, const GeShape &shape);

 private:
  static std::map<NodePtr, std::vector<uint32_t>> map_send_info_;
  static std::map<NodePtr, std::vector<uint32_t>> map_recv_info_;
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_NODE_UTILS_H_
