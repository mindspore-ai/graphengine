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

#ifndef GE_GRAPH_BUILD_STREAM_ALLOCATOR_H_
#define GE_GRAPH_BUILD_STREAM_ALLOCATOR_H_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "engine_manager/dnnengine_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
class StreamAllocator {
 public:
  StreamAllocator(ComputeGraphPtr whole_graph, const Graph2SubGraphInfoList &subgraphs);
  StreamAllocator(const StreamAllocator &) = delete;
  StreamAllocator &operator=(const StreamAllocator &) = delete;
  ~StreamAllocator() = default;

  Status AssignLogicalStreams(const std::map<std::string, int> &max_parallel_num, bool hcom_parallel);
  Status RefreshRealStream(int64_t &stream_num, int64_t &event_num);
  const vector<int64_t> &GetHugeStreams() const { return huge_streams_; }

 private:
  Status SplitStreams(std::vector<std::set<int64_t>> &split_streams);

  Status AssignSingleStream();
  Status SetActiveStreamsByLabel();
  Status UpdateActiveStreams(const std::vector<std::set<int64_t>> &splited_streams);
  void UpdateLabelStreams(const std::vector<std::set<int64_t>> &split_streams);
  Status SetActiveStreamsForSubgraph();
  Status SetActiveStreamsForLoop();
  Status CheckStreamActived() const;
  Status GetMaxStreamAndTask(bool huge_stream, uint32_t &max_stream_count, uint32_t &max_task_count);
  int64_t GetMaxNodeNumPerStream(const NodePtr &node, uint32_t max_node_num_one_stream);

  Status InsertSyncEvents();
  Status InsertOneEventInTwoNodes(const NodePtr &cur_node_ptr, const NodePtr &next_node_ptr);

  Status OptimizeSyncEvents();
  Status OptimizeBySendEvents(const std::map<int64_t, std::vector<NodePtr>> &stream_nodes);
  Status OptimizeByRecvEvents(const std::map<int64_t, std::vector<NodePtr>> &stream_nodes);
  Status OptimizeByStreamActivate();

  Status RefreshContinuousEvents();
  Status InsertSyncEventNodes();
  Status ReorderEventNodes() const;

  Status InsertActiveNodesAfterSwitch(NodePtr &switch_node);
  Status InsertActiveNodesAfterSwitch(NodePtr &switch_nodes, std::vector<NodePtr> &switch_active_nodes);
  Status SetActiveStreamList(NodePtr &active_node, const std::string &active_label);
  Status AddActiveNodes(NodePtr &switch_node, const std::vector<std::string> &ori_active_label_list,
                        std::vector<std::string> &active_label_list, std::vector<NodePtr> &added_active_nodes);

  Status AddActiveEntryStream();
  Status CollectDeactiveStream(const OpDescPtr &op_desc, std::set<uint32_t> &deactive_streams) const;
  Status InsertActiveEntryStream(const std::vector<uint32_t> &active_streams, int64_t stream_id);

  Status AddEventId(const NodePtr &pre_node, const NodePtr &not_cur, const NodePtr &cur_node, bool not_use_cur);
  void AddSendEventId(const NodePtr &node, uint32_t event_id);
  void AddRecvEventId(const NodePtr &node, uint32_t event_id);
  void RmvSendEventId(const NodePtr &node, uint32_t event_id);
  void RmvRecvEventId(const NodePtr &node, uint32_t event_id);
  void GetSendEventIdList(const NodePtr &node, std::vector<uint32_t> &send_list) const;
  void GetRecvEventIdList(const NodePtr &node, std::vector<uint32_t> &recv_list) const;
  NodePtr GetNodeFromSendEventId(uint32_t send_event_id) const;
  NodePtr GetNodeFromRecvEventId(uint32_t recv_event_id) const;

  void DumpEvents();
  // Determine if the successor node of RecvNode is directly or indirectly activated by the SendNode precursor node
  bool IsRecvNodeActivatedBySendNode(const NodePtr &send_node_ptr, const NodePtr &recv_node_ptr) const;

  ComputeGraphPtr whole_graph_;
  const Graph2SubGraphInfoList &subgraphs_;

  int64_t stream_num_{0};
  uint32_t event_num_{0};
  bool enable_single_stream_{false};
  vector<int64_t> huge_streams_;

  // <stream label, set<stream id>>
  std::map<string, std::set<int64_t>> labeled_streams_;
  std::map<std::string, std::set<NodePtr>> specific_activated_labels_;
  std::set<int64_t> specific_activated_streams_;
  std::map<int64_t, std::set<NodePtr>> specific_activated_streams_nodes_map_;

  // send events corresponding to the node
  std::map<NodePtr, std::vector<uint32_t>> node_to_send_events_;

  // recv events corresponding to the node
  std::map<NodePtr, std::vector<uint32_t>> node_to_recv_events_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_STREAM_ALLOCATOR_H_
