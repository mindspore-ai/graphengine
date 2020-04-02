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

#include "graph/build/stream_allocator.h"
#include <memory>
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "init/gelib.h"

#include "graph/build/logical_stream_allocator.h"

using std::map;
using std::set;
using std::string;
using std::vector;

namespace {
const int64_t kMaxNodeNumInNormalStream = 350;
const int64_t kMaxNodeNumInHcomStream = 5;

const uint32_t kMaxSwitchStreamNum = 1;
}  // namespace

namespace ge {
Status StreamAllocator::AssignLogicalStreams(const std::map<std::string, int> &max_parallel_num, bool hcom_parallel) {
  GELOGI("AssignLogicalStreams start.");
  GE_CHECK_NOTNULL(whole_graph_);
  GraphUtils::DumpGEGraph(whole_graph_, "BeforeAssignedLogicalStreams_whole_graph");
  GraphUtils::DumpGEGraphToOnnx(*whole_graph_, "BeforeAssignedLogicalStreams_whole_graph");

  auto gelib = GELib::GetInstance();
  if (gelib == nullptr) {
    GELOGE(FAILED, "Get GELib instance failed.");
    return FAILED;
  }

  const map<string, SchedulerConf> &scheduler_confs = gelib->DNNEngineManagerObj().GetSchedulers();

  LogicalStreamAllocator logical_allocator(scheduler_confs, max_parallel_num, hcom_parallel);
  Status status = logical_allocator.Assign(whole_graph_, subgraphs_, stream_num_);
  if (status != SUCCESS) {
    GELOGE(status, "Assign logical streams failed.");
    return status;
  }

  GraphUtils::DumpGEGraph(whole_graph_, "AfterAssignedLogicalStreams_whole_graph");
  GraphUtils::DumpGEGraphToOnnx(*whole_graph_, "AfterAssignedLogicalStreams_whole_graph");
  GELOGI("AssignLogicalStreams success.");

  return SUCCESS;
}

// After allocating the logical stream in the graph, refresh the stream in the
// graph and insert the synchronization node.
Status StreamAllocator::RefreshRealStream(int64_t &stream_num, int64_t &event_num) {
  GELOGI("RefreshRealStream start.");
  GE_CHECK_NOTNULL(whole_graph_);
  Status status = ActiveStreamsBySpecificLabels();
  if (status != SUCCESS) {
    GELOGE(status, "ActiveStreams failed!");
    return status;
  }

  status = InsertSyncEvents();
  if (status != SUCCESS) {
    GELOGE(status, "InsertSyncEventId failed!");
    return status;
  }

  status = OptimizeSyncEvents();
  if (status != SUCCESS) {
    GELOGE(status, "OptimizeSyncEventId failed!");
    return status;
  }

  status = SplitStreams();
  if (status != SUCCESS) {
    GELOGE(status, "SplitStreams failed!");
    return status;
  }

  status = ActiveStreamsForLoop();
  if (status != SUCCESS) {
    GELOGE(status, "ActiveStreamsForLoop failed!");
    return status;
  }

  status = AddActiveEntryStream();
  if (status != SUCCESS) {
    GELOGE(status, "AddActiveEntryStream failed!");
    return status;
  }

  status = RefreshContinuousEvents();
  if (status != SUCCESS) {
    GELOGE(status, "RefreshContinuousEvents failed!");
    return status;
  }

  status = InsertSyncEventNodes();
  if (status != SUCCESS) {
    GELOGE(status, "InsertSyncEventNode failed!");
    return status;
  }

  DumpEvents();
  GraphUtils::DumpGEGraph(whole_graph_, "RefreshRealStream");
  GraphUtils::DumpGEGraphToOnnx(*whole_graph_, "RefreshRealStream");

  for (const NodePtr &node : whole_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    auto stream_id = node->GetOpDesc()->GetStreamId();
    if (stream_id == kInvalidStream) {
      node->GetOpDesc()->SetStreamId(0);
    }
  }

  if (stream_num_ == 0) {
    GELOGI("None of nodes need to assign stream, stream num is 0, it will cause error, so change it to 1");
    stream_num_ = 1;
  }
  GELOGI("stream_num_: %ld, event_num_: %u.", stream_num_, event_num_);
  GELOGI("RefreshRealStream successfully.");

  stream_num = stream_num_;
  event_num = static_cast<int64_t>(event_num_);

  return SUCCESS;
}

// Split the stream according to the maximum number of nodes in the stream.
Status StreamAllocator::SplitStreams() {
  if (stream_num_ == 0) {
    GELOGI("stream_num_ is 0");
    return SUCCESS;
  }

  // stream_node_num_vec records the number of all nodes on each stream
  // added_stream_num_vec records the number of streams that each stream needs to increase
  // new_stream_id_vec records the new physical stream id for each stream
  vector<int64_t> stream_node_num_vec(stream_num_);
  vector<int64_t> added_stream_num_vec(stream_num_);
  vector<int64_t> new_stream_id_vec(stream_num_);
  vector<NodePtr> pre_node_vec(stream_num_);
  vector<set<int64_t>> split_streams(stream_num_);

  int64_t last_stream_id = stream_num_ - 1;
  for (auto i = 0; i <= last_stream_id; i++) {
    stream_node_num_vec[i] = 0;
    added_stream_num_vec[i] = 0;
    new_stream_id_vec[i] = i;
    pre_node_vec[i] = nullptr;
  }

  for (const auto &cur_node : whole_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(cur_node->GetOpDesc());
    int64_t stream_id = cur_node->GetOpDesc()->GetStreamId();
    if (stream_id == kInvalidStream) {
      continue;
    }
    if (stream_id > last_stream_id) {
      GELOGE(FAILED, "SplitStreams:streamid(%ld) > last_stream_id(%ld)", stream_id, last_stream_id);
      return FAILED;
    }
    stream_node_num_vec[stream_id]++;

    // The maximum number of tasks per stream.
    int64_t max_node_num_one_stream = kMaxNodeNumInNormalStream;
    const string op_type = cur_node->GetType();
    if ((op_type == HCOMBROADCAST) || (op_type == HCOMALLGATHER) || (op_type == HCOMALLREDUCE) ||
        (op_type == HCOMREDUCESCATTER)) {
      max_node_num_one_stream = kMaxNodeNumInHcomStream;
    }

    // Split the stream if it exceeds the maximum number of nodes in the stream.
    if (stream_node_num_vec[stream_id] > max_node_num_one_stream) {
      last_stream_id++;
      GELOGI(
        "stream_node_num_vec[%ld]= %ld > max_node_num_one_stream : %ld, "
        "It's time to split the stream, split newly-added stream id is %ld",
        stream_id, stream_node_num_vec[stream_id], max_node_num_one_stream, last_stream_id);

      stream_node_num_vec[stream_id] = 1;
      added_stream_num_vec[stream_id]++;
      new_stream_id_vec[stream_id] = last_stream_id;
      split_streams[stream_id].emplace(last_stream_id);

      // Add the send/recv event to the first and last nodes of the split stream.
      NodePtr pre_node = pre_node_vec[stream_id];
      if (pre_node != nullptr) {
        GELOGI("Add send event %u for node %s", event_num_, pre_node->GetName().c_str());
        GELOGI("Add recv event %u for node %s", event_num_, cur_node->GetName().c_str());
        AddSendEventId(pre_node, event_num_);
        AddRecvEventId(cur_node, event_num_);
        ++event_num_;
      }
    }

    /// If the split stream num is greater than 1, the node behind the same
    /// stream must reset the new stream id.
    if (added_stream_num_vec[stream_id] >= 1) {
      cur_node->GetOpDesc()->SetStreamId(new_stream_id_vec[stream_id]);
    }

    pre_node_vec[stream_id] = cur_node;
  }

  if (last_stream_id >= 0) {
    stream_num_ = last_stream_id + 1;
  }

  return UpdateActiveStreams(split_streams);
}

Status StreamAllocator::UpdateActiveStreams(vector<set<int64_t>> &split_streams) {
  for (const auto &node : whole_graph_->GetDirectNode()) {
    vector<uint32_t> active_streams;
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams)) {
      vector<uint32_t> new_active_streams = active_streams;
      for (const uint32_t logical_stream : active_streams) {
        if (static_cast<size_t>(logical_stream) >= split_streams.size()) {
          GELOGE(FAILED, "logical stream is out of range.");
          return FAILED;
        }
        const set<int64_t> &new_split_streams = split_streams[logical_stream];
        if (!new_split_streams.empty()) {
          for (int64_t split_stream : new_split_streams) {
            specific_activated_streams_.emplace(split_stream);
            new_active_streams.emplace_back(static_cast<uint32_t>(split_stream));
          }
        }
      }
      if (!AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, new_active_streams)) {
        GELOGE(FAILED, "UpdateActiveStreams failed, node name : (%s).", node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status StreamAllocator::ActiveStreamsBySpecificLabels() {
  // <stream label, set<stream id>>
  map<string, set<int64_t>> labeled_streams;
  for (const auto &node : whole_graph_->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    string stream_label;
    if (AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label) && !stream_label.empty()) {
      int64_t stream_id = op_desc->GetStreamId();
      if (stream_id != kInvalidStream) {
        labeled_streams[stream_label].emplace(stream_id);
      }
    }
  }

  for (const auto &node : whole_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    vector<string> activated_label_list;
    if (!AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, activated_label_list) ||
        activated_label_list.empty()) {
      continue;
    }

    vector<uint32_t> activated_stream_list;
    for (string &activated_label : activated_label_list) {
      specific_activated_labels_[activated_label].emplace(node);
      for (int64_t activated_stream : labeled_streams[activated_label]) {
        activated_stream_list.push_back(static_cast<uint32_t>(activated_stream));
        specific_activated_streams_.emplace(activated_stream);
        specific_activated_streams_nodes_map_[activated_stream].emplace(node);
        GELOGI("Node %s active stream %ld by %s.", node->GetName().c_str(), activated_stream, activated_label.c_str());
      }
    }
    GE_CHK_BOOL_EXEC(AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, activated_stream_list),
                     GELOGE(FAILED, "SetListInt failed.");
                     return FAILED);
  }

  return SUCCESS;
}

Status StreamAllocator::ActiveStreamsForLoop() {
  vector<uint32_t> loop_active_streams;
  for (int64_t stream_id = 0; stream_id < stream_num_; stream_id++) {
    if (specific_activated_streams_.count(stream_id) == 0) {
      loop_active_streams.emplace_back(static_cast<uint32_t>(stream_id));
    }
  }
  // Set the stream that needs to be activated
  for (const auto &node : whole_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    bool is_loop_active = false;
    if (AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, is_loop_active) && is_loop_active) {
      vector<string> activated_label_list;
      if (!AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, activated_label_list) ||
          activated_label_list.empty()) {
        GE_CHK_BOOL_EXEC(AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, loop_active_streams),
                         GELOGE(FAILED, "SetListInt failed.");
                         return FAILED);
        for (const auto &stream_id : loop_active_streams) {
          GELOGI("Active stream %u for node: %s", stream_id, node->GetName().c_str());
        }

        break;
      }
    }
  }

  return CheckStreamActived();
}

Status StreamAllocator::CheckStreamActived() const {
  for (const auto &node : whole_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    vector<uint32_t> active_streams;
    if (AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams)) {
      uint32_t stream_id = static_cast<uint32_t>(node->GetOpDesc()->GetStreamId());
      auto iter = find(active_streams.begin(), active_streams.end(), stream_id);
      if (iter != active_streams.end()) {
        GELOGE(FAILED, "Node %s cannot active its own stream %u.", node->GetName().c_str(), stream_id);
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

// Insert the send/recv event id to the graph
Status StreamAllocator::InsertSyncEvents() {
  for (const auto &cur_node : whole_graph_->GetDirectNode()) {
    // Take the adjacent points, then judge whether need to insert the event
    for (const OutDataAnchorPtr &anchor : cur_node->GetAllOutDataAnchors()) {
      for (const InDataAnchorPtr &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
        NodePtr next_node = peer_in_anchor->GetOwnerNode();
        Status status = InsertOneEventInTwoNodes(cur_node, next_node);
        if (status != SUCCESS) {
          GELOGE(status, "InsertOneEventInTwoNodes failed!");
          return status;
        }
      }
    }

    /// If the two nodes of the control side belong to two streams,
    /// you also need to add the send/recv event.
    if (cur_node->GetOutControlAnchor() != nullptr) {
      for (const AnchorPtr &peer_in_anchor : cur_node->GetOutControlAnchor()->GetPeerAnchors()) {
        NodePtr next_node = peer_in_anchor->GetOwnerNode();
        Status status = InsertOneEventInTwoNodes(cur_node, next_node);
        if (status != SUCCESS) {
          GELOGE(status, "InsertOneEventInTwoNodes failed!");
          return status;
        }
      }
    }
  }

  return SUCCESS;
}

// Insert one send/recv event in two nodes
Status StreamAllocator::InsertOneEventInTwoNodes(const NodePtr &cur_node, const NodePtr &next_node) {
  GE_CHECK_NOTNULL(cur_node->GetOpDesc());
  GE_CHECK_NOTNULL(next_node->GetOpDesc());

  // No need to insert events after node that do not assign streams.
  int64_t cur_stream_id = cur_node->GetOpDesc()->GetStreamId();
  if (cur_stream_id == kInvalidStream) {
    GELOGD("No need to insert event after node %s.", cur_node->GetName().c_str());
    return SUCCESS;
  }

  // No need to insert events between nodes in the same stream.
  int64_t next_stream_id = next_node->GetOpDesc()->GetStreamId();
  if (cur_stream_id == next_stream_id) {
    return SUCCESS;
  }

  // No event needs to be inserted between the active node and the activated stream.
  string next_node_label;
  if (AttrUtils::GetStr(next_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, next_node_label) && !next_node_label.empty()) {
    auto iter = specific_activated_labels_.find(next_node_label);
    if (iter != specific_activated_labels_.end()) {
      for (const auto &active_node : iter->second) {
        OpDescPtr active_op = active_node->GetOpDesc();
        GE_CHECK_NOTNULL(active_op);
        if ((cur_stream_id == active_op->GetStreamId()) && (cur_node->GetOpDesc()->GetId() <= active_op->GetId())) {
          GELOGI("No need to insert event between node %s and %s.", cur_node->GetName().c_str(),
                 next_node->GetName().c_str());
          return SUCCESS;
        }
      }
    }
  }

  // Add send and receive events.
  AddSendEventId(cur_node, event_num_);
  AddRecvEventId(next_node, event_num_);
  GELOGD("Insert event %u between node %s(stream %ld) and %s(stream %ld)", event_num_, cur_node->GetName().c_str(),
         cur_stream_id, next_node->GetName().c_str(), next_stream_id);

  ++event_num_;

  return SUCCESS;
}

// Optimize the event in the graph, delete the redundant sync event according to the stream information
Status StreamAllocator::OptimizeSyncEvents() {
  map<int64_t, vector<NodePtr>> stream_nodes;

  for (const auto &node : whole_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    int64_t stream_id = node->GetOpDesc()->GetStreamId();
    stream_nodes[stream_id].emplace_back(node);
  }

  Status status = OptimizeBySendEvents(stream_nodes);
  if (status != SUCCESS) {
    GELOGE(status, "OptimizeBySendEvents failed!");
    return status;
  }

  status = OptimizeByRecvEvents(stream_nodes);
  if (status != SUCCESS) {
    GELOGE(status, "OptimizeByRecvEvents failed!");
    return status;
  }

  status = OptimizeByStreamActivate();
  if (status != SUCCESS) {
    GELOGE(status, "OptimizeByStreamActivate failed!");
    return status;
  }

  return SUCCESS;
}

/// Optimization scenario: one stream has multiple send events in one node,
/// and multiple nodes for recv events on another stream
/// Example:
/// Stream0            Stream1
///   N1 - - - event - > N1
///     \                |
///      \               v
///        - - event - > N2
Status StreamAllocator::OptimizeBySendEvents(const map<int64_t, vector<NodePtr>> &stream_nodes) {
  for (const auto &one_pair : stream_nodes) {
    // The nodes on a stream in order
    const vector<NodePtr> &nodes = one_pair.second;

    map<NodePtr, uint32_t> send_node_to_event_id;

    for (const auto &recv_node_ptr : nodes) {
      GE_CHECK_NOTNULL(recv_node_ptr);
      // Get all recv events of the current node, then traverse the event
      vector<uint32_t> recv_events;
      GetRecvEventIdList(recv_node_ptr, recv_events);

      for (const auto &event_id : recv_events) {
        NodePtr send_node_ptr = GetNodeFromSendEventId(event_id);
        GE_CHECK_NOTNULL(send_node_ptr);

        /// If the record to the stream is found in the map,
        /// and the recv node is the node, then remove sync event
        if (send_node_to_event_id.find(send_node_ptr) != send_node_to_event_id.end()) {
          RmvSendEventId(send_node_ptr, event_id);
          RmvRecvEventId(recv_node_ptr, event_id);
          GELOGI("Remove send event %u for node: %s", event_id, send_node_ptr->GetName().c_str());
          GELOGI("Remove recv event %u for node: %s", event_id, recv_node_ptr->GetName().c_str());
        } else {
          send_node_to_event_id[send_node_ptr] = event_id;
        }
      }
    }
  }

  return SUCCESS;
}

/// Scenario: multiple send nodes on a stream sent to a single recv node on the destination stream
/// Example:
/// Stream0            Stream1
///   N1 - -
///   |    |
///   |    - - event - - -
///   |                  |
///   V                  V
///   N2 - - - event - > N2
Status StreamAllocator::OptimizeByRecvEvents(const map<int64_t, vector<NodePtr>> &stream_nodes) {
  for (const auto &one_pair : stream_nodes) {
    // The nodes on a stream in order
    const vector<NodePtr> &nodes = one_pair.second;

    map<NodePtr, uint32_t> recv_node_to_event_id;

    for (const auto &send_node_ptr : nodes) {
      GE_CHECK_NOTNULL(send_node_ptr);
      //  Get all send events of the current node, then traverse the event
      vector<uint32_t> send_id_list;
      GetSendEventIdList(send_node_ptr, send_id_list);

      for (const auto &event_id : send_id_list) {
        NodePtr recv_node_ptr = GetNodeFromRecvEventId(event_id);
        GE_CHECK_NOTNULL(recv_node_ptr);

        /// If the record to the stream is found in the map,
        /// and the send node is the node, then remove sync event
        auto it = recv_node_to_event_id.find(recv_node_ptr);
        if (it != recv_node_to_event_id.end()) {
          uint32_t pre_event_id = it->second;
          NodePtr pre_send_node_ptr = GetNodeFromSendEventId(pre_event_id);
          GE_CHECK_NOTNULL(pre_send_node_ptr);

          RmvSendEventId(pre_send_node_ptr, pre_event_id);
          RmvRecvEventId(recv_node_ptr, pre_event_id);
          GELOGI("Remove event %u between node %s and node %s.", event_id, pre_send_node_ptr->GetName().c_str(),
                 recv_node_ptr->GetName().c_str());
        }
        recv_node_to_event_id[recv_node_ptr] = event_id;
      }
    }
  }

  return SUCCESS;
}

// In situation : stream(normal) -> stream(streamActivate)->
// -> stream(streamSwitch) -> stream(streamActivate) -> stream(stream true or false)
// No need to insert an event between node in stream(normal) and node in stream(stream true or false)
bool StreamAllocator::IsRecvNodeActivatedBySendNode(const NodePtr &send_node_ptr, const NodePtr &recv_node_ptr) const {
  GE_CHECK_NOTNULL_EXEC(send_node_ptr->GetOpDesc(), GELOGE(FAILED, "op desc is nullptr"); return false);
  GE_CHECK_NOTNULL_EXEC(recv_node_ptr->GetOpDesc(), GELOGE(FAILED, "op desc is nullptr"); return false);
  auto cur_stream_id = send_node_ptr->GetOpDesc()->GetStreamId();
  if (AttrUtils::HasAttr(recv_node_ptr->GetOpDesc(), ATTR_NAME_STREAM_LABEL)) {
    // find streamActivate node
    auto iter = specific_activated_streams_nodes_map_.find(recv_node_ptr->GetOpDesc()->GetStreamId());
    set<NodePtr> activate_stream_nodes;
    if (iter != specific_activated_streams_nodes_map_.end()) {
      activate_stream_nodes = iter->second;
    }
    set<NodePtr> visited_nodes{recv_node_ptr};
    while (!activate_stream_nodes.empty()) {
      set<NodePtr> activate_stream_nodes_temp;
      for (const auto &activate_stream_node : activate_stream_nodes) {
        GE_IF_BOOL_EXEC(activate_stream_node->GetOpDesc() == nullptr, continue);
        if (visited_nodes.find(activate_stream_node) != visited_nodes.end() ||
            AttrUtils::HasAttr(activate_stream_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE)) {
          return false;
        }
        visited_nodes.insert(activate_stream_node);
        // nodes in stream link to streamActivate no need to add event before activated node
        for (const auto &pre_activate_stream_node : activate_stream_node->GetInNodes()) {
          GE_IF_BOOL_EXEC(pre_activate_stream_node->GetOpDesc() == nullptr, continue);
          if (pre_activate_stream_node->GetOpDesc()->GetStreamId() == cur_stream_id &&
              pre_activate_stream_node->GetOpDesc()->GetId() >= send_node_ptr->GetOpDesc()->GetId()) {
            return true;
          }
          auto in_nodes_of_pre = pre_activate_stream_node->GetInNodes();
          if (std::find(in_nodes_of_pre.begin(), in_nodes_of_pre.end(), send_node_ptr) != in_nodes_of_pre.end()) {
            return true;
          }
        }
        auto iterator = specific_activated_streams_nodes_map_.find(activate_stream_node->GetOpDesc()->GetStreamId());
        if (iterator != specific_activated_streams_nodes_map_.end()) {
          auto active_nodes = iterator->second;
          for (const auto &active_node : active_nodes) {
            activate_stream_nodes_temp.emplace(active_node);
          }
        }
      }
      activate_stream_nodes = activate_stream_nodes_temp;
    }
  }
  return false;
}

Status StreamAllocator::OptimizeByStreamActivate() {
  auto node_to_send_events_temp = node_to_send_events_;
  for (const auto &node_event_id_pair : node_to_send_events_temp) {
    const NodePtr &send_node_ptr = node_event_id_pair.first;
    for (const auto &event_id : node_event_id_pair.second) {
      NodePtr recv_node_ptr = GetNodeFromRecvEventId(event_id);
      GE_CHECK_NOTNULL(recv_node_ptr);
      if (IsRecvNodeActivatedBySendNode(send_node_ptr, recv_node_ptr)) {
        RmvSendEventId(send_node_ptr, event_id);
        RmvRecvEventId(recv_node_ptr, event_id);
      }
    }
  }
  return SUCCESS;
}

// Refresh events to continuous events
Status StreamAllocator::RefreshContinuousEvents() {
  // Establish a mapping relationship from old to new event id
  map<uint32_t, uint32_t> old_to_new_events;
  uint32_t new_event_id = 0;
  for (const auto &one_pair : node_to_send_events_) {
    for (const auto &event_id : one_pair.second) {
      old_to_new_events[event_id] = new_event_id;
      new_event_id++;
    }
  }

  // Refresh send event id
  for (auto &one_pair : node_to_send_events_) {
    vector<uint32_t> &send_events = one_pair.second;
    for (size_t i = 0; i < send_events.size(); i++) {
      auto find_it = old_to_new_events.find(send_events[i]);
      if (find_it == old_to_new_events.end()) {
        GELOGE(FAILED, "RefreshContinuousEvents: invalid send event %u", send_events[i]);
        return FAILED;
      }
      send_events[i] = find_it->second;
    }
  }

  // Refresh recv event id
  for (auto &one_pair : node_to_recv_events_) {
    vector<uint32_t> &recv_events = one_pair.second;
    for (size_t i = 0; i < recv_events.size(); i++) {
      auto find_it = old_to_new_events.find(recv_events[i]);
      if (find_it == old_to_new_events.end()) {
        GELOGE(FAILED, "RefreshContinuousEvents: invalid recv event %u", recv_events[i]);
        return FAILED;
      }
      recv_events[i] = find_it->second;
    }
  }

  event_num_ = static_cast<uint32_t>(old_to_new_events.size());

  return SUCCESS;
}

// Insert the real send/recv node in the graph
Status StreamAllocator::InsertSyncEventNodes() {
  for (const auto &node : whole_graph_->GetDirectNode()) {
    // Add the node corresponding to the recv event
    vector<uint32_t> recv_event_id_list;
    GetRecvEventIdList(node, recv_event_id_list);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GE_CHECK_NOTNULL(node->GetInControlAnchor());
    GE_CHECK_NOTNULL(node->GetOutControlAnchor());
    for (auto &event_id : recv_event_id_list) {
      string recv_node_name = "_Recv_" + to_string(event_id);
      OpDescPtr op_desc_ptr = MakeShared<OpDesc>(recv_node_name, RECV);
      GE_CHECK_NOTNULL(op_desc_ptr);

      int64_t temp_stream_id = node->GetOpDesc()->GetStreamId();
      op_desc_ptr->SetStreamId(temp_stream_id);
      GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc_ptr, RECV_ATTR_EVENT_ID, event_id), GELOGE(FAILED, "SetInt failed.");
                       return FAILED);
      (void)AttrUtils::SetListStr(op_desc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                  std::move(std::vector<std::string>()));
      NodePtr recv_node = whole_graph_->AddNode(op_desc_ptr);
      GE_CHECK_NOTNULL(recv_node);
      GE_CHECK_NOTNULL(recv_node->GetOutControlAnchor());
      Status status = GraphUtils::AddEdge(recv_node->GetOutControlAnchor(), node->GetInControlAnchor());
      if (status != SUCCESS) {
        GELOGE(status, "Add edge for node %s and node %s failed.", recv_node->GetName().c_str(),
               node->GetName().c_str());
        return status;
      }

      GELOGI("Add recv %u before node: %s", event_id, node->GetName().c_str());
    }

    // Add the node corresponding to the send event
    vector<uint32_t> send_event_id_list;
    GetSendEventIdList(node, send_event_id_list);

    for (auto &event_id : send_event_id_list) {
      string send_node_name = "_Send_" + to_string(event_id);
      OpDescPtr op_desc_ptr = MakeShared<OpDesc>(send_node_name, SEND);
      GE_CHECK_NOTNULL(op_desc_ptr);

      int64_t temp_stream_id = node->GetOpDesc()->GetStreamId();
      op_desc_ptr->SetStreamId(temp_stream_id);
      GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc_ptr, SEND_ATTR_EVENT_ID, event_id), GELOGE(FAILED, "SetInt failed.");
                       return FAILED);
      (void)AttrUtils::SetListStr(op_desc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                  std::move(std::vector<std::string>()));
      NodePtr send_node = whole_graph_->AddNode(op_desc_ptr);
      GE_CHECK_NOTNULL(send_node);
      GE_CHECK_NOTNULL(send_node->GetInControlAnchor());
      Status status = GraphUtils::AddEdge(node->GetOutControlAnchor(), send_node->GetInControlAnchor());
      if (status != SUCCESS) {
        GELOGE(status, "Add edge for node %s and node %s failed.", node->GetName().c_str(),
               send_node->GetName().c_str());
        return status;
      }

      GELOGI("Add send event %u after node: %s", event_id, node->GetName().c_str());
    }
  }

  Status status = whole_graph_->InsertEventNodes();
  if (status != SUCCESS) {
    GELOGE(status, "whole_graph_->InsertEventNodes failed");
    return status;
  }

  return SUCCESS;
}

// Insert send event id on a node
void StreamAllocator::AddSendEventId(const NodePtr &node, uint32_t event_id) {
  node_to_send_events_[node].emplace_back(event_id);
}

// Insert recv event id on a node
void StreamAllocator::AddRecvEventId(const NodePtr &node, uint32_t event_id) {
  node_to_recv_events_[node].emplace_back(event_id);
}

// Remove send event id from a node
void StreamAllocator::RmvSendEventId(const NodePtr &node, uint32_t event_id) {
  auto find_it = node_to_send_events_.find(node);
  if (find_it == node_to_send_events_.end()) {
    return;
  }

  vector<uint32_t> &send_events = find_it->second;
  for (auto it = send_events.begin(); it != send_events.end(); ++it) {
    if (*it == event_id) {
      send_events.erase(it);
      return;
    }
  }
}

// Remove recv event id from a node
void StreamAllocator::RmvRecvEventId(const NodePtr &node, uint32_t event_id) {
  auto find_it = node_to_recv_events_.find(node);
  if (find_it == node_to_recv_events_.end()) {
    return;
  }

  vector<uint32_t> &recv_events = find_it->second;
  for (auto it = recv_events.begin(); it != recv_events.end(); ++it) {
    if (*it == event_id) {
      recv_events.erase(it);
      return;
    }
  }
}

// Get send event id list from a node
void StreamAllocator::GetSendEventIdList(const NodePtr &node, vector<uint32_t> &send_list) const {
  send_list.clear();
  auto find_it = node_to_send_events_.find(node);
  if (find_it != node_to_send_events_.end()) {
    send_list = find_it->second;
  }
}

// Get recv event id list from a node
void StreamAllocator::GetRecvEventIdList(const NodePtr &node, vector<uint32_t> &recv_list) const {
  recv_list.clear();
  auto find_it = node_to_recv_events_.find(node);
  if (find_it != node_to_recv_events_.end()) {
    recv_list = find_it->second;
  }
}

// Get a specific send node according to the recv event
NodePtr StreamAllocator::GetNodeFromSendEventId(uint32_t send_event_id) const {
  for (const auto &one_pair : node_to_send_events_) {
    const vector<uint32_t> &events = one_pair.second;
    for (const auto &event_id : events) {
      if (event_id == send_event_id) {
        return one_pair.first;
      }
    }
  }

  return nullptr;
}

// Get a specific recv node according to the recv event
NodePtr StreamAllocator::GetNodeFromRecvEventId(uint32_t recv_event_id) const {
  for (const auto &one_pair : node_to_recv_events_) {
    const vector<uint32_t> &events = one_pair.second;
    for (const auto &event_id : events) {
      if (event_id == recv_event_id) {
        return one_pair.first;
      }
    }
  }

  return nullptr;
}

void StreamAllocator::DumpEvents() {
  map<int64_t, vector<NodePtr>> after_refresh_stream_nodes;
  for (const auto &node : whole_graph_->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    int64_t stream_id = node->GetOpDesc()->GetStreamId();
    after_refresh_stream_nodes[stream_id].emplace_back(node);
  }

  for (const auto &one_pair : after_refresh_stream_nodes) {
    int64_t stream_id = one_pair.first;
    GELOGI("After RefreshRealStream: stream %ld.", stream_id);

    for (const auto &node : one_pair.second) {
      string send_event_str;
      for (const auto &send_event_id : node_to_send_events_[node]) {
        send_event_str += " " + to_string(send_event_id);
      }
      if (!send_event_str.empty()) {
        GELOGI("node: %s, send events: %s", node->GetName().c_str(), send_event_str.c_str());
      }

      string recv_event_str;
      for (const auto &recv_event_id : node_to_recv_events_[node]) {
        recv_event_str += " " + to_string(recv_event_id);
      }
      if (!recv_event_str.empty()) {
        GELOGI("node: %s, recv events: %s", node->GetName().c_str(), recv_event_str.c_str());
      }
    }
  }
}

// Add active entry stream for special env.
Status StreamAllocator::AddActiveEntryStream() {
  auto gelib = GELib::GetInstance();
  bool head_stream = (gelib == nullptr) ? false : gelib->HeadStream();
  GELOGI("Configured head stream: %u", head_stream);
  if (!head_stream) {
    return SUCCESS;
  }

  // Collect streams active by StreamSwitch/StreamActive node.
  std::set<uint32_t> deactive_stream;
  for (ge::NodePtr &node : whole_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    Status ret = CollectDeactiveStream(node->GetOpDesc(), deactive_stream);
    if (ret != SUCCESS) {
      return ret;
    }
  }

  // Collect default active stream, Add to active entry stream.
  std::vector<uint32_t> active_stream_list;
  for (int64_t stream_id = 0; stream_id < stream_num_; ++stream_id) {
    if (deactive_stream.count(stream_id) == 0) {
      active_stream_list.push_back(stream_id);
    }
  }

  int64_t new_stream_id = stream_num_;
  stream_num_++;
  return InsertActiveEntryStream(active_stream_list, new_stream_id);
}

// Collect deactive stream from flowctrl op.
Status StreamAllocator::CollectDeactiveStream(const OpDescPtr &op_desc, std::set<uint32_t> &deactive_streams) const {
  GE_CHECK_NOTNULL(op_desc);
  std::string op_type = op_desc->GetType();
  if (op_type == STREAMSWITCH) {
    std::vector<uint32_t> active_stream_list;
    // If GetListInt fail, active_stream_list is empty.
    (void)ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);
    if (active_stream_list.size() != kMaxSwitchStreamNum) {
      GELOGE(INTERNAL_ERROR, "Stream num of switch true branch must be %u.", kMaxSwitchStreamNum);
      return INTERNAL_ERROR;
    }

    deactive_streams.insert(active_stream_list[0]);
    GELOGI("Flowctrl_op node:%s, flowctrl stream id:%u.", op_desc->GetName().c_str(), active_stream_list[0]);
  } else if (op_type == STREAMACTIVE) {
    if (op_desc->HasAttr(ATTR_NAME_SWITCH_BRANCH_NODE_LABEL)) {
      std::vector<uint32_t> active_stream_list;
      if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list)) {
        GELOGE(INTERNAL_ERROR, "StreamActiveOp get attr ACTIVE_STREAM fail.");
        return INTERNAL_ERROR;
      }

      for (uint32_t deactive_stream : active_stream_list) {
        deactive_streams.insert(deactive_stream);
        GELOGI("Flowctrl_op node:%s, flowctrl stream id:%u.", op_desc->GetName().c_str(), deactive_stream);
      }
    }
  }

  return SUCCESS;
}

// Insert StreamActive Op for Entry Stream.
Status StreamAllocator::InsertActiveEntryStream(const std::vector<uint32_t> &active_streams, int64_t stream_id) {
  string node_name = "ActiveEntryStream_" + string(STREAMACTIVE);
  OpDescPtr op_desc = ge::MakeShared<OpDesc>(node_name, STREAMACTIVE);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Failed to new opdesc.");
    return FAILED;
  }
  GELOGI("Create StreamActive op:%s.", op_desc->GetName().c_str());

  GE_CHK_BOOL_EXEC(
    AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::move(std::vector<std::string>())),
    GELOGE(FAILED, "SetListStr failed.");
    return FAILED);

  NodePtr active_node = whole_graph_->AddNodeFront(op_desc);
  GE_IF_BOOL_EXEC(active_node == nullptr,
                  GELOGE(FAILED, "Create StreamActive op: %s failed.", op_desc->GetName().c_str());
                  return INTERNAL_ERROR);
  GE_CHECK_NOTNULL(active_node->GetOpDesc());
  // Add one stream for ActiveEntryStream Task.
  active_node->GetOpDesc()->SetStreamId(stream_id);

  GE_CHK_BOOL_EXEC(AttrUtils::SetBool(op_desc, "is_aicpu_stream", true), GELOGE(FAILED, "SetBool failed.");
                   return FAILED);
  GE_CHK_BOOL_EXEC(AttrUtils::SetListInt(active_node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams),
                   GELOGE(FAILED, "SetListInt failed.");
                   return FAILED);

  std::vector<std::string> group_names;
  GE_CHK_BOOL_EXEC(AttrUtils::SetListStr(active_node->GetOpDesc(), ATTR_NAME_SWITCH_BRANCH_NODE_LABEL, group_names),
                   GELOGE(FAILED, "SetLisStr failed.");
                   return FAILED);

  return SUCCESS;
}
}  // namespace ge
