/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef D_BASE_GRAPH_PARTITION_DEFAULT_SUBGRAPH_BUILDER_H
#define D_BASE_GRAPH_PARTITION_DEFAULT_SUBGRAPH_BUILDER_H

#include "graph/partition/base_subgraph_builder.h"

namespace ge {
class DefaultSubgraphBuilder : public BaseSubgraphBuilder {
 public:
  DefaultSubgraphBuilder() = default;
  ~DefaultSubgraphBuilder() = default;
  graphStatus BuildSubgraph() override;
 private:
  graphStatus MakePartitionedCallOp(const std::string &sub_graph_name);
  graphStatus MakeSubgraph(const std::string &sub_graph_name);

  graphStatus AddDataInput(const NodePtr &node);
  graphStatus AddDataOutput(const NodePtr &node);
  graphStatus AddControlInput(const NodePtr &node);
  graphStatus AddControlOutput(const NodePtr &node);
  graphStatus BuildPartitionedCallNodeIO();
  graphStatus MakePartitionedCallNode();
  // create partitioned call node, subgraph and build partitioned call node inputs/outputs
  graphStatus BuildSubgraphFrame();

  // delete combination between different subgraphs, and combine to parent partitioned call node
  graphStatus CombinePartitionedCall();
  graphStatus CombineToPartitionedCallDataInputs();
  graphStatus CombineToPartitionedCallDataOutputs() const;
  graphStatus CombineToPartitionedCallControlInput() const;
  graphStatus CombineToPartitionedCallControlOutput() const;

  // add data node，and set input_desc to it's output tensor
  // add netoutput node, and set output_desc to it's input tensors
  graphStatus AddSubgraphInputOutputNodes();
  graphStatus MakeDataNode(const size_t parent_node_index, const InDataAnchorPtr &data_anchor, NodePtr &data_node);
  graphStatus MakeNetOutputNode(NodePtr &net_output_node) const;
  graphStatus UpdateNetOutputInputTensors(const NodePtr &net_output_node) const;
  graphStatus AddSubgraphInputNodes();
  graphStatus AddSubgraphOutputNode() const;

  bool AddFrameInput(const InDataAnchorPtr &anchor);
  void AddFrameOutput(const OutDataAnchorPtr &anchor);
  // reset subgraph frame io and node/subgraph for build next subgraph
  void ResetSubgraphFrame();
  graphStatus CheckInputValid() const;

  ComputeGraphPtr subgraph_{nullptr};
  OpDescPtr partitioned_call_op_{nullptr};
  NodePtr partitioned_call_node_{nullptr};

  // data input/output
  std::vector<InDataAnchorPtr> inputs_;
  std::vector<OutDataAnchorPtr> outputs_;
  // control input/output
  std::vector<InControlAnchorPtr> control_inputs_;
  std::vector<OutControlAnchorPtr> control_outputs_;

  // used for merge_inputs, key: peer_node_name + ":" + peer_node_index
  std::map<std::string, size_t> src_key_to_frame_input_index_;
  // frame input index map to nodes which linked to the same data node or output anchor
  std::map<size_t, std::vector<InDataAnchorPtr>> frame_input_index_to_inputs_;
  // unique_id for partitioned call
  size_t unique_id_ = 0UL;
};
}  // namespace ge
#endif  // D_BASE_GRAPH_PARTITION_DEFAULT_SUBGRAPH_BUILDER_H
