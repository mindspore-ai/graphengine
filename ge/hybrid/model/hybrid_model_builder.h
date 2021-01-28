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

#ifndef GE_HYBRID_MODEL_HYBRID_MODEL_BUILDER_H_
#define GE_HYBRID_MODEL_HYBRID_MODEL_BUILDER_H_

#include <vector>
#include <queue>
#include <memory>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/node.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/model/node_item.h"
#include "model/ge_model.h"

namespace ge {
class VarManager;
namespace hybrid {
class HybridModelBuilder {
 public:
  explicit HybridModelBuilder(HybridModel &hybrid_model);
  ~HybridModelBuilder() = default;
  Status Build();
  Status BuildForSingleOp();

 private:
  static Status UpdateAnchorStatus(const NodePtr &node);
  static Status DoUnlinkDataAnchors(const OutDataAnchorPtr &out_data_anchor, const InDataAnchorPtr &in_data_anchor);
  static Status DoLinkDataAnchors(OutDataAnchorPtr &out_data_anchor, InDataAnchorPtr &in_data_anchor);
  static NodePtr GetPeerNode(const InDataAnchorPtr &in_data_anchor);
  static Status GetParentNodeOutputIndex(const OpDesc &op_desc, int index, uint32_t &out_index);
  static Status GetPeerNodeAcrossSubGraphs(const NodePtr &data_node, NodePtr &peer_node, int &peer_out_index);
  static Status HandleDtString(const GeTensor &tensor, void *var_addr);
  static Status MergeInputNodes(ComputeGraph &compute_graph);
  static Status MergeNetOutputNode(ComputeGraph &compute_graph);
  static Status UnfoldSubgraphs(ComputeGraph &root_graph, ComputeGraphPtr &merged_graph);
  static Status UnfoldSubgraph(ComputeGraph &root_graph, ComputeGraph &parent_graph, ComputeGraph &sub_graph);
  static Status BuildInputMapping(GraphItem &graph_item,
                                  std::vector<NodeItem *> &data_nodes,
                                  bool is_root_graph);
  static Status ResolveRefIo(NodeItem &node_item);
  Status BuildOutputMapping(GraphItem &partitioned_call, const NodeItem &node_item, bool is_root_graph);
  Status ValidateParams();
  Status LoadGraph();
  Status LoadGeModel(ComputeGraph &graph, const GeModelPtr &ge_model);
  Status LoadTasks();
  Status IdentifyVariableOutputs(NodeItem &node_item);
  Status IdentifySameInputs(NodeItem &node_item);
  Status BuildNodeItem(const NodePtr &node, NodeItem &node_item);
  Status GetOrCreateNodeItem(const NodePtr &node, NodeItem **node_item);
  Status ParseDependentInputNodes(NodeItem &node_item, const std::vector<string> &dependencies);
  Status ParseDependentForFusedSubgraph(NodeItem &node_item);
  Status IndexTaskDefs();
  Status IndexTaskDefs(const ComputeGraphPtrs &sub_graph, const GeModelPtr &ge_model);
  Status IndexSpecialNodes();
  Status InitRuntimeParams();
  Status InitModelMem();
  Status InitWeights();
  Status TransAllVarData();
  Status CopyVarData();
  Status VarNodeToTensor(const NodePtr &var_node, std::unique_ptr<TensorValue> &tensor);
  Status AssignUninitializedConstantOps();
  Status InitConstantOps();
  Status InitVariableTensors();
  Status LoadDynamicSubgraph(ComputeGraph &graph, bool is_root_graph);
  Status ParseVarOutputs(NodeItem &node_item);
  Status LoadKnownShapedSubgraph(ComputeGraph &graph, NodeItem *parent_node_item);
  Status RecoverGraphUnknownFlag();
  Status CheckAicpuOpList();
  Status CreateProfilingNodeBefore(GraphItem &graph_item, const NodePtr &node);
  Status CreateProfilingNodeAfter(GraphItem &graph_item, const NodePtr &node);
  Status GenerateFpProfilingTask(const OpDescPtr &op_desc, vector<domi::TaskDef> &task_def_list);
  Status GenerateBpProfilingTask(const OpDescPtr &op_desc, vector<domi::TaskDef> &task_def_list);
  Status GenerateEndProfilingTask(const OpDescPtr &op_desc, vector<domi::TaskDef> &task_def_list);
  Status GenerateArProfilingTask(const OpDescPtr &op_desc, int64_t log_id, vector<domi::TaskDef> &task_def_list);

  const char* GetGraphName() const {
    return hybrid_model_.model_name_.c_str();
  }

  const NodeItem *GetNodeItem(const NodePtr &node) const;
  NodeItem *MutableNodeItem(const NodePtr &node);

  GeRootModelPtr ge_root_model_;
  std::map<std::string, GeModelPtr> subgraph_models_;
  std::map<std::string, NodePtr> constant_op_nodes_;

  HybridModel &hybrid_model_;
  std::map<NodePtr, std::vector<std::pair<int, NodePtr>>> node_ref_inputs_;
  int node_index = 0;

  RuntimeParam &runtime_param_;
  VarManager *var_manager_ = nullptr;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_MODEL_HYBRID_MODEL_BUILDER_H_
