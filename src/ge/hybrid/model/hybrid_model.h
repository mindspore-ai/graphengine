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

#ifndef GE_HYBRID_HYBRID_GRAPH_H_
#define GE_HYBRID_HYBRID_GRAPH_H_

#include <vector>
#include <queue>
#include <memory>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/new_model_manager/data_inputer.h"
#include "graph/load/new_model_manager/task_info/task_info.h"
#include "graph/node.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/model/node_item.h"
#include "hybrid/model/graph_item.h"
#include "model/ge_root_model.h"

namespace ge {
namespace hybrid {
class HybridModel {
 public:
  explicit HybridModel(GeRootModelPtr ge_model);

  ~HybridModel();

  Status Init();

  const NodeItem *GetNodeItem(const NodePtr &node) const;

  uint64_t GetSessionId() const { return root_runtime_param_.session_id; }

  GeModelPtr GetGeModel(const NodePtr &node) const;

  NodeItem *MutableNodeItem(const NodePtr &node);

  size_t TotalVarMemSize() const { return root_runtime_param_.var_size; }

  const uint8_t *GetVarMemBase() const { return var_mem_base_; }

  void SetDeviceId(uint32_t device_id) { device_id_ = device_id; }

  void SetModelId(uint32_t model_id) { model_id_ = model_id; }

  uint32_t GetModelId() const { return model_id_; }

  TensorValue *GetVariable(const string &name) const;

  NodePtr GetVariableNode(const string &name) const;

  const std::vector<domi::TaskDef> *GetTaskDefs(const NodePtr &node) const;

  const GraphItem *GetRootGraphItem() const;

  const GraphItem *GetSubgraphItem(const std::string &graph_name) const;

  const GraphItem *GetSubgraphItem(const ComputeGraphPtr &subgraph) const;

  const string &GetModelName() const;

 private:
  friend class HybridModelBuilder;
  friend class HybridModelAsyncExecutor;

  std::string model_name_;
  GeRootModelPtr ge_root_model_;
  std::map<uint32_t, NodeItem *> input_nodes_;
  std::map<std::string, NodePtr> constant_op_nodes_;
  std::map<std::string, NodePtr> variable_nodes_;
  std::map<std::string, std::unique_ptr<TensorValue>> variable_tensors_;
  std::map<NodePtr, std::vector<domi::TaskDef>> task_defs_;
  std::map<NodePtr, GeModelPtr> known_shape_sub_models_;

  std::unique_ptr<GraphItem> root_graph_item_;
  std::map<std::string, std::unique_ptr<GraphItem>> subgraph_items_;
  std::map<NodePtr, std::unique_ptr<NodeItem>> node_items_;

  // runtime fields
  uint32_t device_id_ = 0;
  uint32_t model_id_ = 0;
  uint8_t *var_mem_base_ = nullptr;
  RuntimeParam root_runtime_param_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_HYBRID_GRAPH_H_
