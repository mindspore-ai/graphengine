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
#include "model/ge_root_model.h"

namespace ge {
namespace hybrid {
class HybridModelAsyncExecutor;
class HybridModel {
 public:
  explicit HybridModel(GeRootModelPtr ge_model);

  ~HybridModel() = default;

  Status Init();

  const std::vector<NodeItem *> &RootNodes() const { return root_nodes_; }

  const NodeItem *GetNodeItem(const NodePtr &node) const;

  size_t NumNodes() const { return node_items_.size(); }

  uint64_t GetSessionId() const { return root_runtime_param_.session_id; }

  int TotalInputs() const { return total_inputs_; }

  const map<uint32_t, NodeItem *> &GetInputNodes() const { return input_nodes_; }

  const std::map<uint32_t, std::vector<int>> &GetInputOffsets() const { return input_offsets_; }

  const vector<int> &GetNetOutputInputOffsets() const;

  const std::vector<int> &GetOutputOffsets() const { return output_offsets_; }

  const std::vector<NodeItem *> &GetConstNodes() const { return const_nodes_; }

  GeModelPtr GetGeModel(const NodePtr &node) const;

  NodeItem *MutableNodeItem(const NodePtr &node);

  size_t TotalVarMemSize() const { return root_runtime_param_.var_size; }

  const uint8_t *GetVarMemBase() const { return var_mem_base_; }

  void SetDeviceId(uint32_t device_id);

  void SetModelId(uint32_t model_id) { model_id_ = model_id; }

  uint32_t GetModelId() const { return model_id_; }

  TensorValue *GetWeight(const NodeItem *const_node) const;

  TensorValue *GetVariable(const string &name) const;

  NodePtr GetVariableNode(const string &name) const;

  const std::vector<domi::TaskDef> *GetTaskDefs(const NodePtr &node) const;

  int TotalOutputs() const { return total_outputs_; }

  GeRootModelPtr GetGeRootModel() const { return ge_root_model_; }
  void Print() const;

 private:
  friend class HybridModelBuilder;
  friend class HybridModelAsyncExecutor;

  GeRootModelPtr ge_root_model_;
  std::vector<NodeItem *> root_nodes_;
  std::map<uint32_t, NodeItem *> input_nodes_;
  std::map<uint32_t, std::vector<int>> input_offsets_;
  std::vector<int> output_offsets_;
  std::vector<int> net_output_input_offsets_;
  NodeItem *net_output_node_ = nullptr;
  std::vector<std::unique_ptr<NodeItem>> node_items_;
  std::vector<NodeItem *> const_nodes_;
  std::map<std::string, NodePtr> constant_op_nodes_;
  std::map<std::string, NodePtr> variable_nodes_;
  std::map<std::string, std::unique_ptr<TensorValue>> variable_tensors_;
  std::map<int, std::unique_ptr<TensorValue>> weights_;
  std::map<NodePtr, std::vector<domi::TaskDef>> task_defs_;
  std::map<NodePtr, GeModelPtr> known_shape_sub_graphs_;
  int total_inputs_ = 0;
  int total_outputs_ = 0;

  // runtime fields
  uint32_t device_id_ = 0;
  uint32_t model_id_ = 0;
  uint8_t *var_mem_base_ = nullptr;
  RuntimeParam root_runtime_param_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_HYBRID_GRAPH_H_
