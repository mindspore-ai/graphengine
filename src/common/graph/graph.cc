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

#include "external/graph/graph.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

using std::map;
using std::pair;
using std::string;
using std::vector;

namespace ge {
class GraphImpl {
 public:
  friend class GraphUtils;
  GraphImpl(const GraphImpl &) = delete;
  GraphImpl &operator=(const GraphImpl &) = delete;

  explicit GraphImpl(const std::string &name) : name_(name) {}

  ~GraphImpl() {
    if (IsValid()) {
      if (compute_graph_ != nullptr) {
        GraphUtils::BreakConnect(compute_graph_->GetAllNodesInfo());
      }
    }
    for (const auto &it : op_list_) {
      Operator op = it.second;
      op.BreakConnect();
    }
  }

  graphStatus SetInputs(const std::vector<Operator> &inputs) {
    compute_graph_ = GraphUtils::CreateGraphFromOperator(name_, inputs);
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "Build Graph failed.");
    GE_CHK_BOOL_RET_STATUS(inputs.size() != 0, GRAPH_FAILED, "set input NULL.");
    compute_graph_->SetInputSize(static_cast<uint32_t>(inputs.size()));
    return GRAPH_SUCCESS;
  }

  graphStatus SetOutputs(const std::vector<Operator> &outputs) {
    if (compute_graph_ == nullptr) {
      GELOGE(GRAPH_FAILED, "set ComputeGraph failed.");
      return GRAPH_FAILED;
    }
    if (outputs.empty()) {
      GELOGW("set outputs size is 0.");
      return GRAPH_SUCCESS;
    }

    // Construct special output node
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    for (size_t i = 0; i < outputs.size(); ++i) {
      output_indexs.emplace_back(outputs[i], std::vector<size_t>{});
    }

    graphStatus ret = SetOutputs(output_indexs);
    return ret;
  }

  graphStatus SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexs) {
    if (compute_graph_ == nullptr) {
      GELOGE(GRAPH_FAILED, "set ComputeGraph failed.");
      return GRAPH_FAILED;
    }
    if (output_indexs.empty()) {
      GELOGW("set outputs size is 0.");
      return GRAPH_SUCCESS;
    }

    // Construct special output node
    std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes;
    for (const auto &item : output_indexs) {
      const Operator &output = item.first;
      const vector<size_t> &indexs = item.second;
      ge::NodePtr node = compute_graph_->FindNode(output.GetName());
      if (node == nullptr) {
        GELOGW("user designated out_node [%s] not exist in graph, will ignored!", output.GetName().c_str());
        continue;
      }

      ge::OpDescPtr tmp_op_ptr = node->GetOpDesc();
      GE_CHECK_NOTNULL_EXEC(tmp_op_ptr, continue);
      size_t out_size = tmp_op_ptr->GetOutputsSize();
      if (indexs.empty()) {
        for (size_t i = 0; i < out_size; ++i) {
          output_name_ += output.GetName() + ":" + std::to_string(i) + ";";
          output_nodes.emplace_back(node, i);
        }
      } else {
        for (size_t i = 0; i < indexs.size(); ++i) {
          if (indexs[i] >= out_size) {
            GELOGW("index[%zu] is not belong to out_node[%s]", indexs[i], output.GetName().c_str());
          } else {
            output_name_ += output.GetName() + ":" + std::to_string(i) + ";";
            output_nodes.emplace_back(node, indexs[i]);
          }
        }
      }
    }

    // Del last ";"
    if (!output_name_.empty()) {
      output_name_ = output_name_.substr(0, output_name_.length() - 1);
    }
    compute_graph_->SetUserDefOutput(output_name_);
    compute_graph_->SetOutputSize(static_cast<uint32_t>(output_indexs.size()));
    compute_graph_->SetGraphOutNodesInfo(output_nodes);
    return GRAPH_SUCCESS;
  }

  graphStatus SetOutputs(const std::vector<pair<Operator, string>> &outputs) {
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "set ComputeGraph faild.");
    GE_CHK_BOOL_EXEC_INFO(outputs.size() != 0, return GRAPH_SUCCESS, "set outputs size is 0.");

    // Construct specified output
    std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes;
    for (auto item : outputs) {
      ge::NodePtr node = compute_graph_->FindNode(item.first.GetName());
      if (node == nullptr) {
        GELOGE(GRAPH_FAILED, " Warning, user designated out_node (%s) not exist in graph, this out_node ignored!",
               item.first.GetName().c_str());
        return GRAPH_FAILED;
      }
      ge::OpDescPtr tmp_op_ptr = node->GetOpDesc();
      GE_CHECK_NOTNULL_EXEC(tmp_op_ptr, continue);
      size_t out_size = tmp_op_ptr->GetOutputsSize();

      if (item.second.empty()) {
        for (size_t i = 0; i < out_size; ++i) {
          output_name_ += item.first.GetName() + ":" + std::to_string(i) + ";";
          output_nodes.push_back(std::make_pair(node, i));
        }
      } else {
        int32_t index = tmp_op_ptr->GetOutputIndexByName(item.second);
        if (index < 0) {
          GELOGE(GRAPH_FAILED,
                 " Warning, user designated out_node (%s):(%s) not exist in graph, this out_node ignored!",
                 item.first.GetName().c_str(), item.second.c_str());
          return GRAPH_FAILED;
        }
        output_name_ += item.first.GetName() + ":" + std::to_string(index) + ";";
        output_nodes.push_back(std::make_pair(node, index));
      }
    }
    // Del last ";"
    if (!output_name_.empty()) {
      output_name_ = output_name_.substr(0, output_name_.length() - 1);
    }
    compute_graph_->SetOutputSize(static_cast<uint32_t>(outputs.size()));
    compute_graph_->SetGraphOutNodesInfo(output_nodes);
    GELOGI("********************SetOutputs Success***********************");
    GE_IF_BOOL_EXEC(!output_name_.empty(), GELOGI(" NetOutputs: (%s)", output_name_.c_str()));

    return GRAPH_SUCCESS;
  }

  graphStatus SetTargets(const std::vector<Operator> &targets) {
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "set ComputeGraph faild.");
    GE_CHK_BOOL_EXEC_INFO(targets.size() != 0, return GRAPH_SUCCESS, "set targets size is 0.");

    std::vector<ge::NodePtr> target_nodes;
    for (auto item : targets) {
      ge::NodePtr node = compute_graph_->FindNode(item.GetName());
      if (node == nullptr) {
        GELOGW(" Warning, user designated target_node (%s) not exist in graph, this target_node ignored!",
               item.GetName().c_str());
        continue;
      }
      target_nodes.push_back(node);
    }
    compute_graph_->SetGraphTargetNodesInfo(target_nodes);
    return GRAPH_SUCCESS;
  }
  bool IsValid() const { return (compute_graph_ != nullptr); }

  graphStatus AddOp(const ge::Operator &op) {
    std::pair<std::map<string, ge::Operator>::iterator, bool> ret;
    ret = op_list_.emplace(std::pair<string, ge::Operator>(op.GetName(), op));
    GE_CHK_BOOL_RET_STATUS(ret.second != false, GRAPH_FAILED, "the op have added before, op name:%s.",
                           op.GetName().c_str());
    return GRAPH_SUCCESS;
  }

  graphStatus GetAllOpName(std::vector<string> &op_name) const {
    for (const auto &it : op_list_) {
      op_name.push_back(it.second.GetName());
    }
    return GRAPH_SUCCESS;
  }

  graphStatus FindOpByName(const string &name, ge::Operator &op) const {
    auto it = op_list_.find(name);
    GE_CHK_BOOL_EXEC(it != op_list_.end(), return GRAPH_FAILED, "there is no op: %s.", name.c_str());
    op = it->second;
    return GRAPH_SUCCESS;
  }

  graphStatus FindOpByType(const string &type, std::vector<ge::Operator> &ops) const {
    for (auto &op : op_list_) {
      auto op_type = op.second.GetOpType();
      if (op_type == type) {
        ops.push_back(op.second);
        continue;
      }
      if (op_type == ge::FRAMEWORKOP) {
        op.second.GetAttr(ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, op_type);
        if (op_type == type) {
          ops.push_back(op.second);
        }
      }
    }
    return GRAPH_SUCCESS;
  }

  void SetNeedIteration(bool need_iteration) {
    if (compute_graph_ == nullptr) {
      GELOGE(GRAPH_FAILED, "Set need iteration failed, as compute graph is null.");
      return;
    }
    compute_graph_->SetNeedIteration(need_iteration);
  }

  const std::string &GetName() const { return name_; }

 private:
  std::string name_;
  std::string output_name_;
  std::map<string, ge::Operator> op_list_;
  ComputeGraphPtr compute_graph_{nullptr};
};

Graph::Graph(const std::string &name) {
  impl_ = ComGraphMakeShared<GraphImpl>(name);
  if (impl_ == nullptr) {
    GELOGW("GraphImpl make shared failed, impl_ is nullptr");
  }
}

graphStatus Graph::AddOp(const ge::Operator &op) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, return GRAPH_FAILED, "AddOp failed: graph can not be used, impl is nullptr.");
  return impl_->AddOp(op);
}

graphStatus Graph::GetAllOpName(std::vector<string> &op_name) const {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, return GRAPH_FAILED,
                   "GetAllOpName failed: graph can not be used, impl is nullptr.");
  return impl_->GetAllOpName(op_name);
}

graphStatus Graph::FindOpByName(const std::string &name, Operator &op) const {
  Operator op_find_op_def("NULL");
  op = op_find_op_def;
  GE_CHK_BOOL_EXEC(impl_ != nullptr, return GRAPH_FAILED,
                   "FindOpByName failed: graph can not be used, impl is nullptr.");
  return impl_->FindOpByName(name, op);
}

graphStatus Graph::FindOpByType(const string &type, std::vector<ge::Operator> &ops) const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->FindOpByType(type, ops);
}

Graph &Graph::SetInputs(const vector<ge::Operator> &inputs) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, return *this, "SetInputs failed: graph can not be used, impl is nullptr.")
  GE_CHK_BOOL_EXEC(inputs.size() > 0, return *this, "SetInputs failed: input operator size can not be 0.");
  (void)impl_->SetInputs(inputs);
  return *this;
}

Graph &Graph::SetOutputs(const vector<ge::Operator> &outputs) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "SetOutputs failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetOutputs(outputs);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexs) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "SetOutputs failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetOutputs(output_indexs);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<pair<Operator, string>> &outputs) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, return *this, "SetOutputs failed: graph can not be used, impl is nullptr.")
  (void)impl_->SetOutputs(outputs);
  return *this;
}

Graph &Graph::SetTargets(const vector<ge::Operator> &targets) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "SetTargets failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetTargets(targets);
  return *this;
}

bool Graph::IsValid() const {
  if (impl_ == nullptr) {
    return false;
  }
  return impl_->IsValid();
}

void Graph::SetNeedIteration(bool need_iteration) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "Set need iteration failed, as impl is null.");
    return;
  }
  impl_->SetNeedIteration(need_iteration);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr GraphUtils::GetComputeGraph(const ge::Graph &graph) {
  GE_CHK_BOOL_EXEC_NOLOG(graph.IsValid(), return nullptr);
  return graph.impl_->compute_graph_;
}

graphStatus Graph::SaveToFile(const string &file_name) const {
  Model model = Model();
  model.SetGraph(*this);
  return model.SaveToFile(file_name);
}

graphStatus Graph::LoadFromFile(const string &file_name) {
  Model model = Model();
  graphStatus ret = model.LoadFromFile(file_name);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  *this = model.GetGraph();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::string &Graph::GetName() const { return impl_->GetName(); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Graph
GraphUtils::CreateGraphFromComputeGraph(const ge::ComputeGraphPtr compute_graph) {
  GE_CHK_BOOL_EXEC_NOLOG(compute_graph != nullptr, return Graph(""));

  auto name = compute_graph->GetName();
  auto graph = Graph(name);

  GE_CHK_BOOL_EXEC_NOLOG(graph.impl_ != nullptr, return graph);
  graph.impl_->compute_graph_ = compute_graph;

  return graph;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RecoverGraphOperators(const Graph &graph) {
  GE_CHECK_NOTNULL(graph.impl_);
  GE_CHECK_NOTNULL(graph.impl_->compute_graph_);

  graph.impl_->op_list_.clear();
  for (const auto &node : graph.impl_->compute_graph_->GetDirectNode()) {
    graph.impl_->op_list_[node->GetName()] = OpDescUtils::CreateOperatorFromNode(node);
  }
  return SUCCESS;
}
}  // namespace ge
