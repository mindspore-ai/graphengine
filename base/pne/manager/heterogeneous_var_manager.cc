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

#include "pne/manager/heterogeneous_var_manager.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
constexpr int32_t kVarStateCompiled = 1;
}  // namespace
std::map<uint64_t, std::shared_ptr<HeterogeneousVarManager>> HeterogeneousVarManager::var_manager_map_;
std::mutex HeterogeneousVarManager::mu_;

Status HeterogeneousVarManager::Initialize(const uint64_t session_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto &instance = var_manager_map_[session_id];
  if (instance == nullptr) {
    instance = MakeShared<HeterogeneousVarManager>();
  }
  GE_CHECK_NOTNULL(instance, ", failed to create HeterogeneousVarManager");
  return SUCCESS;
}

void HeterogeneousVarManager::Finalize(const uint64_t session_id) {
  std::lock_guard<std::mutex> lk(mu_);
  var_manager_map_.erase(session_id);
}

std::shared_ptr<HeterogeneousVarManager> HeterogeneousVarManager::GetInstance(const uint64_t session_id) {
  std::lock_guard<std::mutex> lk(mu_);
  return var_manager_map_[session_id];
}

void HeterogeneousVarManager::SetInitGraphNode(const GraphNodePtr &graph_node) {
  graph_nodes_[graph_node->GetGraphId()] = graph_node;
  GELOGI("Init graph node cached, id = %u", graph_node->GetGraphId());
}

const std::map<uint32_t, GraphNodePtr> &HeterogeneousVarManager::GetInitGraphNodes() const {
  return graph_nodes_;
}

bool HeterogeneousVarManager::IsSuspended(const uint32_t graph_id) const {
  return graph_nodes_.count(graph_id) > 0U;
}

Status HeterogeneousVarManager::RecordInitOp(const uint32_t graph_id, const std::vector<GeTensor> &inputs) {
  HeterogeneousVarManager::InitVarOperation operation{};
  operation.graph_id = graph_id;
  operation.inputs = inputs;
  for (size_t i = 0U; i < inputs.size(); ++i) {
    auto &recorded_input = operation.inputs[i];
    recorded_input.MutableData().clear();
    GE_CHK_GRAPH_STATUS_RET(recorded_input.SetData(inputs[i].GetData()));
  }
  pending_init_operations_[graph_id].emplace_back(std::move(operation));
  GELOGI("Init operation recorded, graph id = %u, num_inputs = %zu", graph_id, inputs.size());
  return SUCCESS;
}

void HeterogeneousVarManager::UpdateVarDeployments(const std::map<std::string, DeploymentInfo> &var_deployments) {
  for (const auto &var_name_and_deployment : var_deployments) {
    auto &var_deployment = var_deployments_[var_name_and_deployment.first];
    var_deployment.state = kVarStateCompiled;
    var_deployment.deployment_info = var_name_and_deployment.second;
    GELOGD("The state of variable [%s] transferred to COMPILED", var_name_and_deployment.first.c_str());
  }
}

Status HeterogeneousVarManager::RegisterInitModel(const FlowModelPtr &flow_model,
                                                  const std::vector<size_t> &data_indices) {
  const auto &root_graph = flow_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  const auto graph_id = root_graph->GetGraphID();
  GE_CHK_BOOL_RET_STATUS(!flow_model->GetSubmodels().empty(),
                         PARAM_INVALID,
                         "submodel is empty, graph_id = %u",
                         graph_id);
  auto &partial_models = graph_id_to_partial_models_[graph_id];
  GE_CHK_BOOL_RET_STATUS(partial_models.empty(),
                         UNSUPPORTED,
                         "Multiply partial models are NOT supported yet, graph_id = %u",
                         graph_id);
  PartialModel partial_model{};
  partial_model.model_id = UINT32_MAX;
  partial_model.flow_model = flow_model;
  partial_model.input_indices = data_indices;
  partial_models.emplace_back(std::move(partial_model));
  GELOGI("Suspended model initialized, graph_id = %u", graph_id);
  return SUCCESS;
}

Status HeterogeneousVarManager::LoadPendingModels(const HeterogeneousVarManager::LoadModelFunc &load_model_func) {
  for (auto &graph_id_and_partial_models : graph_id_to_partial_models_) {
    const auto graph_id = graph_id_and_partial_models.first;
    auto &partial_models = graph_id_and_partial_models.second;
    for (auto &partial_model : partial_models) {
      if (partial_model.model_id == UINT32_MAX) {
        GELOGD("Load pending model, graph_id = %u", graph_id);
        const std::map<uint32_t, GraphNodePtr>::const_iterator &it = graph_nodes_.find(graph_id);
        GE_CHK_BOOL_RET_STATUS(it != graph_nodes_.cend(), FAILED, "graph node NOT found, graph_id = %u", graph_id);
        const auto &graph_node = it->second;
        GE_CHECK_NOTNULL(graph_node);
        GE_CHK_STATUS_RET(load_model_func(partial_model.flow_model, graph_node),
                          "Failed to load model");
        partial_model.model_id = partial_model.flow_model->GetModelId();
        GELOGD("Pending model loaded, graph_id = %u, model_id = %u", graph_id, partial_model.model_id);
      }
    }
  }
  return SUCCESS;
}

Status HeterogeneousVarManager::ExecutePendingInitOps(const ExecModelFunc &execute_model_func) {
  std::vector<uint32_t> to_remove;
  for (auto &graph_id_and_init_op : pending_init_operations_) {
    const auto graph_id = graph_id_and_init_op.first;
    auto &partial_models = graph_id_to_partial_models_[graph_id];
    if (partial_models.empty() || (partial_models.front().model_id == UINT32_MAX)) {
      GELOGI("init model not built yet, graph_id = %u", graph_id);
      continue;
    }
    auto &init_ops = graph_id_and_init_op.second;
    GE_CHK_STATUS_RET_NOLOG(ExecutePendingInitOps(graph_id, partial_models, init_ops, execute_model_func));
    GELOGI("Pending init operation executed successfully, graph_id = %u", graph_id);
    to_remove.emplace_back(graph_id);
  }
  for (const auto graph_id : to_remove) {
    pending_init_operations_.erase(graph_id);
  }
  return SUCCESS;
}

Status HeterogeneousVarManager::ExecutePendingInitOps(
    const uint32_t graph_id,
    std::vector<PartialModel> &partial_models,
    std::vector<InitVarOperation> &init_ops,
    const HeterogeneousVarManager::ExecModelFunc &execute_model_func) {
  const auto &graph_node = graph_nodes_[graph_id];
  GE_CHECK_NOTNULL(graph_node, ", graph node is null, graph id = %u", graph_id);
  for (auto &init_op : init_ops) {
    auto &partial_model = partial_models.front();  // only support one for now
    GELOGD("Execute pending model, graph_id = %u", graph_id);
    graph_node->SetFlowModel(partial_model.flow_model);
    std::vector<GeTensor> partial_inputs;
    GE_CHK_STATUS_RET_NOLOG(GetPartialModelInput(partial_model, init_op.inputs, partial_inputs));
    GE_CHK_STATUS_RET(execute_model_func(graph_node, partial_inputs),
                      "Failed to execute model, graph_id = %u, model_id = %u", graph_id, partial_model.model_id);
    GELOGD("Pending model executed, graph_id = %u, model_id = %u", graph_id, partial_model.model_id);
    init_op.inputs.clear();
  }
  return SUCCESS;
}

Status HeterogeneousVarManager::GetPartialModelInput(const HeterogeneousVarManager::PartialModel &partial_model,
                                                     const std::vector<GeTensor> &inputs,
                                                     std::vector<GeTensor> &partial_inputs) {
  for (const auto &input_index : partial_model.input_indices) {
    GE_CHK_BOOL_RET_STATUS(input_index < inputs.size(),
                           PARAM_INVALID,
                           "input out of range, model_id = %u, index = %zu, total input size = %zu",
                           partial_model.model_id, input_index, inputs.size());
    partial_inputs.emplace_back(inputs[input_index]);
  }
  return SUCCESS;
}

const HeterogeneousVarManager::DeploymentInfo *HeterogeneousVarManager::GetVarDeployment(
    const std::string &var_name) const {
  const decltype(var_deployments_)::const_iterator it = var_deployments_.find(var_name);
  if (it == var_deployments_.cend()) {
    return nullptr;
  }
  return &it->second.deployment_info;
}

Status HeterogeneousVarManager::UnloadGraph(const uint32_t graph_id, const UnloadModelFunc &load_model_func) {
  (void) graph_nodes_.erase(graph_id);
  (void) pending_init_operations_.erase(graph_id);
  Status ret = SUCCESS;
  for (const auto &partial_model : graph_id_to_partial_models_[graph_id]) {
    if (load_model_func(partial_model.flow_model, graph_id) != SUCCESS) {
      ret = FAILED;
      GELOGW("Failed to unload model, model_id = %u, graph_id = %u", partial_model.flow_model->GetModelId(), graph_id);
    } else {
      GELOGI("unload model successfully, model_id = %u, graph_id = %u",
             partial_model.flow_model->GetModelId(),
             graph_id);
    }
  }
  (void) graph_id_to_partial_models_.erase(graph_id);
  return ret;
}
}  // namespace ge
