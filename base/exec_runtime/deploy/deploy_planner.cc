/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "exec_runtime/deploy/deploy_planner.h"

namespace ge {
namespace {
constexpr int32_t kLocalNodeId = 0;
constexpr uint32_t kDepDefQueDepth = 128U;
constexpr uint32_t kMaxQueueNameLen = 127U;
const DeployPlan::DeviceInfo kLocalDeviceInfo{CPU, kLocalNodeId, 0};
}  // namespace

std::atomic<int64_t> DeployPlannerBase::endpoint_name_id_gen_{};
std::atomic<int64_t> DeployPlannerBase::plan_id_gen_{};

const std::vector<DeployPlan::QueueInfo> &DeployPlan::GetQueueInfoList() const {
  return queues_;
}

const std::vector<DeployPlan::QueueInfo> &DeployPlan::GetGroupEntryInfoList() const {
  return group_entries_;
}

const std::vector<std::pair<int32_t, int32_t>> &DeployPlan::GetQueueBindings() const {
  return queue_bindings_;
}

const std::vector<int32_t> &DeployPlan::GetInputQueueIndices() const {
  return root_model_info_.input_queue_indices;
}

const std::vector<int32_t> &DeployPlan::GetOutputQueueIndices() const {
  return root_model_info_.output_queue_indices;
}

const std::map<std::string, DeployPlan::SubmodelInfo> &DeployPlan::GetSubmodels() const {
  return submodels_;
}

std::map<std::string, DeployPlan::SubmodelInfo> &DeployPlan::MutableSubmodels() {
  return submodels_;
}

const std::map<int32_t, std::vector<int32_t>> &DeployPlan::GetGroups() const {
  return groups_;
}

bool DeployPlan::IsGroupEndpoint(const int32_t queue_index) const {
  return groups_.find(queue_index) != groups_.end();
}

const std::vector<HcomCommGroup> &DeployPlan::GetCommGroups() const {
  return comm_groups_;
}

void DeployPlan::AddCommGroup(const HcomCommGroup &comm_group) {
  comm_groups_.emplace_back(comm_group);
}

Status DeployPlan::GetQueueInfo(const int32_t queue_index, const DeployPlan::QueueInfo *&queue_info) const {
  if ((queue_index < 0) || (static_cast<size_t>(queue_index) >= queues_.size())) {
    GELOGE(PARAM_INVALID, "Queue index(%d) out of range: [0, %zu)", queue_index, queues_.size());
    return PARAM_INVALID;
  }
  queue_info = &queues_[static_cast<size_t>(queue_index)];
  return SUCCESS;
}

std::vector<int32_t> DeployPlan::GetAllInputQueueIndices() const {
  auto all_indices = root_model_info_.input_queue_indices;
  (void) all_indices.insert(all_indices.cend(),
                            root_model_info_.control_input_queue_indices.cbegin(),
                            root_model_info_.control_input_queue_indices.cend());
  return all_indices;
}

const std::vector<int32_t> &DeployPlan::GetControlInputQueueIndices() const {
  return root_model_info_.control_input_queue_indices;
}

const std::vector<int32_t> &DeployPlan::GetControlOutputQueueIndices() const {
  return root_model_info_.control_output_queue_indices;
}

DeployPlan::DeviceInfo::DeviceInfo(const int32_t type, const int32_t node_id, const int32_t device_id) noexcept
    : type_(type), node_id_(node_id), device_id_(device_id) {
  key_ = std::to_string(type) + "_" + std::to_string(node_id) + "_" + std::to_string(device_id);
}

int32_t DeployPlan::DeviceInfo::GetType() const {
  return type_;
}

int32_t DeployPlan::DeviceInfo::GetNodeId() const {
  return node_id_;
}

int32_t DeployPlan::DeviceInfo::GetDeviceId() const {
  return device_id_;
}

const std::string &DeployPlan::DeviceInfo::GetKey() const {
  return key_;
}

const std::string &DeployPlan::DeviceInfo::GetDesc() const {
  return key_;
}

DeployPlanner::DeployPlanner(const PneModelPtr &root_model)
    : DeployPlannerBase(), root_model_(root_model) {
}

Status DeployPlannerBase::BuildPlan(DeployPlan &deploy_plan) {
  GE_CHK_STATUS_RET(Initialize(), "Failed to initialize deploy planner.");
  GE_CHK_STATUS_RET(ParseModelRelation(), "Failed to parse model relation.");
  plan_id_gen_++;
  deploy_plan = std::move(deploy_plan_);
  return SUCCESS;
}

Status DeployPlannerBase::Initialize() {
  GE_CHK_STATUS_RET(PrepareModelsAndRelation(model_relation_), "Failed to prepare");
  UpdateRelationForControlIo();  // add control input/output for submodels if needed
  relation_reader_ = MakeUnique<ModelRelationReader>(model_relation_);
  GE_CHECK_NOTNULL(relation_reader_);
  GE_CHK_STATUS_RET(relation_reader_->Initialize(), "Failed to initialize model relation reader");
  const auto &root_model_queue_info = model_relation_.root_model_queue_info;
  head_model_queue_info_.output_queue_names = root_model_queue_info.input_queue_names;
  head_model_queue_info_.external_output_queue_names = root_model_queue_info.external_input_queue_names;
  head_model_queue_info_.model_name = "__head";
  tail_model_queue_info_.input_queue_names = root_model_queue_info.output_queue_names;
  tail_model_queue_info_.external_input_queue_names = root_model_queue_info.external_output_queue_names;
  tail_model_queue_info_.model_name = "__tail";
  return SUCCESS;
}

Status DeployPlanner::PrepareModelsAndRelation(ModelRelation &model_relation) {
  GE_CHECK_NOTNULL(root_model_->GetModelRelation().get());
  ModelRelationFlattener flattener(root_model_);
  std::map<std::string, PneModelPtr> name_to_models;
  GE_CHK_STATUS_RET_NOLOG(flattener.Flatten(model_relation, name_to_models));
  GE_CHK_STATUS_RET_NOLOG(ValidateModelAndRelation(name_to_models, model_relation));
  for (const auto &it : name_to_models) {
    const auto &model_name = it.first;
    const auto &submodel = it.second;
    auto &submodel_info = MutableSubmodelInfo(model_name);
    submodel_info.model = submodel;
    submodel_info.device_info = kLocalDeviceInfo;
    GELOGD("Model [%s] will be deployed on device [%d]", model_name.c_str(), submodel_info.device_info.GetNodeId());
  }
  return SUCCESS;
}

void DeployPlannerBase::UpdateForInputControlIo() {
  std::vector<std::string> models_without_input;
  for (const auto &it : model_relation_.submodel_queue_infos) {
    const auto &submodel_queue_info = it.second;
    if (submodel_queue_info.input_queue_names.empty() && submodel_queue_info.external_input_queue_names.empty()) {
      // need control input queue
      // all empty goes to LoadModelWithoutQ for now
      if (!submodel_queue_info.output_queue_names.empty()) {
        GELOGI("submodel [%s] needs control input", it.first.c_str());
        models_without_input.emplace_back(it.first);
      }
    }
  }

  if (!models_without_input.empty()) {
    const std::string control_input_queue_name = "__control_input";
    ModelRelation::QueueDef queue_def{};
    queue_def.name = control_input_queue_name;
    queue_def.depth = kDepDefQueDepth;
    queue_def.is_control_ = true;
    model_relation_.queue_defs.emplace_back(queue_def);
    model_relation_.root_model_queue_info.input_queue_names.emplace_back(control_input_queue_name);
    for (const auto &model_name : models_without_input) {
      model_relation_.submodel_queue_infos[model_name].input_queue_names.emplace_back(
          control_input_queue_name);
    }
  }
}

void DeployPlannerBase::UpdateForOutputControlIo() {
  std::map<std::string, std::vector<std::string>> models_without_output;
  for (const auto &it : model_relation_.submodel_queue_infos) {
    const auto &submodel_queue_info = it.second;
    if (submodel_queue_info.output_queue_names.empty()) {
      // need control input queue
      // all empty goes to LoadModelWithoutQ for now
      if (!submodel_queue_info.input_queue_names.empty() || !submodel_queue_info.external_input_queue_names.empty()) {
        GELOGI("submodel [%s] needs control output", it.first.c_str());
        models_without_output[submodel_queue_info.model_name].emplace_back(it.first);
      }
    }
  }

  for (const auto &it : models_without_output) {
    const auto &model_name = it.first;
    const std::string control_output_queue_name = "__" + model_name + "_control_output";
    ModelRelation::QueueDef queue_def{};
    queue_def.name = control_output_queue_name;
    queue_def.depth = kDepDefQueDepth;
    queue_def.is_control_ = true;
    model_relation_.queue_defs.emplace_back(queue_def);
    model_relation_.root_model_queue_info.output_queue_names.emplace_back(control_output_queue_name);
    for (const auto &model_instance_name : it.second) {
      model_relation_.submodel_queue_infos[model_instance_name].output_queue_names.emplace_back(
          control_output_queue_name);
    }
  }
}

void DeployPlannerBase::UpdateRelationForControlIo() {
  UpdateForInputControlIo();
  UpdateForOutputControlIo();
}

Status DeployPlannerBase::ValidateModelAndRelation(const std::map<std::string, PneModelPtr> &models,
                                                   const ModelRelation &model_relation) {
  // check all model in model_relation exist in RootModel
  for (const auto &it : model_relation.submodel_queue_infos) {
    const auto &model_instance_name = it.first;
    const auto &submodel = models.find(model_instance_name);
    if (submodel == models.end()) {
      GELOGE(PARAM_INVALID,
             "model exists in ModelRelation bot not found in RootModel, name = %s",
             model_instance_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::ParseModelRelation() {
  Mark2PgModels();
  GE_CHK_STATUS_RET(AssignEnqueueQueues(), "Failed to assign enqueue queues");
  GE_CHK_STATUS_RET(ResolveDataFlows(), "Failed to resolve flow relations");
  LogDataFlow();
  GE_CHK_STATUS_RET(ResolveReusableQueues(), "Failed to resolve reusable queues");
  GE_CHK_STATUS_RET(AssignDequeueQueues(), "Failed to assign dequeue queues");
  GE_CHK_STATUS_RET(BindOutputToRemoteInputs(), "Failed to bind output groups");
  GE_CHK_STATUS_RET(BindRemoteOutputGroupToInput(), "Failed to bind input groups");
  UpdateDeployPlan();
  return SUCCESS;
}

Status DeployPlannerBase::AssignEnqueueQueues() {
  GE_CHK_STATUS_RET_NOLOG(CreateOutputEndpoints(head_model_queue_info_.model_name,
                                                head_model_queue_info_.output_queue_names));
  GE_CHK_STATUS_RET_NOLOG(CreateOutputEndpoints(head_model_queue_info_.model_name,
                                                head_model_queue_info_.external_output_queue_names,
                                                false));
  for (const auto &it : model_relation_.submodel_queue_infos) {
    const auto &model_instance_name = it.first;
    GE_CHK_STATUS_RET_NOLOG(CreateOutputEndpoints(model_instance_name, it.second.output_queue_names));
    const auto &invoke_model_keys = it.second.invoke_model_keys;
    for (const auto &invoke_model_key : invoke_model_keys) {
      auto invoked_model_queue_info = relation_reader_->GetInvokedModelQueueInfo(invoke_model_key);
      GE_CHECK_NOTNULL(invoked_model_queue_info, ", get invoked model queue info is null, model_name = %s",
                       model_instance_name.c_str());
      // invoked model input is as feed for this model
      GE_CHK_STATUS_RET_NOLOG(
          CreateFeedEndpoints(model_instance_name, invoked_model_queue_info->input_queue_names, invoke_model_key));
    }
  }
  return SUCCESS;
}

void DeployPlannerBase::Mark2PgModels() {
  std::map<std::string, std::map<int32_t, std::vector<int32_t>>> model_placements;
  std::map<std::string, std::vector<std::string>> model_instances;
  for (const auto &it : model_relation_.submodel_queue_infos) {
    const auto &model_instance_name = it.first;
    const auto &submodel_queue_info = it.second;
    const auto &device_info = MutableSubmodelInfo(model_instance_name).device_info;
    model_placements[submodel_queue_info.model_name][device_info.GetNodeId()].emplace_back(device_info.GetDeviceId());
    model_instances[submodel_queue_info.model_name].emplace_back(model_instance_name);
  }
  for (const auto &it : model_placements) {
    const auto &model_name = it.first;
    const auto &devices = it.second;
    if (devices.begin()->second.size() > 1U) {
      GELOGD("Submodel [%s] is deploy to multiple devices", model_name.c_str());
      deploy_to_devlist_.emplace(model_name);
      for (const auto &model_instance_name : model_instances[model_name]) {
        deploy_to_devlist_.emplace(model_instance_name);
      }
    }
  }
}

Status DeployPlannerBase::ResolveDataFlows() {
  for (const auto &it : model_relation_.submodel_queue_infos) {
    const auto &model_instance_name = it.first;
    const auto &submodel_queue_info = it.second;
    GE_CHK_STATUS_RET_NOLOG(ResolveModelInputs(model_instance_name, submodel_queue_info));
  }
  GE_CHK_STATUS_RET_NOLOG(ResolveModelInputs(tail_model_queue_info_.model_name, tail_model_queue_info_));
  return SUCCESS;
}

Status DeployPlannerBase::ResolveModelInputs(const std::string &model_instance_name,
                                             const ModelRelation::ModelQueueInfo &model_queue_info) {
  std::vector<const ModelRelation::QueueDef *> model_input_queue_defs;
  GE_CHK_STATUS_RET_NOLOG(relation_reader_->BatchGetQueueDefs(model_queue_info.input_queue_names,
                                                              model_input_queue_defs));
  GE_CHK_STATUS_RET_NOLOG(relation_reader_->BatchGetQueueDefs(model_queue_info.external_input_queue_names,
                                                              model_input_queue_defs));
  std::vector<ModelQueueIndex> model_queue_ids;
  model_queue_ids.reserve(model_input_queue_defs.size());
  for (size_t input_index = 0UL; input_index < model_queue_info.input_queue_names.size(); ++input_index) {
    ModelQueueIndex input_queue_index{model_queue_info.model_name, "", static_cast<int32_t>(input_index)};
    model_queue_ids.emplace_back(std::move(input_queue_index));
  }
  // external index mark as -1;
  ModelQueueIndex external_queue_index{model_queue_info.model_name, "", -1};
  model_queue_ids.resize(model_input_queue_defs.size(), external_queue_index);

  for (const auto &invoke_model_key : model_queue_info.invoke_model_keys) {
    auto invoked_model_queue_info = relation_reader_->GetInvokedModelQueueInfo(invoke_model_key);
    // denpend output is as fetch input here
    GE_CHK_STATUS_RET_NOLOG(
        relation_reader_->BatchGetQueueDefs(invoked_model_queue_info->output_queue_names, model_input_queue_defs));
    for (size_t feed_index = 0UL; feed_index < invoked_model_queue_info->output_queue_names.size(); ++feed_index) {
      ModelQueueIndex feed_queue_index{model_queue_info.model_name, invoke_model_key, static_cast<int32_t>(feed_index)};
      model_queue_ids.emplace_back(std::move(feed_queue_index));
    }
  }

  GE_CHK_BOOL_RET_STATUS(model_queue_ids.size() == model_input_queue_defs.size(), INTERNAL_ERROR,
                         "model_queue_ids.size=%zu is not same as model_input_queue_defs.size=%zu, model=%s",
                         model_queue_ids.size(), model_input_queue_defs.size(), model_instance_name.c_str());

  auto &submodel_info = MutableSubmodelInfo(model_instance_name);
  for (size_t i = 0UL; i < model_input_queue_defs.size(); ++i) {
    const auto &model_queue_id = model_queue_ids[i];
    const auto *queue_def = model_input_queue_defs[i];
    const auto &queue_name = queue_def->name;
    const auto &src_endpoint_indices = src_endpoint_indices_[queue_name];
    if (src_endpoint_indices.empty()) {
      GELOGE(PARAM_INVALID, "Failed to find enqueue operation for queue [%s]", queue_name.c_str());
      return PARAM_INVALID;
    }
    const bool dst_is_devlist = deploy_to_devlist_.find(model_instance_name) != deploy_to_devlist_.cend();
    for (auto src_endpoint_index : src_endpoint_indices) {
      const auto &src_endpoint = deploy_plan_.queues_[static_cast<size_t>(src_endpoint_index)];
      // group to group : only deal with bind_relation in same node and same device
      if (dst_is_devlist && (deploy_to_devlist_.find(src_endpoint.model_instance_name) != deploy_to_devlist_.cend())) {
        const auto &src_device_info = src_endpoint.device_info;
        if ((src_device_info.GetNodeId() != submodel_info.device_info.GetNodeId()) ||
            (src_endpoint.device_info.GetDeviceId() != submodel_info.device_info.GetDeviceId())) {
          GELOGD("Skip bind endpoints: name = %s, from %s to %s:%d@%s",
                 queue_name.c_str(),
                 src_endpoint.model_instance_name.c_str(),
                 model_queue_info.model_name.c_str(),
                 model_queue_id.id,
                 submodel_info.device_info.GetDesc().c_str());
          continue;
        }
      }

      auto &dst_endpoint_groups = endpoint_pairs_[src_endpoint_index];
      auto queue_info = BuildQueueInfo(*queue_def, model_instance_name);
      queue_info.name = GetEndpointFullName(queue_info, model_queue_id);
      dst_endpoint_groups[model_queue_id].emplace_back(std::move(queue_info));
      GELOGD("Bind endpoints: name = %s, from %s to %s:%d@%s, invoke_key=%s.",
             queue_name.c_str(),
             src_endpoint.model_instance_name.c_str(),
             model_queue_info.model_name.c_str(),
             model_queue_id.id,
             submodel_info.device_info.GetDesc().c_str(),
             model_queue_id.invoke_key.c_str());
    }
  }
  return SUCCESS;
}

void DeployPlannerBase::LogDataFlow() const {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return;
  }

  for (const auto &endpoint_pair : endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    const auto &src_endpoint_info = deploy_plan_.queues_[src_endpoint_idx];

    std::map<ModelQueueIndex, std::vector<std::string>> group_by_dst_loc;
    for (const auto &dst_loc_and_queue_info : endpoint_pair.second) {
      for (const auto &dst_queue_info : dst_loc_and_queue_info.second) {
        group_by_dst_loc[dst_loc_and_queue_info.first].emplace_back(dst_queue_info.device_info.GetDesc());
      }
    }
    GELOGD("Bindings for queue [%s@%s] are:",
           src_endpoint_info.name.c_str(),
           src_endpoint_info.device_info.GetDesc().c_str());
    for (const auto &it : group_by_dst_loc) {
      GELOGD("    %s:%d@%s", it.first.model_name.c_str(), it.first.id, ToString(it.second).c_str());
    }
  }
}

Status DeployPlannerBase::ResolveReusableQueues() {
  for (const auto &endpoint_pair : endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    const auto &src_endpoint_info = deploy_plan_.queues_[src_endpoint_idx];
    const auto &queue_name = src_endpoint_info.name;
    if (endpoint_pair.second.size() != 1U) {
      GELOGD("Queue[%s@%s] has one-to-many relation to models",
             queue_name.c_str(),
             src_endpoint_info.device_info.GetKey().c_str());
      continue;
    }

    const auto &dst_queue_infos = *endpoint_pair.second.begin();
    if (dst_queue_infos.second.size() != 1U) {
      GELOGD("Queue[%s@%s] has multi-device dest endpoints",
             queue_name.c_str(),
             src_endpoint_info.device_info.GetKey().c_str());
      continue;
    }

    const auto &dst_device_info = dst_queue_infos.second.begin()->device_info;
    if (src_endpoint_info.device_info.GetKey() != dst_device_info.GetKey()) {
      GELOGD("Queue[%s@%s] has remote dest endpoints, device = [%s]",
             queue_name.c_str(),
             src_endpoint_info.device_info.GetKey().c_str(),
             dst_device_info.GetDesc().c_str());
      continue;
    }

    GELOGD("Queue[%s@%s] is reusable, index = %d",
           queue_name.c_str(),
           src_endpoint_info.device_info.GetKey().c_str(),
           src_endpoint_idx);
    (void)reusable_queue_indices_.emplace(src_endpoint_idx);
  }
  return SUCCESS;
}

Status DeployPlannerBase::AssignDequeueQueues() {
  // key order [model_instance_name][invoke_key][index]
  std::map<std::string, std::map<std::string, std::map<int32_t, int32_t>>> model_input_indices;
  std::map<std::string, int32_t> external_input_indices;
  std::map<std::string, std::set<int32_t>> model_control_input_indices;
  for (const auto &endpoint_pair : endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    // group by model_and_input_idx
    for (const auto &queue_loc_and_queue_infos : endpoint_pair.second) {
      const auto &model_queue_loc = queue_loc_and_queue_infos.first;
      for (const auto &queue_info : queue_loc_and_queue_infos.second) {
        int32_t dst_endpoint_idx = -1;
        const auto &model_instance_name = queue_info.model_instance_name;
        if (reusable_queue_indices_.count(src_endpoint_idx) > 0UL) {
          GELOGD("Reuse src queue, queue name = %s, queue index = %d",
                 deploy_plan_.queues_[src_endpoint_idx].name.c_str(),
                 src_endpoint_idx);
          dst_endpoint_idx = src_endpoint_idx;
        } else {
          GE_CHK_STATUS_RET_NOLOG(GetOrCreateInputEndpoint(model_queue_loc, queue_info, dst_endpoint_idx));
          GE_CHK_STATUS_RET_NOLOG(CreateTags(src_endpoint_idx, dst_endpoint_idx, model_queue_loc, queue_info));
          GELOGD("Endpoint binding added, src = %s, dst = %s",
                 ToEndpointDesc(src_endpoint_idx).c_str(),
                 ToEndpointDesc(dst_endpoint_idx).c_str());
        }
        if (queue_info.is_control) {
          model_control_input_indices[model_instance_name].emplace(dst_endpoint_idx);
        } else if (model_queue_loc.id >= 0) {
          model_input_indices[model_instance_name][model_queue_loc.invoke_key][model_queue_loc.id] = dst_endpoint_idx;
        } else {
          external_input_indices[model_instance_name] = dst_endpoint_idx;
        }
      }
    }
  }
  for (const auto &name_and_input_indices : model_control_input_indices) {
    auto &submodel_info = MutableSubmodelInfo(name_and_input_indices.first);
    submodel_info.control_input_queue_indices.assign(name_and_input_indices.second.cbegin(),
                                                     name_and_input_indices.second.cend());
  }
  for (const auto &name_and_input_indices : model_input_indices) {
    auto &submodel_info = MutableSubmodelInfo(name_and_input_indices.first);
    // group by invoke key
    for (const auto &group_indices : name_and_input_indices.second) {
      const auto &invoke_key = group_indices.first;
      for (const auto &input_index_and_endpoint_index : group_indices.second) {
        if (!invoke_key.empty()) {
          // invoke model's fetch queue as input
          submodel_info.invoked_model_queue_infos[invoke_key].fetch_queue_indices.emplace_back(
              input_index_and_endpoint_index.second);
          continue;
        }
        // already sorted by input index, it's OK to use emplace_back
        submodel_info.input_queue_indices.emplace_back(input_index_and_endpoint_index.second);
      }
    }
  }
  for (const auto &name_and_input_index : external_input_indices) {
    auto &submodel_info = MutableSubmodelInfo(name_and_input_index.first);
    submodel_info.input_queue_indices.emplace_back(name_and_input_index.second);
  }
  return SUCCESS;
}

Status DeployPlannerBase::CreateTags(const int32_t src_endpoint_idx,
                                     const int32_t dst_endpoint_idx,
                                     const ModelQueueIndex &model_queue_loc,
                                     const DeployPlan::QueueInfo &queue_info) {
  int32_t dst_tag_idx = -1;
  // Queue -> Queue
  if (deploy_plan_.queues_[src_endpoint_idx].device_info.GetKey() == queue_info.device_info.GetKey()) {
    auto entry_info = queue_info;
    entry_info.ref_index = dst_endpoint_idx;
    GE_CHK_STATUS_RET_NOLOG(CreateGroupEntry(entry_info, dst_tag_idx));
    (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(queue_info.device_info.GetKey(), dst_tag_idx);
    GELOGD("%s add input queue %s",
           ToEndpointDesc(dst_endpoint_idx).c_str(),
           ToEndpointDesc(src_endpoint_idx).c_str());
    return SUCCESS;
  }

  // In src device, create output Queue -> Group of Tags
  GE_CHK_STATUS_RET_NOLOG(CreateGroupEntry(queue_info, dst_tag_idx));
  (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(queue_info.device_info.GetKey(), dst_tag_idx);
  GELOGD("%s add output tag %s, tag_name = %s, dst_device = %s, dst_tag_idx = %d",
         ToEndpointDesc(src_endpoint_idx).c_str(),
         ToEndpointDesc(dst_endpoint_idx).c_str(),
         queue_info.name.c_str(),
         queue_info.device_info.GetDesc().c_str(),
         dst_tag_idx);

  // In dst device, create input group of Tags -> Queue
  int32_t src_tag_idx = -1;
  auto src_tag_info = queue_info;
  src_tag_info.device_info = deploy_plan_.queues_[src_endpoint_idx].device_info;
  GE_CHK_STATUS_RET_NOLOG(CreateGroupEntry(src_tag_info, src_tag_idx));
  input_groups_[dst_endpoint_idx].emplace_back(src_tag_idx);
  GELOGD("%s add input tag %s, tag_name = %s, src_device = %s, src_tag_idx = %d",
         ToEndpointDesc(dst_endpoint_idx).c_str(),
         ToEndpointDesc(src_endpoint_idx).c_str(),
         queue_info.name.c_str(),
         src_tag_info.device_info.GetDesc().c_str(),
         src_tag_idx);
  return SUCCESS;
}

Status DeployPlannerBase::BindRemoteOutputGroupToInput() {
  for (const auto &it : input_groups_) {
    const auto endpoint_index = it.first;
    const auto &input_endpoint_info = deploy_plan_.queues_[endpoint_index];
    DeployPlan::QueueInfo group_info{};
    group_info.name = input_endpoint_info.name;
    group_info.device_info = input_endpoint_info.device_info;
    group_info.model_instance_name = input_endpoint_info.model_instance_name;
    int32_t group_index = -1;
    GE_CHK_STATUS_RET_NOLOG(CreateGroupInfo(group_info, it.second, group_index));
    deploy_plan_.queue_bindings_.emplace_back(group_index, endpoint_index);
    GELOGD("Input group binding added, peer = %s, local = %s@%s",
           ToString(ToEndpointDescs(deploy_plan_.groups_[group_index], true)).c_str(),
           group_info.name.c_str(),
           deploy_plan_.queues_[endpoint_index].device_info.GetDesc().c_str());
  }
  return SUCCESS;
}

Status DeployPlannerBase::BindOutputToRemoteInputs() {
  for (auto &it : output_groups_) {
    const auto endpoint_index = it.first;
    for (auto &grouped_peer_inputs : it.second) {
      const auto &model_queue_loc = grouped_peer_inputs.first;
      const auto &grouped_inputs = grouped_peer_inputs.second;
      DeployPlan::QueueInfo group_info{};
      const auto &output_endpoint_info = deploy_plan_.queues_[endpoint_index];
      group_info.name = output_endpoint_info.name;
      group_info.device_info = output_endpoint_info.device_info;
      group_info.model_instance_name = output_endpoint_info.model_instance_name;
      int32_t group_index = -1;
      std::vector<int32_t> grouped_inputs_order_by_device;
      for (const auto &device_and_index : grouped_inputs) {
        grouped_inputs_order_by_device.emplace_back(device_and_index.second);
      }
      GE_CHK_STATUS_RET(CreateGroupInfo(group_info, grouped_inputs_order_by_device, group_index));
      deploy_plan_.queue_bindings_.emplace_back(endpoint_index, group_index);
      GELOGD("Output group binding added, local = %s@%s, peer model = %s:%d, peer input indices = %s",
             group_info.name.c_str(),
             deploy_plan_.queues_[endpoint_index].device_info.GetDesc().c_str(),
             model_queue_loc.model_name.c_str(),
             model_queue_loc.id,
             ToString(ToEndpointDescs(deploy_plan_.groups_[group_index], true)).c_str());
    }
  }
  return SUCCESS;
}

void DeployPlannerBase::UpdateDeployPlan() {
  deploy_plan_.root_model_info_.input_queue_indices = std::move(head_model_info_.output_queue_indices);
  deploy_plan_.root_model_info_.control_input_queue_indices = std::move(head_model_info_.control_output_queue_indices);
  deploy_plan_.root_model_info_.output_queue_indices = std::move(tail_model_info_.input_queue_indices);
  deploy_plan_.root_model_info_.control_output_queue_indices = std::move(tail_model_info_.control_input_queue_indices);
}

DeployPlan::SubmodelInfo &DeployPlannerBase::MutableSubmodelInfo(const std::string &name) {
  if (name == head_model_queue_info_.model_name) {
    return head_model_info_;
  } else if (name == tail_model_queue_info_.model_name) {
    return tail_model_info_;
  }
  return deploy_plan_.submodels_[name];
}

std::string DeployPlannerBase::ToEndpointDesc(const int32_t endpoint_indices, const bool is_group_entry) const {
  if (is_group_entry) {
    return deploy_plan_.group_entries_[endpoint_indices].name + "@"
        + deploy_plan_.group_entries_[endpoint_indices].device_info.GetDesc();
  } else {
    return deploy_plan_.queues_[endpoint_indices].name + "@"
        + deploy_plan_.queues_[endpoint_indices].device_info.GetDesc();
  }
}

std::vector<std::string> DeployPlannerBase::ToEndpointDescs(const std::vector<int32_t> &endpoint_indices,
                                                            const bool is_group_entry) const {
  std::vector<std::string> ret;
  (void) std::transform(endpoint_indices.cbegin(), endpoint_indices.cend(), std::back_inserter(ret),
                        [this, is_group_entry](const int32_t index) {
                          return ToEndpointDesc(index, is_group_entry);
                        });
  return ret;
}

DeployPlan::QueueInfo DeployPlannerBase::BuildQueueInfo(const ModelRelation::QueueDef &queue_def,
                                                        const std::string &model_instance_name) {
  DeployPlan::QueueInfo queue_info{};
  queue_info.device_info = MutableSubmodelInfo(model_instance_name).device_info;
  queue_info.depth = queue_def.depth;
  queue_info.name = queue_def.name;
  queue_info.model_instance_name = model_instance_name;
  queue_info.is_control = queue_def.is_control_;
  queue_info.enqueue_policy = queue_def.enqueue_policy;
  return queue_info;
}

Status DeployPlannerBase::CreateEndpointInfo(const DeployPlan::QueueInfo &queue_info, int32_t &queue_idx) {
  const auto queue_size = deploy_plan_.queues_.size();
  GE_CHECK_LE(queue_size, static_cast<size_t>(INT32_MAX));
  deploy_plan_.queues_.emplace_back(queue_info);
  queue_idx = static_cast<int32_t>(queue_size);
  return SUCCESS;
}

Status DeployPlannerBase::CreateGroupEntry(const DeployPlan::QueueInfo &queue_info, int32_t &entry_index) {
  const auto entry_size = deploy_plan_.group_entries_.size();
  GE_CHECK_LE(entry_size, static_cast<size_t>(INT32_MAX));
  deploy_plan_.group_entries_.emplace_back(queue_info);
  entry_index = static_cast<int32_t>(entry_size);
  return SUCCESS;
}

Status DeployPlannerBase::CreateGroupInfo(const DeployPlan::QueueInfo &queue_info,
                                          const std::vector<int32_t> &grouped_indices,
                                          int32_t &group_index) {
  GE_CHK_STATUS_RET(CreateEndpointInfo(queue_info, group_index));
  deploy_plan_.groups_[group_index] = grouped_indices;
  GELOGD("Group created, name = %s, group_index = %d, endpoint_indices = %s, endpoint_descs = %s",
         queue_info.name.c_str(),
         group_index,
         ToString(grouped_indices).c_str(),
         ToString(ToEndpointDescs(grouped_indices, true)).c_str());
  return SUCCESS;
}

Status DeployPlannerBase::GetOrCreateInputEndpoint(const ModelQueueIndex &model_queue_index,
                                                   const DeployPlan::QueueInfo &queue_info,
                                                   int32_t &endpoint_index) {
  auto key = std::make_pair(model_queue_index, queue_info.device_info.GetKey());
  const std::map<std::pair<ModelQueueIndex, std::string>, int32_t>::const_iterator
      it = input_endpoint_indices_.find(key);
  if (it != input_endpoint_indices_.cend()) {
    endpoint_index = it->second;
    return SUCCESS;
  }

  GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(queue_info, endpoint_index));
  GELOGD("Input endpoint created, queue name = %s, device = %s, index = %d",
         queue_info.name.c_str(),
         queue_info.device_info.GetDesc().c_str(),
         endpoint_index);
  input_endpoint_indices_[key] = endpoint_index;
  return SUCCESS;
}

Status DeployPlannerBase::CreateOutputEndpoints(const std::string &model_instance_name,
                                                const std::vector<std::string> &queue_names,
                                                const bool is_owned) {
  std::vector<const ModelRelation::QueueDef *> queue_defs;
  GE_CHK_STATUS_RET_NOLOG(relation_reader_->BatchGetQueueDefs(queue_names, queue_defs));
  for (size_t output_idx = 0U; output_idx < queue_defs.size(); ++output_idx) {
    const auto queue_def = queue_defs[output_idx];
    int32_t endpoint_index = -1;
    auto queue_info = BuildQueueInfo(*queue_def, model_instance_name);
    queue_info.owned = is_owned;
    GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(queue_info, endpoint_index));
    src_endpoint_indices_[queue_info.name].emplace_back(endpoint_index);
    if (queue_info.owned) {
      auto &submodel_info = MutableSubmodelInfo(queue_info.model_instance_name);
      if (queue_info.is_control) {
        submodel_info.control_output_queue_indices.emplace_back(endpoint_index);
      } else {
        submodel_info.output_queue_indices.emplace_back(endpoint_index);
      }
    }
    GELOGD("Output endpoint created, model = %s, output_index = %zu, queue name = %s, queue index = %d",
           model_instance_name.c_str(), output_idx, queue_def->name.c_str(), endpoint_index);
  }
  return SUCCESS;
}

Status DeployPlannerBase::CreateFeedEndpoints(const std::string &model_instance_name,
                                              const std::vector<std::string> &queue_names,
                                              const std::string &invoke_key) {
  std::vector<const ModelRelation::QueueDef *> queue_defs;
  GE_CHK_STATUS_RET_NOLOG(relation_reader_->BatchGetQueueDefs(queue_names, queue_defs));
  for (size_t output_idx = 0UL; output_idx < queue_defs.size(); ++output_idx) {
    const auto queue_def = queue_defs[output_idx];
    int32_t endpoint_index = -1;
    auto queue_info = BuildQueueInfo(*queue_def, model_instance_name);
    queue_info.owned = true;
    GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(queue_info, endpoint_index));
    src_endpoint_indices_[queue_info.name].emplace_back(endpoint_index);
    auto &submodel_info = MutableSubmodelInfo(queue_info.model_instance_name);
    submodel_info.invoked_model_queue_infos[invoke_key].feed_queue_indices.emplace_back(endpoint_index);
    GELOGD("Feed endpoint created, model = %s, invoke_key = %s, feed_index = %zu, queue name = %s, queue index = %d",
           model_instance_name.c_str(), invoke_key.c_str(), output_idx, queue_def->name.c_str(), endpoint_index);
  }
  return SUCCESS;
}

std::string DeployPlannerBase::GetEndpointFullName(const DeployPlan::QueueInfo &endpoint_info,
                                                   const DeployPlannerBase::ModelQueueIndex &model_queue_index) {
  std::stringstream ss;
  ss << model_queue_index.model_name << ":" << model_queue_index.id
     << "_FROM_" << endpoint_info.name << "@" << endpoint_info.device_info.GetKey()
     << "_T" << std::to_string(plan_id_gen_);
  const auto &name = ss.str();
  if (name.length() <= kMaxQueueNameLen) {
    return name;
  }

  auto &short_name = short_names_[name];
  if (short_name.empty()) {
    short_name = "deploy_planner.auto_generated:" + std::to_string(endpoint_name_id_gen_++);
    GELOGD("endpoint name too long, change from %s to %s", name.c_str(), short_name.c_str());
  }
  return short_name;
}

Status DeployPlannerBase::CreateTransferInfo(const std::string &route_name,
                                             const DeployPlan::DeviceInfo &src_device_info,
                                             const DeployPlan::DeviceInfo &dst_device_info) {
  DeployPlan::QueueInfo src_queue_info{};
  src_queue_info.name = route_name;
  src_queue_info.device_info = src_device_info;
  src_queue_info.depth = 128U;

  DeployPlan::QueueInfo dst_queue_info{};
  dst_queue_info.name = route_name;
  dst_queue_info.device_info = dst_device_info;
  dst_queue_info.depth = 128U;

  int32_t src_queue_index = -1;
  GE_CHK_STATUS_RET(CreateEndpointInfo(src_queue_info, src_queue_index));
  int32_t dst_tag_index = -1;
  GE_CHK_STATUS_RET_NOLOG(CreateGroupEntry(dst_queue_info, dst_tag_index));
  int32_t dst_tag_group_index = -1;
  std::vector<int32_t> dst_tag_group = {dst_tag_index};
  GE_CHK_STATUS_RET_NOLOG(CreateGroupInfo(src_queue_info, dst_tag_group, dst_tag_group_index));
  deploy_plan_.queue_bindings_.emplace_back(src_queue_index, dst_tag_group_index);

  int32_t src_tag_index = -1;
  GE_CHK_STATUS_RET_NOLOG(CreateGroupEntry(src_queue_info, src_tag_index));
  int32_t dst_queue_index = -1;
  GE_CHK_STATUS_RET(CreateEndpointInfo(dst_queue_info, dst_queue_index));
  int32_t src_tag_group_index = -1;
  std::vector<int32_t> src_tag_group = {src_tag_index};
  GE_CHK_STATUS_RET_NOLOG(CreateGroupInfo(dst_queue_info, src_tag_group, src_tag_group_index));
  deploy_plan_.queue_bindings_.emplace_back(src_tag_group_index, dst_queue_index);
  return SUCCESS;
}

Status DeployPlannerBase::BuildTransferPlan(const std::pair<DeployPlan::DeviceInfo, DeployPlan::DeviceInfo> &routes,
                                            DeployPlan &deploy_plan) {
  const auto &src_device = routes.first;
  const auto &dst_device = routes.second;
  const std::string route_name = src_device.GetKey() + "_to_" + dst_device.GetKey() +
                                 "_T" + std::to_string(plan_id_gen_);
  GE_CHK_STATUS_RET_NOLOG(CreateTransferInfo(route_name, src_device, dst_device));
  plan_id_gen_++;
  deploy_plan = std::move(deploy_plan_);
  return SUCCESS;
}

ModelRelationFlattener::ModelRelationFlattener(PneModelPtr root_model) : root_model_(std::move(root_model)) {}

Status ModelRelationFlattener::Flatten(ModelRelation &flattened_model_relation,
                                       std::map<std::string, PneModelPtr> &name_to_models) {
  const auto &model_relation = root_model_->GetModelRelation();
  GE_CHECK_NOTNULL(model_relation,
                   ", FlowModel's ModelRelation is nullptr, model_name = %s",
                   root_model_->GetModelName().c_str());
  flattened_model_relation_.root_model_queue_info = model_relation->root_model_queue_info;
  flattened_model_relation_.invoked_model_queue_infos = model_relation->invoked_model_queue_infos;
  MergeQueueDefs({}, model_relation->queue_defs);
  for (auto &it : model_relation->submodel_queue_infos) {
    auto &submodel_info_info = it.second;
    auto submodel = root_model_->GetSubmodel(submodel_info_info.model_name);
    GE_CHECK_NOTNULL(submodel, ", Failed to get submodel, submodel_name = %s", submodel_info_info.model_name.c_str());
    GE_CHK_STATUS_RET(FlattenSubmodel(submodel_info_info, submodel, 0),
                      "Failed to flatten submodel %s", submodel_info_info.model_name.c_str());
  }

  flattened_model_relation = std::move(flattened_model_relation_);
  name_to_models = std::move(leaf_models_);
  return SUCCESS;
}

bool ModelRelationFlattener::NeedFlatten(const PneModelPtr &root_model) {
  const auto &submodels = root_model->GetSubmodels();
  for (const auto &submodel : submodels) {
    if (!submodel.second->GetSubmodels().empty()) {
      return true;
    }
  }
  return false;
}

Status ModelRelationFlattener::Flatten(const PneModelPtr &root_model) {
  GE_CHECK_NOTNULL(root_model);
  const auto is_need_flatten = NeedFlatten(root_model);
  if (!is_need_flatten) {
    GELOGD("model is no need flatten, model %s", root_model->GetModelName().c_str());
    return SUCCESS;
  }
  if (root_model->GetModelRelation() == nullptr) {
    GELOGD("model need flatten but relation is null, need build relation, model %s",
           root_model->GetModelName().c_str());
    const auto &root_graph = root_model->GetRootGraph();
    GE_CHECK_NOTNULL(root_graph, ", need build model relation, but root graph is null, model %s",
                     root_model->GetModelName().c_str());
    auto model_relation = MakeShared<ModelRelation>();
    GE_CHECK_NOTNULL(model_relation, ", need build model relation, but make shared failed, model %s",
                     root_model->GetModelName().c_str());
    GE_CHK_STATUS_RET(ModelRelationBuilder().BuildForSingleModel(*root_graph, *model_relation),
                      "Failed to build ModelRelation from root graph: %s.", root_graph->GetName().c_str());
    root_model->SetModelRelation(model_relation);
    GELOGD("make model relation success, model %s", root_model->GetModelName().c_str());
  }
  auto flattened_model_relation = MakeShared<ModelRelation>();
  GE_CHECK_NOTNULL(flattened_model_relation, ", Failed to make flatten model relation for model %s",
                   root_model->GetModelName().c_str());
  ModelRelationFlattener flattener(root_model);
  std::map<std::string, PneModelPtr> flattened_submodels;
  GE_CHK_STATUS_RET_NOLOG(flattener.Flatten(*flattened_model_relation, flattened_submodels));
  root_model->SetModelRelation(flattened_model_relation);
  root_model->SetSubmodels(flattened_submodels);
  GELOGD("model flatten end, model %s", root_model->GetModelName().c_str());
  return SUCCESS;
}

Status ModelRelationFlattener::FlattenSubmodel(const ModelRelation::ModelQueueInfo &parent_model_queue_info,
                                               const PneModelPtr &pne_model,
                                               const int32_t depth) {
  const auto &submodels = pne_model->GetSubmodels();
  if (submodels.empty()) {  // is_leaf
    const auto &model_name = pne_model->GetModelName();
    (void)leaf_models_.emplace(model_name, pne_model);
    (void)flattened_model_relation_.submodel_queue_infos.emplace(model_name, parent_model_queue_info);
    GELOGD("Leaf submodel %s(%s) flattened to parent model %s",
           pne_model->GetModelName().c_str(),
           pne_model->GetModelType().c_str(),
           parent_model_queue_info.model_name.c_str());
    return SUCCESS;
  }

  if (depth >= max_depth_) {
    GELOGE(UNSUPPORTED, "Depth limit(%d) reached", max_depth_);
    return UNSUPPORTED;
  }

  GELOGD("To flatten submodel %s(%s) to parent model %s",
         pne_model->GetModelName().c_str(),
         pne_model->GetModelType().c_str(),
         parent_model_queue_info.model_name.c_str());
  const auto &model_relation = pne_model->GetModelRelation();
  GE_CHECK_NOTNULL(model_relation);
  GE_CHK_STATUS_RET_NOLOG(CheckConsistency(parent_model_queue_info, model_relation->root_model_queue_info));
  auto name_refs = BuildNameRefs(parent_model_queue_info, model_relation->root_model_queue_info);
  // add inner queue defs
  MergeQueueDefs(name_refs, model_relation->queue_defs);
  flattened_model_relation_.invoked_model_queue_infos.insert(model_relation->invoked_model_queue_infos.begin(),
                                                             model_relation->invoked_model_queue_infos.end());
  // process submodels
  for (auto &it : model_relation->submodel_queue_infos) {
    auto &submodel_info_info = it.second;
    ReplaceQueueNames(name_refs, submodel_info_info.input_queue_names);
    ReplaceQueueNames(name_refs, submodel_info_info.output_queue_names);
    auto submodel = pne_model->GetSubmodel(submodel_info_info.model_name);
    GE_CHECK_NOTNULL(submodel, "Failed to get submodel, parent_model = %s, submodel_name = %s",
                     pne_model->GetModelName().c_str(), submodel_info_info.model_name.c_str());
    GE_CHK_STATUS_RET(FlattenSubmodel(submodel_info_info, submodel, depth + 1),
                      "Failed to flatten submodel %s", submodel_info_info.model_name.c_str());
  }
  return SUCCESS;
}

void ModelRelationFlattener::ReplaceQueueNames(const std::map<std::string, std::string> &name_refs,
                                               std::vector<std::string> &names) {
  for (auto &name : names) {
    auto it = name_refs.find(name);
    if (it != name_refs.cend()) {
      name = it->second;
    }
  }
}

void ModelRelationFlattener::MergeQueueDefs(const map<std::string, std::string> &name_refs,
                                            const vector<ModelRelation::QueueDef> &queue_defs) {
  for (const auto &queue_def : queue_defs) {
    auto it = name_refs.find(queue_def.name);
    if (it == name_refs.cend()) {  // inner queue defs
      flattened_model_relation_.queue_defs.emplace_back(queue_def);
    }
  }
}

std::map<std::string, std::string> ModelRelationFlattener::BuildNameRefs(
    const ModelRelation::ModelQueueInfo &parent_model_queue_info,
    const ModelRelation::ModelQueueInfo &root_model_queue_info) {
  std::map<std::string, std::string> name_refs;
  const auto &input_queue_names = root_model_queue_info.input_queue_names;
  const auto &output_queue_names = root_model_queue_info.output_queue_names;
  for (size_t i = 0; i < input_queue_names.size(); ++i) {
    name_refs[input_queue_names[i]] = parent_model_queue_info.input_queue_names[i];
  }
  for (size_t i = 0; i < output_queue_names.size(); ++i) {
    name_refs[output_queue_names[i]] = parent_model_queue_info.output_queue_names[i];
  }
  return name_refs;
}

Status ModelRelationFlattener::CheckConsistency(const ModelRelation::ModelQueueInfo &parent_model_queue_info,
                                                const ModelRelation::ModelQueueInfo &root_model_queue_info) {
  if (root_model_queue_info.input_queue_names.size() != parent_model_queue_info.input_queue_names.size()) {
    GELOGE(PARAM_INVALID, "input queue size(%zu) mismatches that of parent's (%zu)",
           root_model_queue_info.input_queue_names.size(),
           parent_model_queue_info.input_queue_names.size());
    return PARAM_INVALID;
  }

  if (root_model_queue_info.output_queue_names.size() != parent_model_queue_info.output_queue_names.size()) {
    GELOGE(PARAM_INVALID, "output queue size(%zu) mismatches that of parent's (%zu)",
           root_model_queue_info.output_queue_names.size(),
           parent_model_queue_info.output_queue_names.size());
    return PARAM_INVALID;
  }

  return SUCCESS;
}
}  // namespace ge
