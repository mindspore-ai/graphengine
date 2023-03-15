/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "pne/model/flow_model_om_saver.h"
#include "graph/model.h"
#include "graph/detail/model_serialize_imp.h"
#include "common/model/model_deploy_resource.h"
#include "proto/flow_model.pb.h"
#include "graph/utils/graph_utils.h"
namespace ge {
namespace {
void ConvertModelQueueInfo(const ModelRelation::ModelQueueInfo &model_queue_info,
                           flow_model::proto::ModelRelationDef_ModelQueueInfo &proto_model_queue_info) {
  proto_model_queue_info.set_model_name(model_queue_info.model_name);
  for (const auto &input_queue_name : model_queue_info.input_queue_names) {
    proto_model_queue_info.add_input_queue_name(input_queue_name);
  }
  for (const auto &output_queue_name : model_queue_info.output_queue_names) {
    proto_model_queue_info.add_output_queue_name(output_queue_name);
  }
  for (const auto &external_input_queue_name : model_queue_info.external_input_queue_names) {
    proto_model_queue_info.add_external_input_queue_name(external_input_queue_name);
  }
  for (const auto &external_output_queue_name : model_queue_info.external_output_queue_names) {
    proto_model_queue_info.add_external_output_queue_name(external_output_queue_name);
  }
  for (const auto &invoke_model_key : model_queue_info.invoke_model_keys) {
    proto_model_queue_info.add_invoke_model_key(invoke_model_key);
  }
}
void ConvertModelRealtion(const ModelRelation &model_relation,
                          flow_model::proto::ModelRelationDef &model_relation_def) {
  for (const auto &queue_def : model_relation.queue_defs) {
    auto *proto_queue_def = model_relation_def.add_queue_def();
    proto_queue_def->set_name(queue_def.name);
    proto_queue_def->set_depth(queue_def.depth);
    proto_queue_def->set_enqueue_policy(queue_def.enqueue_policy);
    proto_queue_def->set_is_control(queue_def.is_control_);
  }
  auto *proto_submodel_queue_info = model_relation_def.mutable_submodel_queue_info();
  for (const auto &submodel_queue_info : model_relation.submodel_queue_infos) {
    const auto &model_queue = submodel_queue_info.second;
    flow_model::proto::ModelRelationDef_ModelQueueInfo proto_model_queue_info;
    ConvertModelQueueInfo(model_queue, proto_model_queue_info);
    (*proto_submodel_queue_info)[submodel_queue_info.first] = proto_model_queue_info;
  }

  auto *proto_invoked_model_queue_info = model_relation_def.mutable_invoked_model_queue_info();
  for (const auto &invoked_model_queue_info : model_relation.invoked_model_queue_infos) {
    const auto &invoked_model_queue = invoked_model_queue_info.second;
    flow_model::proto::ModelRelationDef_InvokedModelQueueInfo proto_invoked_model_queue;
    for (const auto &input_queue_name : invoked_model_queue.input_queue_names) {
      proto_invoked_model_queue.add_input_queue_name(input_queue_name);
    }
    for (const auto &output_queue_name : invoked_model_queue.output_queue_names) {
      proto_invoked_model_queue.add_output_queue_name(output_queue_name);
    }
    (*proto_invoked_model_queue_info)[invoked_model_queue_info.first] = proto_invoked_model_queue;
  }

  auto *proto_root_model_queue_info = model_relation_def.mutable_root_model_queue_info();
  ConvertModelQueueInfo(model_relation.root_model_queue_info, *proto_root_model_queue_info);
}
}  // namespace

Status FlowModelOmSaver::SaveToOm(const std::string &output_file) {
  GE_CHK_STATUS_RET_NOLOG(AddModelDefPartition());
  GE_CHK_STATUS_RET_NOLOG(AddFlowModelPartition());
  GE_CHK_STATUS_RET_NOLOG(AddFlowSubModelPartitions());
  GE_CHK_STATUS_RET_NOLOG(UpdateModelHeader());
  GE_CHK_STATUS_RET_NOLOG(SaveFlowModelToFile(output_file));
  buffers_.clear();
  GELOGI("save to om success, output_file=%s.", output_file.c_str());
  return SUCCESS;
}

Status FlowModelOmSaver::AddModelDefPartition() {
  const auto &root_graph = flow_model_->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  FixNonStandardGraph(root_graph);
  ge::Model ge_model;
  ge_model.SetName(root_graph->GetName());
  ge_model.SetGraph(root_graph);
  ge::Buffer model_buffer;
  (void)ge_model.Save(model_buffer);
  GELOGD("MODEL_DEF size is %zu", model_buffer.GetSize());
  if (model_buffer.size() == 0UL) {
    GELOGE(FAILED, "save model def failed, as save size is 0.");
    return FAILED;
  }
  const auto ret = AddPartition(model_buffer, MODEL_DEF);
  GE_CHK_STATUS_RET(ret, "[Add][ModelDefPartition]Failed, partition size %zu", model_buffer.size());
  return SUCCESS;
}

Status FlowModelOmSaver::AddFlowModelPartition() {
  flow_model::proto::FlowModelDef flow_model_def;
  flow_model_def.set_model_name(flow_model_->GetModelName());
  // add model relation.
  const auto &model_relation = flow_model_->GetModelRelation();
  if (model_relation != nullptr) {
    auto *proto_relation = flow_model_def.mutable_relation();
    ConvertModelRealtion(*model_relation, *proto_relation);
  }

  const auto &submodels = flow_model_->GetSubmodels();
  for (const auto &submodel : submodels) {
    flow_model_def.add_submodel_name(submodel.second->GetModelName());
  }

  for (const auto &models_esched_priority : flow_model_->GetModelsEschedPriority()) {
    auto *const proto_models_esched_priority = flow_model_def.mutable_models_esched_priority();
    flow_model::proto::FlowModelDef_EschedPriority proto_esched_priority;
    auto *const proto_esched_priority_map = proto_esched_priority.mutable_esched_priority();
    for (const auto &esched_priority : models_esched_priority.second) {
      (*proto_esched_priority_map)[esched_priority.first] = esched_priority.second;
    }
    (*proto_models_esched_priority)[models_esched_priority.first] = proto_esched_priority;
  }

  for (const auto &model_name_to_rank_id : flow_model_->GetModelNameToRankId()) {
    auto *const proto_model_name_to_rank_id = flow_model_def.mutable_model_name_to_rank_id();
    (*proto_model_name_to_rank_id)[model_name_to_rank_id.first] = model_name_to_rank_id.second;
  }

  for (const auto &group_name_to_rank_ids : flow_model_->GetGroupNameToRankIds()) {
    auto *const proto_group_name_to_rank_ids = flow_model_def.mutable_group_name_to_rank_ids();
    flow_model::proto::FlowModelDef_RankIds rank_ids;
    for (const auto &rank_id : group_name_to_rank_ids.second) {
      rank_ids.add_rank_id(rank_id);
    }
    (*proto_group_name_to_rank_ids)[group_name_to_rank_ids.first] = rank_ids;
  }

  for (const auto &device_to_rank_ids : flow_model_->GetDeviceToRankIds()) {
    auto *const proto_device_to_rank_ids = flow_model_def.mutable_device_to_rank_ids();
    flow_model::proto::FlowModelDef_RankIds rank_ids;
    for (const auto &rank_id : device_to_rank_ids.second) {
      rank_ids.add_rank_id(rank_id);
    }
    (*proto_device_to_rank_ids)[device_to_rank_ids.first] = rank_ids;
  }

  GE_CHK_STATUS_RET(AddPartition(flow_model_def, FLOW_MODEL), "[Add][FlowModelDef]Failed, model=%s",
                    flow_model_->GetModelName().c_str());
  return SUCCESS;
}

Status FlowModelOmSaver::AddFlowSubModelPartitions() {
  const auto &submodels = flow_model_->GetSubmodels();
  for (const auto &submodel_iter : submodels) {
    const auto &submodel = submodel_iter.second;
    if (!submodel->GetSubmodels().empty()) {
      GELOGE(FAILED, "flow model is not flatten, sub model[%s] has [%zu] submodels", submodel->GetModelName().c_str(),
             submodel->GetSubmodels().size());
      return FAILED;
    }
    flow_model::proto::SubmodelDef submodel_def;
    submodel_def.set_model_name(submodel->GetModelName());
    submodel_def.set_model_type(submodel->GetModelType());
    ModelBufferData serialize_buff{};
    const auto ret = submodel->SerializeModel(serialize_buff);
    GE_CHK_STATUS_RET(ret, "[Serialize][Model]Failed, model name=%s, model_type=%s", submodel->GetModelName().c_str(),
                      submodel->GetModelType().c_str());
    submodel_def.set_om_data(serialize_buff.data.get(), serialize_buff.length);
    if (submodel->GetModelType() == PNE_ID_UDF) {
      auto *const udf_graph = submodel_def.mutable_graph();
      const ModelSerializeImp serialize_imp;
      if (!serialize_imp.SerializeGraph(submodel->GetRootGraph(), udf_graph, false)) {
        GELOGE(FAILED, "serialize udf graph failed, model name=%s", submodel->GetModelName().c_str());
        return FAILED;
      }
      const auto logic_device_id = submodel->GetLogicDeviceId();
      if (!logic_device_id.empty()) {
        auto *deploy_info = submodel_def.mutable_deploy_info();
        deploy_info->set_logic_device_id(logic_device_id);
      }
      const auto deploy_reource = submodel->GetDeployResource();
      if (deploy_reource != nullptr) {
        auto *deploy_resource_proto = submodel_def.mutable_deploy_resource();
        deploy_resource_proto->set_resource_type(deploy_reource->resource_type);
        // deploy resource other field is not support now
      }
    }
    GE_CHK_STATUS_RET(AddPartition(submodel_def, FLOW_SUBMODEL),
                      "[Add][FlowSubModelPartition]Failed, model=%s, model_type=%s", submodel->GetModelName().c_str(),
                      submodel->GetModelType().c_str());
    GELOGD("add flow submodel partition end, model=%s, model_type=%s", submodel->GetModelName().c_str(),
           submodel->GetModelType().c_str());
  }
  return SUCCESS;
}

Status FlowModelOmSaver::UpdateModelHeader() {
  // Save target/version to model_header
  ModelFileHeader &model_header = om_file_save_helper_.GetModelFileHeader();
  // just 1 model.
  model_header.model_num = 1U;
  model_header.modeltype = MODEL_TYPE_FLOW_MODEL;

  const auto &model_name = flow_model_->GetModelName();
  size_t name_len = model_name.length();
  name_len = (name_len > (MODEL_NAME_LENGTH - 1U)) ? (MODEL_NAME_LENGTH - 1U) : name_len;
  const auto err = memcpy_s(model_header.name, MODEL_NAME_LENGTH, model_name.c_str(), name_len);
  if (err != EOK) {
    GELOGW("[Save][Model]Failed copy model name for model %s, err %d", model_name.c_str(), err);
  }
  GELOGD("Model name save:%s", model_name.c_str());
  return SUCCESS;
}

Status FlowModelOmSaver::SaveFlowModelToFile(const std::string &output_file) {
  ModelBufferData model{};
  const auto ret = om_file_save_helper_.SaveModel(output_file.c_str(), model, true);
  GE_CHK_STATUS_RET(ret, "save model to file failed, filename:%s.", output_file.c_str());
  return SUCCESS;
}

Status FlowModelOmSaver::AddPartition(const google::protobuf::Message &partition_msg,
                                      ModelPartitionType partition_type) {
  Buffer buffer(partition_msg.ByteSizeLong());
  if (buffer.GetData() == nullptr) {
    GELOGE(FAILED, "alloc buffer failed, size=%zu, partition_type=%d.", partition_msg.ByteSizeLong(), partition_type);
    return FAILED;
  }
  if (!partition_msg.SerializePartialToArray(buffer.GetData(), static_cast<int32_t>(buffer.GetSize()))) {
    GELOGE(FAILED, "SerializePartialToArray failed, size=%zu, partition_type=%d", partition_msg.ByteSizeLong(),
           partition_type);
    return FAILED;
  }
  return AddPartition(buffer, partition_type);
}

Status FlowModelOmSaver::AddPartition(Buffer &buffer, ModelPartitionType partition_type) {
  ModelPartition partition;
  partition.data = buffer.data();
  partition.size = static_cast<uint64_t>(buffer.size());
  partition.type = partition_type;
  const auto ret = om_file_save_helper_.AddPartition(partition, 0UL);
  GE_CHK_STATUS_RET(ret, "[Add][Partition]Failed, partition size %zu, partition_type=%d", buffer.size(),
                    partition_type);
  buffers_.emplace_back(buffer);
  return SUCCESS;
}
void FlowModelOmSaver::FixNonStandardGraph(const ComputeGraphPtr &graph) {
  // remove invalid output nodes.
  const auto output_nodes = graph->GetOutputNodes();
  for (const auto &node : output_nodes) {
    if (node->GetOwnerComputeGraph() != graph) {
      (void)graph->RemoveOutputNode(node);
    }
  }

  const auto subgraphs = graph->GetAllSubgraphs();
  // flow model root graph is no need save subgraph
  for (const auto &subgraph : subgraphs) {
    (void)graph->RemoveSubGraph(subgraph);
  }
}
}  // namespace ge