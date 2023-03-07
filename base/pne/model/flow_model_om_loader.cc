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

#include "pne/model/flow_model_om_loader.h"
#include "framework/common/helper/model_helper.h"
#include "common/util/mem_utils.h"
#include "common/model/model_deploy_resource.h"
#include "common/model/model_relation.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/model.h"
#include "pne/model/serialized_model.h"
#include "proto/flow_model.pb.h"
namespace ge {
namespace {
constexpr size_t kFlowModelPartitionsModeDefIdx = 0UL;
constexpr size_t kFlowModelPartitionsFlowModelIdx = 1UL;
constexpr size_t kFlowModelPartitionsFlowSubModelStartIdx = 2UL;
void ConvertModelQueueInfo(const flow_model::proto::ModelRelationDef_ModelQueueInfo &proto_model_queue_info,
                           ModelRelation::ModelQueueInfo &model_queue_info) {
  model_queue_info.model_name = proto_model_queue_info.model_name();
  model_queue_info.input_queue_names.assign(proto_model_queue_info.input_queue_name().cbegin(),
                                            proto_model_queue_info.input_queue_name().cend());
  model_queue_info.output_queue_names.assign(proto_model_queue_info.output_queue_name().cbegin(),
                                             proto_model_queue_info.output_queue_name().cend());

  model_queue_info.external_input_queue_names.assign(proto_model_queue_info.external_input_queue_name().cbegin(),
                                                     proto_model_queue_info.external_input_queue_name().cend());

  model_queue_info.external_output_queue_names.assign(proto_model_queue_info.external_output_queue_name().cbegin(),
                                                      proto_model_queue_info.external_output_queue_name().cend());

  model_queue_info.invoke_model_keys.assign(proto_model_queue_info.invoke_model_key().cbegin(),
                                            proto_model_queue_info.invoke_model_key().cend());
}

void ConvertModelRealtion(const flow_model::proto::ModelRelationDef &model_relation_def,
                          ModelRelation &model_relation) {
  for (const auto &proto_queue_def : model_relation_def.queue_def()) {
    ModelRelation::QueueDef queue_def;
    queue_def.name = proto_queue_def.name();
    queue_def.depth = proto_queue_def.depth();
    queue_def.enqueue_policy = proto_queue_def.enqueue_policy();
    queue_def.is_control_ = proto_queue_def.is_control();
    model_relation.queue_defs.emplace_back(queue_def);
  }

  for (const auto &proto_submodel_queue_info : model_relation_def.submodel_queue_info()) {
    ModelRelation::ModelQueueInfo model_queue_info;
    ConvertModelQueueInfo(proto_submodel_queue_info.second, model_queue_info);
    model_relation.submodel_queue_infos[proto_submodel_queue_info.first] = model_queue_info;
  }

  for (const auto &proto_invoked_model_queue_iter : model_relation_def.invoked_model_queue_info()) {
    const auto &proto_invoked_model_queue = proto_invoked_model_queue_iter.second;
    ModelRelation::InvokedModelQueueInfo invoked_model_queue;
    invoked_model_queue.input_queue_names.assign(proto_invoked_model_queue.input_queue_name().cbegin(),
                                                 proto_invoked_model_queue.input_queue_name().cend());
    invoked_model_queue.output_queue_names.assign(proto_invoked_model_queue.output_queue_name().cbegin(),
                                                  proto_invoked_model_queue.output_queue_name().cend());

    model_relation.invoked_model_queue_infos[proto_invoked_model_queue_iter.first] = invoked_model_queue;
  }

  const auto &proto_root_model_queue_info = model_relation_def.root_model_queue_info();
  ConvertModelQueueInfo(proto_root_model_queue_info, model_relation.root_model_queue_info);
}

Status LoadGeRootModel(const flow_model::proto::SubmodelDef &flow_submodel_def, PneModelPtr &pne_model) {
  const auto &om_data = flow_submodel_def.om_data();
  ge::ModelData model;
  model.model_len = om_data.size();
  model.model_data = const_cast<char *>(om_data.c_str());
  ModelHelper model_helper;
  const auto ret = model_helper.LoadRootModel(model);
  GE_CHK_STATUS_RET(ret, "load ge root model failed, model_name=%s, model_type=%s.",
                    flow_submodel_def.model_name().c_str(), flow_submodel_def.model_type().c_str());
  pne_model = model_helper.GetGeRootModel();
  GELOGD("load ge root model success, model name=%s.", flow_submodel_def.model_name().c_str());
  return SUCCESS;
}

Status LoadSerializedModel(flow_model::proto::SubmodelDef &flow_submodel_def, PneModelPtr &pne_model) {
  ModelSerializeImp serialize_imp;
  ComputeGraphPtr graph;
  if (!serialize_imp.UnserializeGraph(graph, *(flow_submodel_def.mutable_graph()))) {
    GELOGE(FAILED, "UnserializeGraph failed, model_name=%s, model_type=%s.", flow_submodel_def.model_name().c_str(),
           flow_submodel_def.model_type().c_str());
    return FAILED;
  }
  const auto serialized_model = MakeShared<SerializedModel>(graph);
  GE_CHECK_NOTNULL(serialized_model, ", make SerializedModel failed");
  const auto &om_data = flow_submodel_def.om_data();
  if (!om_data.empty()) {
    std::shared_ptr<uint8_t> data_buf(new (std::nothrow) uint8_t[om_data.size()], std::default_delete<uint8_t[]>());
    GE_CHECK_NOTNULL(data_buf, ", make data buf failed, size=%zu", om_data.size());
    if (memcpy_s(data_buf.get(), om_data.size(), om_data.c_str(), om_data.size()) != EOK) {
      GELOGE(FAILED, "copy data failed, size=%zu.", om_data.size());
      return FAILED;
    }
    ModelBufferData model_buff;
    model_buff.data = data_buf;
    model_buff.length = om_data.size();
    GE_CHK_STATUS_RET(serialized_model->UnSerializeModel(model_buff), "UnSerializeModel failed, size=%zu",
                      om_data.size());
  }
  if (flow_submodel_def.has_deploy_resource()) {
    const auto &deploy_resource_proto = flow_submodel_def.deploy_resource();
    const auto deploy_resource = MakeShared<ModelDeployResource>();
    GE_CHECK_NOTNULL(deploy_resource);
    deploy_resource->resource_type = deploy_resource_proto.resource_type();
    // deploy resource other field is not support now.
    serialized_model->SetDeployResource(deploy_resource);
  }
  if (flow_submodel_def.has_deploy_info()) {
    const auto &deploy_info = flow_submodel_def.deploy_info();
    serialized_model->SetLogicDeviceId(deploy_info.logic_device_id());
  }
  pne_model = serialized_model;
  GELOGD("load serialized model success, model name=%s.", flow_submodel_def.model_name().c_str());
  return SUCCESS;
}
}  // namespace

Status FlowModelOmLoader::LoadToFlowModel(const ge::ModelData &model_data, FlowModelPtr &flow_model) {
  OmFileLoadHelper om_file_load_helper;
  auto ret = om_file_load_helper.Init(model_data);
  GE_CHK_STATUS_RET(ret, "om file load helper init failed.");
  const auto &model_partitions = om_file_load_helper.GetModelPartitions(0);
  ret = CheckModelPartitions(model_partitions);
  GE_CHK_STATUS_RET(ret, "check model partitions failed.");
  // load Root Graph
  const auto root_graph = LoadRootGraph(model_partitions[kFlowModelPartitionsModeDefIdx]);
  GE_CHECK_NOTNULL(root_graph, ", load root graph is null");
  const auto tmp_flow_model = MakeShared<FlowModel>(root_graph);
  GE_CHECK_NOTNULL(tmp_flow_model, ", load root graph is null");
  const auto model_relation = MakeShared<ModelRelation>();
  GE_CHECK_NOTNULL(model_relation, ", make shared for model relation failed, graph_name %s",
                   root_graph->GetName().c_str());
  std::string model_name;
  std::vector<string> submodel_names;
  ret = LoadFlowModelPartition(model_partitions[kFlowModelPartitionsFlowModelIdx], model_name, *model_relation,
                               submodel_names);
  GE_CHK_STATUS_RET(ret, "load flow model partition failed.");
  tmp_flow_model->SetModelName(model_name);
  std::map<std::string, PneModelPtr> flow_submodels;
  ret = LoadFlowSubmodelPartition(model_partitions, flow_submodels);
  GE_CHK_STATUS_RET(ret, "load flow submodel partition failed.");
  for (const auto &submodel_name : submodel_names) {
    const auto find_ret = flow_submodels.find(submodel_name);
    if (find_ret == flow_submodels.cend()) {
      GELOGE(FAILED, "flow model with submodel name=%s, but not found in submodel partition");
      return FAILED;
    }
    const auto &submodel = find_ret->second;
    ret = tmp_flow_model->AddSubModel(submodel, submodel->GetModelType());
    GE_CHK_STATUS_RET(ret, "add sub model failed, model_name=%s, model_type=%s.", submodel->GetModelName().c_str(),
                      submodel->GetModelType().c_str());
  }
  tmp_flow_model->SetModelRelation(model_relation);
  flow_model = tmp_flow_model;
  GELOGI("load to flow model success, model name=%s.", flow_model->GetModelName().c_str());
  return SUCCESS;
}

Status FlowModelOmLoader::CheckModelPartitions(const std::vector<ModelPartition> &model_partitions) {
  if (model_partitions.size() < kFlowModelPartitionsFlowSubModelStartIdx) {
    GELOGE(FAILED, "flow model partitions must has 2 partitions[MODEL_DEF, FLOW_MODEL], but size=%zu.",
           model_partitions.size());
    return FAILED;
  }
  // the 0th is model def
  const auto &model_def_partition = model_partitions[kFlowModelPartitionsModeDefIdx];
  if (model_def_partition.type != MODEL_DEF) {
    GELOGE(FAILED, "flow model [0]th partition type must be MODEL_DEF[%d], but %d.", MODEL_DEF,
           model_def_partition.type);
    return FAILED;
  }

  // the 1th partion is flow model.
  const auto &flow_model_partition = model_partitions[kFlowModelPartitionsFlowModelIdx];
  if (flow_model_partition.type != FLOW_MODEL) {
    GELOGE(FAILED, "flow model [1]th partition type must be FLOW_MODEL[%d], but %d.", FLOW_MODEL,
           flow_model_partition.type);
    return FAILED;
  }
  for (auto idx = kFlowModelPartitionsFlowSubModelStartIdx; idx < model_partitions.size(); ++idx) {
    const auto &flow_submodel_partition = model_partitions[idx];
    if (flow_submodel_partition.type != FLOW_SUBMODEL) {
      GELOGE(FAILED, "flow model [%zu]th partition type must be FLOW_SUBMODEL[%d], but %d.", FLOW_SUBMODEL,
             flow_submodel_partition.type);
      return FAILED;
    }
  }
  return SUCCESS;
}

ComputeGraphPtr FlowModelOmLoader::LoadRootGraph(const ModelPartition &model_def_partition) {
  Model model;
  const auto status = Model::Load(model_def_partition.data, model_def_partition.size, model);
  if (status != GRAPH_SUCCESS) {
    GELOGE(status, "load model def failed, size=%lu.", model_def_partition.size);
    return nullptr;
  }
  return model.GetGraph();
}

Status FlowModelOmLoader::LoadFlowModelPartition(const ModelPartition &flow_model_partition, std::string &model_name,
                                                 ModelRelation &model_relation, std::vector<string> &submodel_names) {
  flow_model::proto::FlowModelDef flow_model_def;
  if (!flow_model_def.ParseFromArray(flow_model_partition.data, static_cast<int32_t>(flow_model_partition.size))) {
    GELOGE(FAILED, "parse flow model partition def failed.");
    return FAILED;
  }
  model_name = flow_model_def.model_name();
  submodel_names.assign(flow_model_def.submodel_name().cbegin(), flow_model_def.submodel_name().cend());
  ConvertModelRealtion(flow_model_def.relation(), model_relation);
  return SUCCESS;
}

Status FlowModelOmLoader::LoadFlowSubmodelPartition(const std::vector<ModelPartition> &model_partitions,
                                                    std::map<std::string, PneModelPtr> &flow_submodels) {
  for (auto idx = kFlowModelPartitionsFlowSubModelStartIdx; idx < model_partitions.size(); ++idx) {
    const auto &flow_submodel_partition = model_partitions[idx];
    flow_model::proto::SubmodelDef flow_submodel_def;
    if (!flow_submodel_def.ParseFromArray(flow_submodel_partition.data,
                                          static_cast<int32_t>(flow_submodel_partition.size))) {
      GELOGE(FAILED, "parse flow submodel partition def failed, idx=%zu, size=%lu, type=%d.", idx,
             flow_submodel_partition.size, flow_submodel_partition.type);
      return FAILED;
    }
    PneModelPtr submodel = nullptr;
    if (flow_submodel_def.model_type() == PNE_ID_UDF) {
      GE_CHK_STATUS_RET(LoadSerializedModel(flow_submodel_def, submodel),
                        "LoadSerializedModel failed, model_name=%s, model_type=%s",
                        flow_submodel_def.model_name().c_str(), flow_submodel_def.model_type().c_str());
    } else {
      GE_CHK_STATUS_RET(LoadGeRootModel(flow_submodel_def, submodel),
                        "LoadGeRootModel failed, model_name=%s, model_type=%s", flow_submodel_def.model_name().c_str(),
                        flow_submodel_def.model_type().c_str());
    }
    GE_CHECK_NOTNULL(submodel, ", load flow submodel failed, model_name=%s, model_type=%s.",
                     flow_submodel_def.model_name().c_str(), flow_submodel_def.model_type().c_str());

    submodel->SetModelName(flow_submodel_def.model_name());
    submodel->SetModelType(flow_submodel_def.model_type());
    flow_submodels[flow_submodel_def.model_name()] = submodel;
    GELOGD("load flow submodel success, submodel name=%s.", flow_submodel_def.model_name().c_str());
  }
  return SUCCESS;
}
}  // namespace ge