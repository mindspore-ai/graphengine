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
#include "common/model/model_relation.h"
#include "common/plugin/ge_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/pne/pne_model.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
constexpr int32_t kSubgraphIndex = 0;
constexpr uint32_t kDefaultQueue_Depth = 128U;
constexpr int32_t kDataOutputAnchorIndex = 0;
}  // namespace

Status ModelRelationBuilder::BuildFromRootGraph(const ComputeGraph &root_graph,
                                                std::unique_ptr<ModelRelation> &model_relation) {
  model_relation = MakeUnique<ModelRelation>();
  GE_CHECK_NOTNULL(model_relation);
  GE_CHK_STATUS_RET_NOLOG(DoBuild(root_graph));
  *model_relation = std::move(model_relation_);
  return SUCCESS;
}

Status ModelRelationBuilder::CreateQueueForDataNode(const Node &node, const std::string &prefix,
                                                    std::string &queue_name) {
  queue_name = prefix + ":" + node.GetName();
  GELOGD("queue name is %s.", queue_name.c_str());
  GE_CHK_STATUS_RET_NOLOG(
      CreateQueueDef(node.GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(kDataOutputAnchorIndex)), queue_name));
  int64_t data_index = -1;
  (void) AttrUtils::GetInt(node.GetOpDesc(), ATTR_NAME_INDEX, data_index);
  if ((data_index < 0) || (data_index >= INT32_MAX)) {
    GELOGE(PARAM_INVALID, "[%s] Data index out of range, data index = %ld",
           node.GetName().c_str(), data_index);
    return PARAM_INVALID;
  }
  if (static_cast<size_t>(data_index) >= model_relation_.root_model_queue_info.input_queue_names.size()) {
    model_relation_.root_model_queue_info.input_queue_names.resize(static_cast<uint64_t>(data_index + 1));
  }
  model_relation_.root_model_queue_info.input_queue_names[static_cast<uint64_t>(data_index)] = queue_name;
  GELOGD("Get data node[%s] as input %ld", node.GetName().c_str(), data_index);
  return SUCCESS;
}

Status ModelRelationBuilder::BuildForSingleModel(const ComputeGraph &root_graph, ModelRelation &model_relation) {
  std::vector<std::string> external_queue_names;
  for (const auto &node : root_graph.GetDirectNode()) {
    const auto &op_type = node->GetType();
    if (op_type == DATA) {
      std::string unused;
      GE_CHK_STATUS_RET(CreateQueueForDataNode(*node, root_graph.GetName(), unused),
                        "Failed to create queue for data: %s", node->GetName().c_str());
    } else if (op_type == QUEUE_DATA) {
      std::string queue_name;
      (void) AttrUtils::GetStr(node->GetOpDesc(), "queue_name", queue_name);
      if (queue_name.empty()) {
        GELOGE(PARAM_INVALID, "QueueData node [%s] missing attribute queue_name", node->GetName().c_str());
        return PARAM_INVALID;
      }
      GE_CHK_STATUS_RET_NOLOG(
          CreateQueueDef(node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(kDataOutputAnchorIndex)), queue_name));
      external_queue_names.emplace_back(queue_name);
    } else if (op_type == NETOUTPUT) {
      const auto num_outputs = node->GetOpDesc()->GetAllInputsSize();
      for (size_t i = 0U; i < num_outputs; ++i) {
        const std::string queue_name = root_graph.GetName() + ":output:" + std::to_string(i);
        GE_CHK_STATUS_RET_NOLOG(CreateQueueDef(node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(i)), queue_name));
        model_relation_.root_model_queue_info.output_queue_names.emplace_back(queue_name);
      }
    } else {
      // do nothing
    }
  }
  model_relation_.root_model_queue_info.external_input_queue_names = external_queue_names;
  model_relation_.root_model_queue_info.model_name = root_graph.GetName();
  model_relation_.submodel_queue_infos[root_graph.GetName()] = model_relation_.root_model_queue_info;
  model_relation_.submodel_queue_infos[root_graph.GetName()].external_input_queue_names = external_queue_names;
  model_relation = std::move(model_relation_);
  return SUCCESS;
}

Status ModelRelationBuilder::CheckDataNode(const NodePtr &node, bool &create_relation_flag) const {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  std::string pne_id = PNE_ID_NPU;
  (void) AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
  if (pne_id == PNE_ID_PS) {
    GELOGD("Data node: %s is ps_node, no need create relation.", node->GetName().c_str());
    create_relation_flag = false;
    return SUCCESS;
  }

  const auto &out_data_anchor = node->GetOutDataAnchor(kDataOutputAnchorIndex);
  GE_CHECK_NOTNULL(out_data_anchor);
  for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto &peer_node = in_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    if (peer_node->GetType() != PARTITIONEDCALL) {
      GELOGE(INTERNAL_ERROR, "Peer node of Data is not a PartitionedCall, type = %s", peer_node->GetType().c_str());
      return INTERNAL_ERROR;
    }
    GE_CHECK_NOTNULL(peer_node->GetOpDesc());
    (void) AttrUtils::GetStr(peer_node->GetOpDesc(), ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
    GELOGD("Data node: %s, engine of output peer node:%s is %s.",
           node->GetName().c_str(), peer_node->GetName().c_str(), pne_id.c_str());
    if (pne_id != PNE_ID_PS) {
      create_relation_flag = true;
      return SUCCESS;
    }
  }
  GELOGD("Data node: %s is ps node, no need create relation.", node->GetName().c_str());
  create_relation_flag = false;
  return SUCCESS;
}

Status ModelRelationBuilder::CheckNetOutputNode(const NodePtr &node, bool &create_relation_flag) const {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  std::string pne_id = PNE_ID_NPU;
  (void) AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
  if (pne_id == PNE_ID_PS) {
    GELOGD("NetOutput node: %s is ps_node, no need create relation.", node->GetName().c_str());
    create_relation_flag = false;
    return SUCCESS;
  }

  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(out_data_anchor);
    const auto &peer_node = out_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    if (peer_node->GetType() != PARTITIONEDCALL) {
      GELOGE(INTERNAL_ERROR, "Peer node of NetOutput is not a PartitionedCall, type = %s",
             peer_node->GetType().c_str());
      return INTERNAL_ERROR;
    }
    GE_CHECK_NOTNULL(peer_node->GetOpDesc());
    (void) AttrUtils::GetStr(peer_node->GetOpDesc(), ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
    GELOGD("NetOutput node: %s, engine of input peer node:%s is %s.",
           node->GetName().c_str(), peer_node->GetName().c_str(), pne_id.c_str());
    if (pne_id != PNE_ID_PS) {
      create_relation_flag = true;
      return SUCCESS;
    }
  }
  GELOGD("NetOutput node: %s is ps node, no need create relation.", node->GetName().c_str());
  create_relation_flag = false;
  return SUCCESS;
}

Status ModelRelationBuilder::DoBuildForData(const NodePtr &node,
                                            std::map<NodePtr, std::map<int, std::string>> &paired_inputs,
                                            const ComputeGraph &root_graph) {
  GE_CHECK_NOTNULL(node);
  bool create_relation_flag = true;
  GE_CHK_STATUS_RET_NOLOG(CheckDataNode(node, create_relation_flag));
  if (!create_relation_flag) {
    GELOGI("No need create relation for data node: %s.", node->GetName().c_str());
    return SUCCESS;
  }
  GELOGD("Begin to build relation for data node: %s.", node->GetName().c_str());
  std::string queue_name;
  GE_CHK_STATUS_RET(CreateQueueForDataNode(*node, root_graph.GetName(), queue_name),
                    "Failed to create queue for data: %s", node->GetName().c_str());
  const auto &out_data_anchor = node->GetOutDataAnchor(kDataOutputAnchorIndex);
  GE_CHECK_NOTNULL(out_data_anchor);
  for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto &peer_node = in_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    if (peer_node->GetType() != PARTITIONEDCALL) {
      GELOGE(INTERNAL_ERROR, "Peer node of Data is not a PartitionedCall, type = %s", peer_node->GetType().c_str());
      return INTERNAL_ERROR;
    }
    const auto &op_desc = peer_node->GetOpDesc();
    ModelRelation::ModelQueueInfo *dst_model_queues = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelQueueInfo(*op_desc, dst_model_queues));
    dst_model_queues->input_queue_names[static_cast<uint64_t>(in_data_anchor->GetIdx())] = queue_name;
    (void)paired_inputs[peer_node].emplace(in_data_anchor->GetIdx(), queue_name);
  }
  return SUCCESS;
}

Status ModelRelationBuilder::CreateEmptyModelRelation(const OpDesc &op_desc) {
  const auto &subgraph_names = op_desc.GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    GELOGE(PARAM_INVALID, "PartitionedCall [%s] does not have subgraph.", op_desc.GetName().c_str());
    return PARAM_INVALID;
  }
  const auto &model_name = subgraph_names[static_cast<uint64_t>(kSubgraphIndex)];
  const auto &it = model_relation_.submodel_queue_infos.find(model_name);
  if (it != model_relation_.submodel_queue_infos.end()) {
    return SUCCESS;
  }
  auto &ret = model_relation_.submodel_queue_infos[model_name];
  ret.model_name = model_name;
  GELOGD("Create empty model relation for ps model, model_name is: %s.", model_name.c_str());
  return SUCCESS;
}

Status ModelRelationBuilder::DoBuildForPartitionedCall(const NodePtr &node,
                                                       std::map<NodePtr, std::map<int32_t,
                                                       std::string>> &paired_inputs) {
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::string pne_id = PNE_ID_NPU;
  (void) AttrUtils::GetStr(op_desc, ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
  if (pne_id == PNE_ID_PS) {
    GELOGD("Create empty model relation for ps partitioned call node: %s.", node->GetName().c_str());
    return CreateEmptyModelRelation(*op_desc);
  }

  // check all input are valid
  std::vector<std::string> unused;
  GE_CHK_STATUS_RET_NOLOG(GetInputQueueNames(node, paired_inputs, unused));
  // create queue for submodel outputs, and set input to peer submodel
  ModelRelation::ModelQueueInfo *model_queues = nullptr;
  GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelQueueInfo(*node->GetOpDesc(), model_queues));
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    const int32_t output_idx = out_data_anchor->GetIdx();
    const std::string queue_name = node->GetName() + ":" + std::to_string(output_idx);
    GE_CHK_STATUS_RET_NOLOG(
        CreateQueueDef(node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(output_idx)), queue_name));
    model_queues->output_queue_names[static_cast<uint64_t>(output_idx)] = queue_name;
    for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(in_data_anchor);
      const auto &dequeue_node = in_data_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(dequeue_node);
      GE_CHECK_NOTNULL(dequeue_node->GetOpDesc());
      if (dequeue_node->GetType() == PARTITIONEDCALL) {
        ModelRelation::ModelQueueInfo *dst_model_queues = nullptr;
        GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelQueueInfo(*dequeue_node->GetOpDesc(),
                                                          dst_model_queues));
        dst_model_queues->input_queue_names[static_cast<uint64_t>(in_data_anchor->GetIdx())] = queue_name;
      }
      (void)paired_inputs[dequeue_node].emplace(in_data_anchor->GetIdx(), queue_name);
    }
  }
  return SUCCESS;
}

Status ModelRelationBuilder::DoBuildForNetOutput(const NodePtr &node,
                                                 const std::map<NodePtr, std::map<int32_t, std::string>>
                                                 &paired_inputs) {
  GE_CHECK_NOTNULL(node);
  bool create_relation_flag = true;
  GE_CHK_STATUS_RET_NOLOG(CheckNetOutputNode(node, create_relation_flag));
  if (!create_relation_flag) {
    GELOGI("No need create relation for netoutput node: %s.", node->GetName().c_str());
    return SUCCESS;
  }
  GE_CHK_STATUS_RET_NOLOG(GetInputQueueNames(node,
                                             paired_inputs,
                                             model_relation_.root_model_queue_info.output_queue_names));
  return SUCCESS;
}

Status ModelRelationBuilder::DoBuild(const ComputeGraph &root_graph) {
  model_relation_.root_model_queue_info.model_name = root_graph.GetName();
  std::map<NodePtr, std::map<int, std::string>> paired_inputs;
  for (const auto &node : root_graph.GetDirectNode()) {
    const auto &op_type = node->GetType();
    if (op_type == DATA) {
      GE_CHK_STATUS_RET_NOLOG(DoBuildForData(node, paired_inputs, root_graph));
    } else if (op_type == PARTITIONEDCALL) {
      GE_CHK_STATUS_RET_NOLOG(DoBuildForPartitionedCall(node, paired_inputs));
    } else if (op_type == NETOUTPUT) {
      GE_CHK_STATUS_RET_NOLOG(DoBuildForNetOutput(node, paired_inputs));
    } else {
      GELOGW("Unexpected node in root graph, name = %s, type = %s",
             node->GetName().c_str(),
             op_type.c_str());
    }
  }
  return SUCCESS;
}

Status ModelRelationBuilder::CreateQueueDef(const GeTensorDesc &tensor_desc, const string &queue_name) {
  const std::map<std::string, ModelRelation::QueueDef>::const_iterator &it = queue_defs_.find(queue_name);
  if (it != queue_defs_.end()) {
    GELOGE(PARAM_INVALID, "Duplicate queue name: %s", queue_name.c_str());
    return PARAM_INVALID;
  }

  ModelRelation::QueueDef queue_def{};
  queue_def.name = queue_name;
  queue_def.depth = kDefaultQueue_Depth;
  if (AttrUtils::HasAttr(tensor_desc, ATTR_NAME_FLOW_ATTR)) {
    if (AttrUtils::GetInt(tensor_desc, ATTR_NAME_FLOW_ATTR_DEPTH, queue_def.depth)) {
      GELOGD("[%s] Got queue depth = [%u] from flow attr", queue_name.c_str(), queue_def.depth);
    }
    if (AttrUtils::GetStr(tensor_desc, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, queue_def.enqueue_policy)) {
      GELOGD("[%s] Got enqueue_policy = [%s] from flow attr", queue_name.c_str(), queue_def.enqueue_policy.c_str());
    }
  }
  GE_CHK_BOOL_RET_STATUS(queue_defs_.emplace(queue_name, queue_def).second,
                         PARAM_INVALID,
                         "Duplicate queue name: %s",
                         queue_name.c_str());
  model_relation_.queue_defs.emplace_back(std::move(queue_def));
  return SUCCESS;
}

Status ModelRelationBuilder::GetOrCreateModelQueueInfo(const OpDesc &op_desc,
                                                       ModelRelation::ModelQueueInfo *&model_queue_info) {
  const auto &subgraph_names = op_desc.GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    GELOGE(PARAM_INVALID, "PartitionedCall [%s] does not have subgraph.", op_desc.GetName().c_str());
    return PARAM_INVALID;
  }

  const auto &model_name = subgraph_names[static_cast<uint64_t>(kSubgraphIndex)];
  const auto &it = model_relation_.submodel_queue_infos.find(model_name);
  if (it != model_relation_.submodel_queue_infos.end()) {
    model_queue_info = &it->second;
    return SUCCESS;
  }

  auto &ret = model_relation_.submodel_queue_infos[model_name];
  ret.model_name = model_name;
  ret.input_queue_names.resize(op_desc.GetInputsSize());
  ret.output_queue_names.resize(op_desc.GetOutputsSize());
  model_queue_info = &ret;
  return SUCCESS;
}

Status ModelRelationBuilder::GetInputQueueNames(const NodePtr &node,
                                                const map<NodePtr, std::map<int32_t, std::string>> &paired_inputs,
                                                std::vector<std::string> &input_queue_names) {
  GE_CHECK_NOTNULL(node);
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_LE(op_desc->GetInputsSize(), static_cast<uint64_t>(INT32_MAX));
  const auto input_size = static_cast<int32_t>(op_desc->GetInputsSize());
  if (input_size == 0) {
    GELOGD("Node [%s] does not have input.", op_desc->GetName().c_str());
    return SUCCESS;
  }

  const auto &it = paired_inputs.find(node);
  if (it == paired_inputs.end()) {
    REPORT_INNER_ERROR("E19999", "Node [%s] was not paired", op_desc->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "Node [%s] was not paired", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  for (int32_t i = 0; i < input_size; ++i) {
    const auto name_it = it->second.find(i);
    if (name_it == it->second.end()) {
      REPORT_INNER_ERROR("E19999", "Input[%d] of node [%s] was not paired", i, op_desc->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "Input[%d] of node [%s] was not paired", i, op_desc->GetName().c_str());
      return INTERNAL_ERROR;
    }

    input_queue_names.emplace_back(name_it->second);
  }
  return SUCCESS;
}

const ModelRelation::QueueDef *ModelRelationReader::GetQueueDef(const std::string &queue_name) const {
  const auto &it = queue_defs_.find(queue_name);
  if (it == queue_defs_.end()) {
    REPORT_INNER_ERROR("E19999", "queue name not found. name = %s", queue_name.c_str());
    GELOGE(PARAM_INVALID, "queue name not found. name = %s", queue_name.c_str());
    return nullptr;
  }
  return it->second;
}

Status ModelRelationReader::Initialize() {
  for (const auto &queue_def : model_relation_.queue_defs) {
    (void)queue_defs_.emplace(queue_def.name, &queue_def);
  }
  GE_CHK_STATUS_RET_NOLOG(BatchGetQueueDefs(model_relation_.root_model_queue_info.input_queue_names,
                                            input_queue_defs_));
  GE_CHK_STATUS_RET_NOLOG(BatchGetQueueDefs(model_relation_.root_model_queue_info.output_queue_names,
                                            output_queue_defs_));
  return SUCCESS;
}

Status ModelRelationReader::BatchGetQueueDefs(const vector<std::string> &queue_names,
                                              vector<const ModelRelation::QueueDef *> &queue_defs) const {
  for (const auto &queue_name : queue_names) {
    auto queue_def = GetQueueDef(queue_name);
    GE_CHECK_NOTNULL(queue_def);
    queue_defs.emplace_back(queue_def);
  }
  return SUCCESS;
}

const ModelRelation::InvokedModelQueueInfo *ModelRelationReader::GetInvokedModelQueueInfo(
    const std::string &invoke_key) const {
  const auto find_ret = model_relation_.invoked_model_queue_infos.find(invoke_key);
  if (find_ret == model_relation_.invoked_model_queue_infos.cend()) {
    GELOGE(PARAM_INVALID, "Failed to find invoke model queue, invoke key=%s", invoke_key.c_str());
    return nullptr;
  }
  return &(find_ret->second);
}

ModelRelationReader::ModelRelationReader(const ModelRelation &model_relation) : model_relation_(model_relation) {
}

const ModelRelation::ModelQueueInfo *ModelRelationReader::GetSubmodelQueueInfo(const string &model_name) const {
  const auto &it = model_relation_.submodel_queue_infos.find(model_name);
  if (it == model_relation_.submodel_queue_infos.end()) {
    REPORT_INNER_ERROR("E19999", "Failed to get submodel queue info, name = %s", model_name.c_str());
    GELOGE(PARAM_INVALID, "Failed to get submodel queue info, name = %s", model_name.c_str());
    return nullptr;
  }
  return &it->second;
}
}  // namespace ge
