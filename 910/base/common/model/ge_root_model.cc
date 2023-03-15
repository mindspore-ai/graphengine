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

#include "common/model/ge_root_model.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/helper/model_helper.h"
#include "common/checker.h"

namespace ge {
void GeRootModel::SetSubgraphInstanceNameToModel(const std::string &instance_name, const GeModelPtr &ge_model) {
  (void)subgraph_instance_name_to_model_.insert(std::pair<std::string, GeModelPtr>(instance_name, ge_model));
}

void GeRootModel::RemoveInstanceSubgraphModel(const std::string &instance_name) {
  (void)subgraph_instance_name_to_model_.erase(instance_name);
}

Status GeRootModel::CheckIsUnknownShape(bool &is_dynamic_shape) const {
  const ComputeGraphPtr &comp_graph = GetRootGraph();
  if (comp_graph == nullptr) {
    return FAILED;
  }
  is_dynamic_shape = false;
  (void)AttrUtils::GetBool(comp_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  is_dynamic_shape = (is_dynamic_shape || (comp_graph->GetGraphUnknownFlag()));
  return SUCCESS;
}

Status GeRootModel::SerializeModel(ModelBufferData &model_buff) {
  bool is_unknown_shape = false;
  (void) CheckIsUnknownShape(is_unknown_shape);
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);
  GE_CHK_STATUS_RET(model_helper.SaveToOmRootModel(shared_from_this(), "no-output.om", model_buff,
      is_unknown_shape), "[Serialize][Submodel] failed, model_name = [%s]", GetModelName().c_str());
  GELOGD("[Serialize][Submodel] succeeded, model_name = [%s], size = %lu", GetModelName().c_str(), model_buff.length);
  return SUCCESS;
}

Status GeRootModel::UnSerializeModel(const ModelBufferData &model_buff) {
  (void) model_buff;
  return SUCCESS;
}

std::string GeRootModel::GetLogicDeviceId() const {
  std::string logical_device_id;
  const auto &root_graph = GetRootGraph();
  if (root_graph != nullptr) {
    const auto &it = subgraph_instance_name_to_model_.find(root_graph->GetName());
    if (it != subgraph_instance_name_to_model_.cend()) {
      const auto &root_model = it->second;
      if (AttrUtils::GetStr(root_model, ATTR_NAME_DEPLOY_DEVICE_ID, logical_device_id)) {
        GELOGI("Model[%s] has logical device = %s", root_model->GetName().c_str(), logical_device_id.c_str());
      }
    }
  }
  return logical_device_id;
}

Status GeRootModel::SetLogicDeviceId(const string &logical_device_id) {
  const auto &root_graph = GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  const auto &it = subgraph_instance_name_to_model_.find(root_graph->GetName());
  GeModelPtr ge_model;
  if (it == subgraph_instance_name_to_model_.cend()) {
    GELOGD("submodel root model not found, root_graph name = %s", root_graph->GetName().c_str());
    ge_model = MakeShared<GeModel>();
    GE_CHECK_NOTNULL(ge_model);
    ge_model->SetGraph(root_graph);
  } else {
    GE_CHECK_NOTNULL(it->second);
    ge_model = it->second;
  }
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(*ge_model, ATTR_NAME_DEPLOY_DEVICE_ID, logical_device_id),
                         FAILED,
                         "Failed to set attribute: %s", ATTR_NAME_DEPLOY_DEVICE_ID.c_str());
  GELOGI("Model[%s] set logical device = %s", GetModelName().c_str(), logical_device_id.c_str());
  return SUCCESS;
}

const uint8_t *GeRootModel::GetOpSoStoreData() const { return op_so_store_.Data(); }

size_t GeRootModel::GetOpStoreDataSize() const { return op_so_store_.DataSize(); }

bool GeRootModel::GetSoInOmFlag() const { return so_in_om_; }

SoInOmInfo GeRootModel::GetSoInOmInfo() const { return so_info_; }

bool GeRootModel::LoadSoBinData(const uint8_t *const data, const size_t len) {
  return op_so_store_.Load(data, len);
}

std::vector<OpSoBinPtr> GeRootModel::GetAllSoBin() const {
  return op_so_store_.GetSoBin();
}

void GeRootModel::SetSoInOmInfo(const SoInOmInfo &so_info) {
  so_info_ = so_info;
  return;
}

bool GeRootModel::CheckAndSetNeedSoInOM() {
  const ComputeGraphPtr &comp_graph = GetRootGraph();
  GE_ASSERT_NOTNULL(comp_graph);
  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(comp_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  is_dynamic_shape = (is_dynamic_shape || (comp_graph->GetGraphUnknownFlag()));
  if (is_dynamic_shape) {
    so_in_om_ = true;
    return true;
  }

  bool stc_to_dyn_soft_sync = false;
  for (const auto &node : comp_graph->GetAllNodes()) {
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_static_to_dynamic_softsync_op", stc_to_dyn_soft_sync);
    if (stc_to_dyn_soft_sync) {
      break;
    }
  }

  so_in_om_ = stc_to_dyn_soft_sync;
  return so_in_om_;
}
}  // namespace ge