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

#include "pne/model/flow_model.h"

namespace ge {
FlowModel::FlowModel(const ComputeGraphPtr &root_graph) : PneModel(root_graph) {};

Status FlowModel::SerializeModel(ModelBufferData &model_buff) {
  (void) model_buff;
  return SUCCESS;
}

Status FlowModel::UnSerializeModel(const ModelBufferData &model_buff) {
  (void) model_buff;
  return SUCCESS;
}

const std::map<std::string, std::vector<uint32_t>> &FlowModel::GetGroupNameToRankIds() const {
  return group_name_to_rank_ids_;
}

void FlowModel::SetGroupNameToRankIds(const std::map<std::string, std::vector<uint32_t>> &group_name_to_rank_ids) {
  group_name_to_rank_ids_ = group_name_to_rank_ids;
}

void FlowModel::SetModelNameToRankId(const std::map<std::string, uint32_t> &model_name_to_rank_id) {
  model_name_to_rank_id_ = model_name_to_rank_id;
}

const std::map<std::string, uint32_t> &FlowModel::GetModelNameToRankId() const {
  return model_name_to_rank_id_;
}

const std::map<std::string, std::vector<uint32_t>> &FlowModel::GetDeviceToRankIds() const {
  return device_to_rank_ids_;
}

void FlowModel::SetDeviceToRankIds(const map<std::string, std::vector<uint32_t>> &device_to_rank_ids) {
  device_to_rank_ids_ = device_to_rank_ids;
}
}  // namespace ge