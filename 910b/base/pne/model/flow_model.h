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
#ifndef GE_MODEL_FLOW_MODEL_H_
#define GE_MODEL_FLOW_MODEL_H_

#include <map>
#include "graph/compute_graph.h"
#include "framework/pne/pne_model.h"

namespace ge {
class FlowModel : public PneModel {
 public:
  FlowModel() = default;
  explicit FlowModel(const ComputeGraphPtr &root_graph);
  ~FlowModel() override = default;

  Status SerializeModel(ModelBufferData &model_buff) override;

  Status UnSerializeModel(const ModelBufferData &model_buff) override;

  void SetModelsEschedPriority(std::map<std::string, std::map<std::string, int32_t>> models_esched_priority) {
    models_esched_priority_ = std::move(models_esched_priority);
  }

  const std::map<std::string, std::map<std::string, int32_t>> &GetModelsEschedPriority() const {
    return models_esched_priority_;
  }

  const std::map<std::string, uint32_t> &GetModelNameToRankId() const;

  void SetModelNameToRankId(const std::map<std::string, uint32_t> &model_name_to_rank_id);

  const std::map<std::string, std::vector<uint32_t>> &GetGroupNameToRankIds() const;

  void SetGroupNameToRankIds(const std::map<std::string, std::vector<uint32_t>> &group_name_to_rank_ids);

  const std::map<std::string, std::vector<uint32_t>> &GetDeviceToRankIds() const;
  void SetDeviceToRankIds(const map<std::string, std::vector<uint32_t>> &device_to_rank_ids);

 private:
  std::map<std::string, std::map<std::string, int32_t>> models_esched_priority_;
  std::map<std::string, std::vector<uint32_t>> group_name_to_rank_ids_;
  std::map<std::string, uint32_t> model_name_to_rank_id_;
  std::map<std::string, std::vector<uint32_t>> device_to_rank_ids_;
};
using FlowModelPtr = std::shared_ptr<ge::FlowModel>;
}  // namespace ge
#endif  // GE_MODEL_FLOW_MODEL_H_
