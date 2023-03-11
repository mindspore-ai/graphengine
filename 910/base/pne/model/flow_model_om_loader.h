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

#ifndef BASE_PNE_MODEL_FLOW_MODEL_OM_LOADER_H_
#define BASE_PNE_MODEL_FLOW_MODEL_OM_LOADER_H_

#include "pne/model/flow_model.h"
#include "framework/common/types.h"
#include "framework/common/helper/om_file_helper.h"

namespace ge {
class FlowModelOmLoader {
 public:
  static Status LoadToFlowModel(const ge::ModelData &model_data, FlowModelPtr &flow_model);

 private:
  static Status CheckModelPartitions(const std::vector<ModelPartition> &model_partitions);
  static ComputeGraphPtr LoadRootGraph(const ModelPartition &model_def_partition);
  static Status LoadFlowModelPartition(const ModelPartition &flow_model_partition, const FlowModelPtr &flow_model,
                                       std::vector<string> &submodel_names);
  static Status LoadFlowSubmodelPartition(const std::vector<ModelPartition> &model_partitions,
                                          std::map<std::string, PneModelPtr> &flow_submodels);
};
}  // namespace ge
#endif  // BASE_PNE_MODEL_FLOW_MODEL_OM_LOADER_H_
