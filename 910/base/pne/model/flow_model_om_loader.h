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

#include "framework/pne/flow_model.h"
#include "framework/common/types.h"
#include "framework/common/helper/om_file_helper.h"
#include "graph/manager/graph_var_manager.h"

namespace ge {
using NodeRefreshInfo = std::map<NodePtr, std::map<NodePtr, std::vector<std::pair<size_t, int64_t>>>>;

class FlowModelOmLoader {
 public:
  static Status LoadToFlowModel(const ge::ModelData &model_data, FlowModelPtr &flow_model);
  static Status LoadToFlowModelDesc(const ge::ModelData &model_data, FlowModelPtr &flow_model);
  static Status AssignConstantVarMem(FlowModelPtr &flow_model, const std::string &model_path, const uint64_t session_id,
                                     const uint32_t graph_id, const bool is_cache = false);

 private:
  static Status CheckModelPartitions(const std::vector<ModelPartition> &model_partitions);
  static ComputeGraphPtr LoadRootGraph(const ModelPartition &model_def_partition);
  static Status LoadFlowModelPartition(const ModelPartition &flow_model_partition, const FlowModelPtr &flow_model,
                                       std::vector<string> &submodel_names);
  static Status LoadFlowSubmodelPartition(const std::vector<ModelPartition> &model_partitions,
                                          std::map<std::string, PneModelPtr> &flow_submodels);
  static Status RecordOffsetsRefreshInfo(const ComputeGraphPtr &graph,
                                         const std::map<NodePtr, int64_t> &unrefreshed_offsets,
                                         NodeRefreshInfo &inputs_need_refresh, NodeRefreshInfo &outputs_need_refresh);
  static Status RefreshNodeOffset(NodeRefreshInfo &inputs_need_refresh,
                                  NodeRefreshInfo &outputs_need_refresh,
                                  std::map<int64_t, int64_t> &logical_addr_mapping);
  static Status UpdateModelTaskAddr(const PneModelPtr &pne_model,
                                    const std::map<int64_t, int64_t> &logical_addr_mapping);
};
}  // namespace ge
#endif  // BASE_PNE_MODEL_FLOW_MODEL_OM_LOADER_H_
