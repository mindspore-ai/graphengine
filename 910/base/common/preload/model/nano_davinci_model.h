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
#ifndef GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_
#define GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_
#include "common/preload/model/pre_davinci_model.h"

namespace ge {
class NanoDavinciModel : public PreDavinciModel {
 public:
  NanoDavinciModel() = default;
  virtual ~NanoDavinciModel() = default;
  Status Init() override;
  Status DoPartitionProcess() override;
  Status InitNodes(const ComputeGraphPtr &compute_graph) override;

 private:
  Status InitTaskId();
  Status MatchIndexToTaskIndex(const uint32_t label_idx, uint32_t &task_index) const;
  Status NanoAddSwitchKernel(const OpDescPtr &op_desc);
  Status GetTaskKernelOffset(const std::string &kernel_name, uint32_t &offset) const;
  Status NanoSetWeightData(OpDescPtr &op_desc) const;
  Status NanoAddSwitchConstNode(const std::vector<uint64_t> &cond_task_id_list, const ge::NodePtr &sw_node,
                                size_t &weight_offset, ComputeGraphPtr &graph) const;
  Status NanoSwitchWeightDataInit(ComputeGraphPtr &compute_graph, const ComputeGraph::Vistor<NodePtr> &all_nodes);
  Status InitSwitchWeightData(ComputeGraphPtr &compute_graph);
  Status InitSwitchNodes(const ComputeGraphPtr &compute_graph);
  std::map<uint32_t, int32_t> task_list_;
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_