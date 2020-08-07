/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef GE_GRAPH_PASSES_MULTI_BATCH_PASS_H_
#define GE_GRAPH_PASSES_MULTI_BATCH_PASS_H_

#include <string>
#include <vector>

#include "inc/graph_pass.h"

namespace ge {
class MultiBatchPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  Status FindPredValue(const ComputeGraphPtr &graph, OutDataAnchorPtr &pred_value);
  Status GetDynamicType();
  bool CheckSwitchN(std::vector<std::vector<int64_t>> &batch_shape, std::vector<std::vector<int64_t>> &combined_batch);
  bool GetBatchInfo(uint32_t batch_num, std::vector<std::vector<int64_t>> &batch_shape,
                    std::vector<std::vector<int64_t>> &combined_batch);
  void FindSwitchOutNodes(uint32_t batch_num);
  Status ReplaceSwitchN(const ComputeGraphPtr &graph, const OutDataAnchorPtr &pred_value,
                        const std::vector<std::vector<int64_t>> &batch_shape,
                        const std::vector<std::vector<int64_t>> &combined_batch);

  bool CheckDims(const std::vector<std::vector<int64_t>> &output_shape) const;
  NodePtr CreateSwitchCaseNode(const ComputeGraphPtr &graph, const std::string &name,
                               const OutDataAnchorPtr &pred_value, const std::vector<std::vector<int64_t>> &batch_shape,
                               const std::vector<std::vector<int64_t>> &combined_batch);
  Status BypassSwitchN(const NodePtr &switch_n_node, const NodePtr &switch_case_node);
  Status AttachLabel(const NodePtr &switch_case_node);
  Status AttachBatchLabel(uint32_t batch_idx);
  Status AttachStreamLabel(uint32_t batch_idx, const std::string &stream_label);

  std::vector<NodePtr> switch_n_nodes_;
  std::vector<NodePtr> bypass_nodes_;
  std::vector<std::vector<NodePtr>> batch_head_nodes_;
  int32_t dynamic_type_ = 0;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MULTI_BATCH_PASS_H_
