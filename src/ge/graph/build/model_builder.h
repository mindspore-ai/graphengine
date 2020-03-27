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

#ifndef GE_GRAPH_BUILD_MODEL_BUILDER_H_
#define GE_GRAPH_BUILD_MODEL_BUILDER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/op/ge_op_utils.h"
#include "common/tbe_kernel_store.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/model.h"
#include "graph/node.h"
#include "model/ge_model.h"
#include "omg/omg_inner_types.h"

namespace ge {
class ModelBuilder {
 public:
  ModelBuilder(ge::ComputeGraphPtr whole_graph, const std::vector<SubGraphInfoPtr> &subgraphs,
               const std::map<std::string, int> &stream_max_parallel_num, bool hcom_parallel,
               int mode = static_cast<int>(domi::BuildMode::GEN_TASK_WITHOUT_FUSION));

  ModelBuilder(const ModelBuilder &) = delete;

  ModelBuilder &operator=(const ModelBuilder &op) = delete;

  ~ModelBuilder();

  Status SaveDataToModel(ge::Model &model, ge::GeModel &ge_model);
  Status PreBuildModel();
  Status BuildModelForGetTask(ge::Model &model_def);

  ge::Buffer GetWeightBuffer() const;

 protected:
  Status AssignMemory();

  void AddNodeInputProperty();

  void ClearOriginalFormat();

  Status MergeWeights();

 private:
  void SetInputIsConst(const ge::NodePtr &n);

  void SetModelVersion(ge::Model &model);

  Status CalcOutputSize(const ge::NodePtr &n);

  Status AdjustConstWeightSize(const ge::NodePtr &node, size_t &mem_offset);

  Status SetInputOutputDesc();

  Status AdjustInputTensorFlag();

  Status BuildModelDef(ge::Model &model_def);

  Status CompileSingleOp();

  size_t mem_offset_;

  size_t weight_offset_;

  ge::ComputeGraphPtr compute_graph_;

  const std::vector<SubGraphInfoPtr> &subgraphs_;

  int64_t stream_num_;

  int64_t event_num_;

  ge::Buffer weight_buffer_;

  std::map<std::string, int> stream_max_parallel_num_;
  bool hcom_parallel_;

  int build_mode_;
  size_t max_mem_offset_;

  TBEKernelStore tbe_kernel_store_;

  uint8_t platform_type_;
  bool is_loop_graph_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MODEL_BUILDER_H_
