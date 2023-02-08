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

#include "graph/partition/graph_partitioner_builder.h"
#include "common/plugin/ge_util.h"

namespace ge {
GraphPartitionerBuilder &GraphPartitionerBuilder::AppendMergePolicy(std::unique_ptr<BaseMergePolicy> &merge_policy) {
  merge_policies_.emplace_back(std::move(merge_policy));
  return *this;
}

GraphPartitionerBuilder &GraphPartitionerBuilder::SetSubgraphBuilderPolicy(
    std::unique_ptr<BaseSubgraphBuildPolicy> &build_policy) {
  build_policy_ = std::move(build_policy);
  return *this;
}

std::unique_ptr<GraphPartitioner> GraphPartitionerBuilder::Build(const std::string &partitioner_name) {
  // ownership transfer to graph_partitioner
  return MakeUnique<GraphPartitioner>(partitioner_name, root_graph_, build_policy_, merge_policies_);
}
} // namespace ge