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

#ifndef D_BASE_GRAPH_PARTITION_SUBGRAPH_BUILDER_FOR_OPTIMIZATION_H
#define D_BASE_GRAPH_PARTITION_SUBGRAPH_BUILDER_FOR_OPTIMIZATION_H

#include "graph/partition/base_subgraph_builder.h"
namespace ge {
class SubgraphBuilderForOptimization : public BaseSubgraphBuilder {
 public:
  SubgraphBuilderForOptimization() = default;
  ~SubgraphBuilderForOptimization() override = default;
  Status BuildSubgraph() override;
};
} // namespace ge
#endif  // D_BASE_GRAPH_PARTITION_SUBGRAPH_BUILDER_FOR_OPTIMIZATION_H
