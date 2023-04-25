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

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_H_
#include "graph/node.h"
#include "graph/bin_cache/node_compile_cache_module.h"
#include "external/ge/ge_api_types.h"
#include "graph/ge_local_context.h"

namespace ge {
class NodeBinSelector {
 public:
  /**
   * select a bin for node execution
   * @param node
   * @return
   */
  virtual NodeCompileCacheItem *SelectBin(const NodePtr &node, const GEThreadLocalContext *ge_context,
                                          std::vector<domi::TaskDef> &task_defs) = 0;
  virtual Status Initialize() = 0;
  NodeBinSelector() = default;
  virtual ~NodeBinSelector() = default;

 protected:
  NodeBinSelector(const NodeBinSelector&) = delete;
  NodeBinSelector &operator=(const NodeBinSelector &) & = delete;
};
}  // namespace ge

#endif // AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_H_
