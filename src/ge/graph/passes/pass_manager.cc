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

#include "inc/pass_manager.h"
#include "common/debug/log.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/utils/node_utils.h"
#include "omg/omg_inner_types.h"

namespace ge {
const vector<GraphPass *> &PassManager::GraphPasses() const { return graph_passes_; }

Status PassManager::AddPass(GraphPass *pass) {
  GE_CHECK_NOTNULL(pass);
  graph_passes_.push_back(pass);
  return SUCCESS;
}

Status PassManager::Run(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  return Run(graph, graph_passes_);
}

Status PassManager::Run(const ComputeGraphPtr &graph, vector<GraphPass *> &passes) {
  GE_CHECK_NOTNULL(graph);
  bool not_changed = true;

  for (auto &pass : passes) {
    GE_CHECK_NOTNULL(pass);

    Status status = pass->Run(graph);
    if (status == SUCCESS) {
      not_changed = false;
    } else if (status != NOT_CHANGED) {
      GELOGE(status, "Pass Run failed");
      return status;
    }
  }

  return not_changed ? NOT_CHANGED : SUCCESS;
}

PassManager::~PassManager() {
  for (auto pass : graph_passes_) {
    GE_DELETE_NEW_SINGLE(pass);
  }
}
}  // namespace ge
