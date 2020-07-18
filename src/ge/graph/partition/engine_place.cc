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

#include "graph/partition/engine_place.h"
#include <climits>
#include <memory>
#include <string>
#include <utility>
#include "common/op/ge_op_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"

namespace ge {
Status EnginePlacer::Run() {
  GELOGI("Engine placer starts.");
  if (compute_graph_ == nullptr) {
    GELOGE(GE_GRAPH_NULL_INPUT, "compute_graph_ is null.");
    return FAILED;
  }
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Run enginePlacer failed");
    return FAILED;
  }
  // Assign engine for each node in the graph
  instance_ptr->DNNEngineManagerObj().InitPerformanceStaistic();
  for (const auto &node_ptr : compute_graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node_ptr);
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    std::string engine_name;
    // Check if this node has assigned engine
    if ((!node_ptr->GetOpDesc()->GetOpKernelLibName().empty())) {
      engine_name = node_ptr->GetOpDesc()->GetOpEngineName();
    } else {
      // Call placer cost model to get the "best" engine for this node
      engine_name = instance_ptr->DNNEngineManagerObj().GetDNNEngineName(node_ptr->GetOpDesc());
      // If can't get op's engine name, return failed
      if (engine_name.empty()) {
        GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Can not find engine of op type %s",
               node_ptr->GetOpDesc()->GetType().c_str());
        return FAILED;
      }
    }
    if (AssignEngineAndLog(node_ptr, engine_name) != SUCCESS) {
      GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED, "[GraphPartitioner]: AssignEngineAndLog FAILED");
      return FAILED;
    }
  }
  for (auto &it : instance_ptr->DNNEngineManagerObj().GetCheckSupportCost()) {
    GEEVENT("The time cost of %s::CheckSupported is [%lu] micro second.", it.first.c_str(), it.second);
  }
  GELOGI("Engine placer ends.");
  return SUCCESS;
}

Status EnginePlacer::AssignEngineAndLog(ge::ConstNodePtr node_ptr, const std::string &engine_name) {
  if ((node_ptr == nullptr) || (node_ptr->GetOpDesc() == nullptr)) {
    GELOGE(FAILED, "node_ptr is null.");
    return FAILED;
  }

  // private function, promise node_ptr->GetOpDesc() not null
  GELOGD("Assigning DNNEngine %s to node %s, op type %s", engine_name.c_str(), node_ptr->GetName().c_str(),
         node_ptr->GetOpDesc()->GetType().c_str());

  // Record the node assigned engine name
  node_engine_map_.insert(std::make_pair(node_ptr, engine_name));

  return SUCCESS;
}
}  // namespace ge
