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

#ifndef INC_FRAMEWORK_PROCESS_NODE_ENGINE_H_
#define INC_FRAMEWORK_PROCESS_NODE_ENGINE_H_

#include <map>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/manager/graph_manager_utils.h"
#include "framework/pne/pne_model.h"

namespace ge {
class ProcessNodeEngineImpl {
 public:
  virtual ~ProcessNodeEngineImpl() = default;

  virtual Status OptimizeGraph(const std::vector<GeTensor> &inputs, ComputeGraphPtr &compute_graph) = 0;

  virtual Status BuildGraph(ComputeGraphPtr &compute_graph, PneModelPtr &model) = 0;
};

using ProcessNodeEngineImplPtr = std::shared_ptr<ProcessNodeEngineImpl>;

class ProcessNodeEngine {
 public:
  ProcessNodeEngine() = default;
  virtual ~ProcessNodeEngine() = default;
  ProcessNodeEngine(const ProcessNodeEngine &other) = delete;
  ProcessNodeEngine &operator=(const ProcessNodeEngine &other) = delete;

 public:
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  virtual Status Finalize() = 0;

  virtual Status OptimizeGraph(const std::vector<GeTensor> &inputs, ComputeGraphPtr &compute_graph) = 0;

  virtual Status BuildGraph(ComputeGraphPtr &compute_graph, PneModelPtr &model) = 0;

  virtual const std::string &GetEngineName(const ge::NodePtr &node_ptr = nullptr) const = 0;

  virtual void SetImpl(ProcessNodeEngineImplPtr impl) = 0;

  virtual Status AddGraph(const ComputeGraphPtr &compute_graph, const std::map<std::string, std::string> &options) {
    (void)compute_graph;
    (void)options;
    return SUCCESS;
  }

  virtual Status RemoveGraph(const uint32_t graph_id) {
    (void)graph_id;
    return SUCCESS;
  }

  virtual Status ParallelPartition(const ComputeGraphPtr &compute_graph) {
    (void)compute_graph;
    return NOT_CHANGED;
  }

 protected:
  std::string engine_id_;
  ProcessNodeEngineImplPtr impl_ = nullptr;
};

using ProcessNodeEnginePtr = std::shared_ptr<ProcessNodeEngine>;
}  // namespace ge

#endif  // INC_FRAMEWORK_PROCESS_NODE_ENGINE_H_
