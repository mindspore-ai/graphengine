/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef GRAPH_ENGINE_LLT_STUB_ENGINE_H_
#define GRAPH_ENGINE_LLT_STUB_ENGINE_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include <memory>
#include <string>
#include "inc/st_types.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/optimizer/graph_optimizer.h"
#include "stub_engine/ops_kernel_store/stub_ops_kernel_store.h"

using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using StubOpsKernelInfoStorePtr = std::shared_ptr<ge::st::StubOpsKernelInfoStore>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;

namespace ge {
namespace st {
/**
 * host cpu engine.
 * Used for the ops which executes on host.
 */
class GE_FUNC_VISIBILITY StubEngine {
 public:
  /**
   * get StubEngine instance.
   * @return  StubEngine instance.
   */
  static StubEngine &Instance();

  virtual ~StubEngine() = default;

  /**
   * When Ge start, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  Status Initialize(const std::map<string, string> &options);

  /**
   * After the initialize, GE will invoke this interface
   * to get the Ops kernel Store.
   * @param ops_kernel_map The host cpu's ops kernel info
   */
  void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map);

  /**
   * After the initialize, GE will invoke this interface
   * to get the Graph Optimizer.
   * @param graph_optimizers The host cpu's Graph Optimizer objs
   */
  void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers);

  /**
   * When the graph finished, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  Status Finalize();

  StubEngine(const StubEngine &StubEngine) = delete;
  StubEngine(const StubEngine &&StubEngine) = delete;
  StubEngine &operator=(const StubEngine &StubEngine) = delete;
  StubEngine &operator=(StubEngine &&StubEngine) = delete;

 private:
  StubEngine() = default;
  map<string, OpsKernelInfoStorePtr> ops_kernel_store_map_;
};
}  // namespace st
}  // namespace ge

extern "C" {

/**
 * When Ge start, GE will invoke this interface
 * @return The status whether initialize successfully
 */
GE_FUNC_VISIBILITY ge::Status Initialize(const map<string, string> &options);

/**
 * After the initialize, GE will invoke this interface to get the Ops kernel Store
 * @param ops_kernel_map The host cpu's ops kernel info
 */
GE_FUNC_VISIBILITY void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map);

/**
 * After the initialize, GE will invoke this interface to get the Graph Optimizer
 * @param graph_optimizers The host cpu's Graph Optimizer objs
 */
GE_FUNC_VISIBILITY void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers);

/**
 * When the graph finished, GE will invoke this interface
 * @return The status whether initialize successfully
 */
GE_FUNC_VISIBILITY ge::Status Finalize();
}

#endif  // GRAPH_ENGINE_LLT_STUB_ENGINE_H_
