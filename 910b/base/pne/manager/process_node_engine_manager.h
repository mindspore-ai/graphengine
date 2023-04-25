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

#ifndef BASE_COMMON_PROCESS_NODE_ENGINE_MANAGER_H_
#define BASE_COMMON_PROCESS_NODE_ENGINE_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

#include "common/plugin/plugin_manager.h"
#include "framework/pne/process_node_engine.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
using CreateFn = ProcessNodeEngine *(*)();

class ProcessNodeEngineManager {
 public:
  ProcessNodeEngineManager(const ProcessNodeEngineManager &other) = delete;
  ProcessNodeEngineManager &operator=(const ProcessNodeEngineManager &other) & = delete;
  static ProcessNodeEngineManager &GetInstance();
  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  Status RegisterEngine(const std::string &engine_id, const ProcessNodeEnginePtr &engine, CreateFn const fn);
  ProcessNodeEnginePtr GetEngine(const std::string &engine_id) const;
  ProcessNodeEnginePtr CloneEngine(const std::string &engine_id) const;
  std::string GetEngineName(const std::string &engine_id, const ge::NodePtr &node_ptr);
  inline const std::map<std::string, ProcessNodeEnginePtr> &GetEngines() const { return engines_map_; }
  bool IsEngineRegistered(const std::string &engine_id) const;

 private:
  ProcessNodeEngineManager() = default;
  ~ProcessNodeEngineManager() = default;

 private:
  PluginManager plugin_mgr_;
  std::map<std::string, ProcessNodeEnginePtr> engines_map_;
  std::map<std::string, CreateFn> engines_create_map_;
  std::atomic<bool> init_flag_{false};
  mutable std::mutex mutex_;
};

class ProcessNodeEngineRegisterar {
public:
  ProcessNodeEngineRegisterar(const std::string &engine_id, CreateFn const fn) noexcept;
  ~ProcessNodeEngineRegisterar() = default;
  ProcessNodeEngineRegisterar(const ProcessNodeEngineRegisterar &other) = delete;
  ProcessNodeEngineRegisterar &operator=(const ProcessNodeEngineRegisterar &other) & = delete;
};
}  // namespace ge

#define REGISTER_PROCESS_NODE_ENGINE(id, engine)                                                        \
  static ge::ProcessNodeEngineRegisterar g_##engine##_register __attribute__((unused))((id),            \
      []()->::ge::ProcessNodeEngine * { return new (std::nothrow) ge::engine(); })

#endif  // BASE_COMMON_PROCESS_NODE_ENGINE_MANAGER_H_