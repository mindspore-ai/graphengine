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

#include "pne/manager/process_node_engine_manager.h"

#include <cstdio>
#include <map>

#include "framework/common/debug/log.h"
#include "common/plugin/ge_util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"

namespace ge {
ProcessNodeEngineManager &ProcessNodeEngineManager::GetInstance() {
  static ProcessNodeEngineManager instance;
  return instance;
}

Status ProcessNodeEngineManager::Initialize(const std::map<std::string, std::string> &options) {
  // Multiple initializations are not supported
  if (init_flag_.load()) {
    GELOGW("ProcessNodeEngineManager has been initialized.");
    return SUCCESS;
  }

  // Load process node engine so
  const std::string so_path = "plugin/pnecompiler/";
  std::string path = GetModelPath();
  (void)path.append(so_path);
  const std::vector<std::string> so_func;
  const Status status = plugin_mgr_.Load(path, so_func);
  if (status != SUCCESS) {
    GELOGE(status, "[Load][EngineSo]Failed, lib path %s", path.c_str());
    REPORT_CALL_ERROR("E19999", "Load engine so failed, lib path %s", path.c_str());
    return status;
  }

  GELOGI("The number of ProcessNodeEngine is %zu.", engines_map_.size());

  const std::lock_guard<std::mutex> lock(mutex_);
  // Engines initialize
  for (auto iter = engines_map_.cbegin(); iter != engines_map_.cend(); ++iter) {
    if (iter->second == nullptr) {
      GELOGI("Engine: %s point to nullptr", (iter->first).c_str());
      continue;
    }

    GELOGI("ProcessNodeEngine id:%s.", (iter->first).c_str());

    const Status init_status = iter->second->Initialize(options);
    if (init_status != SUCCESS) {
      GELOGE(init_status, "[Init][Engine]Failed, ProcessNodeEngine:%s", (iter->first).c_str());
      REPORT_CALL_ERROR("E19999", "Initialize ProcessNodeEngine:%s failed", (iter->first).c_str());
      return init_status;
    }
  }

  init_flag_.store(true);
  return SUCCESS;
}

Status ProcessNodeEngineManager::Finalize() {
  // Finalize is not allowed, initialize first is necessary
  if (!init_flag_.load()) {
    GELOGW("ProcessNodeEngineManager has been finalized.");
    return SUCCESS;
  }

  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto iter = engines_map_.cbegin(); iter != engines_map_.cend(); ++iter) {
    if (iter->second != nullptr) {
      GELOGI("ProcessNodeEngine id:%s.", (iter->first).c_str());
      const Status status = iter->second->Finalize();
      if (status != SUCCESS) {
        GELOGE(status, "[Finalize][Engine]Failed, ProcessNodeEngine:%s", (iter->first).c_str());
        REPORT_CALL_ERROR("E19999", "Finalize ProcessNodeEngine:%s failed", (iter->first).c_str());
        return status;
      }
    }
  }
  engines_map_.clear();
  engines_create_map_.clear();
  init_flag_.store(false);
  return SUCCESS;
}

ProcessNodeEnginePtr ProcessNodeEngineManager::GetEngine(const std::string &engine_id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = engines_map_.find(engine_id);
  if (iter != engines_map_.end()) {
    return iter->second;
  }

  GELOGW("Failed to get ProcessNodeEngine object by id:%s.", engine_id.c_str());
  return nullptr;
}

bool ProcessNodeEngineManager::IsEngineRegistered(const std::string &engine_id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = engines_map_.find(engine_id);
  if (iter != engines_map_.end()) {
    return true;
  }
  GELOGW("ProcessNodeEngine id:%s is not registered", engine_id.c_str());
  return false;
}

std::string ProcessNodeEngineManager::GetEngineName(const std::string &engine_id, const ge::NodePtr &node_ptr) {
  const std::lock_guard<std::mutex> lock(mutex_);
  GE_IF_BOOL_EXEC(node_ptr == nullptr,
                  GELOGE(GE_CLI_GE_NOT_INITIALIZED, "ProcessNodeEngineManager: node_ptr is nullptr");
                  return "");
  const auto it = engines_map_.find(engine_id);
  if (it != engines_map_.cend()) {
    return it->second->GetEngineName(node_ptr);
  } else {
    GELOGE(INTERNAL_ERROR, "ProcessNodeEngine id:%s is not registered.", engine_id.c_str());
  }
  return "";
}

Status ProcessNodeEngineManager::RegisterEngine(const std::string &engine_id,
                                                const ProcessNodeEnginePtr engine,
                                                const CreateFn fn) {
  const std::lock_guard<std::mutex> lock(mutex_);
  engines_map_[engine_id] = engine;
  engines_create_map_[engine_id] = fn;
  GELOGI("Register ProcessNodeEngine id:%s success.", engine_id.c_str());
  return SUCCESS;
}

ProcessNodeEnginePtr ProcessNodeEngineManager::CloneEngine(const std::string &engine_id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  ProcessNodeEnginePtr engine = nullptr;
  const auto it = engines_create_map_.find(engine_id);
  if (it != engines_create_map_.end()) {
    const auto fn = it->second;
    if (fn != nullptr) {
      engine.reset(fn());
    }
  }
  if (engine != nullptr) {
    GELOGI("Clone ProcessNodeEngine id:%s success.", engine_id.c_str());
  } else {
    GELOGE(INTERNAL_ERROR, "Clone ProcessNodeEngine id:%s failed.", engine_id.c_str());
  }
  return engine;
}

ProcessNodeEngineRegisterar::ProcessNodeEngineRegisterar(const std::string &engine_id, const CreateFn fn) noexcept {
  ProcessNodeEnginePtr engine = nullptr;
  if (fn != nullptr) {
    engine.reset(fn());
    if (engine == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Create][ProcessNodeEngine] id:%s", engine_id.c_str());
    } else {
      (void)ProcessNodeEngineManager::GetInstance().RegisterEngine(engine_id, engine, fn);
    }
  } else {
    GELOGE(INTERNAL_ERROR, "[Check][Param:fn]Creator is nullptr, ProcessNodeEngine id:%s", engine_id.c_str());
  }
}
}  // namespace ge
