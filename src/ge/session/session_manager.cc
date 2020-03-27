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

#include "session/session_manager.h"

#include <memory>
#include <utility>

#include "framework/common/debug/ge_log.h"
#include "common/ge/ge_util.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/load/new_model_manager/model_manager.h"
#include "graph/ge_context.h"

using std::map;
using std::string;
using std::vector;

namespace ge {
Status SessionManager::Initialize(const std::map<std::string, std::string> &options) {
  if (init_flag_) {
    GELOGW("Session Manager has been initialized.");
    return SUCCESS;
  }
  init_flag_ = true;
  return SUCCESS;
}

Status SessionManager::Finalize() {
  if (!init_flag_) {
    GELOGW("Session Manager has not been initialized.");
    return SUCCESS;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto iter = session_manager_map_.begin(); iter != session_manager_map_.end(); ++iter) {
    (void)iter->second->Finalize();
  }
  session_manager_map_.clear();
  init_flag_ = false;
  return SUCCESS;
}

Status SessionManager::SetrtContext(rtContext_t rt_context) {
  GELOGI("set rt_context RT_CTX_NORMAL_MODE, device id:%u.", GetContext().DeviceId());
  GE_CHK_RT_RET(rtCtxCreate(&rt_context, RT_CTX_NORMAL_MODE, static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_RT_RET(rtCtxSetCurrent(rt_context));
  RtContextUtil::GetInstance().AddrtContext(rt_context);
  return SUCCESS;
}

Status SessionManager::CreateSession(const std::map<std::string, std::string> &options, SessionId &session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionId next_session_id = 0;

  std::lock_guard<std::mutex> lock(mutex_);
  Status next_session_id_ret = GetNextSessionId(next_session_id);
  if (next_session_id_ret != SUCCESS) {
    return next_session_id_ret;
  }

  SessionPtr session_ptr = MakeShared<InnerSession>(next_session_id, options);
  if (session_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }
  Status ret = session_ptr->Initialize();
  if (ret != SUCCESS) {
    return ret;
  }

  (void)session_manager_map_.emplace(std::pair<SessionId, SessionPtr>(next_session_id, session_ptr));
  session_id = next_session_id;

  // create a context
  ret = SetrtContext(rtContext_t());

  return ret;
}

Status SessionManager::DestroySession(SessionId session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
  if (it == session_manager_map_.end()) {
    return GE_SESSION_NOT_EXIST;
  }

  if (ModelManager::GetInstance() != nullptr) {
    ModelManager::GetInstance()->DestroyAicpuSession(session_id);
  }

  // Unified destruct rt_context
  RtContextUtil::GetInstance().DestroyrtContexts();

  SessionPtr inner_session = it->second;
  Status ret = inner_session->Finalize();
  if (ret != SUCCESS) {
    return ret;
  }
  (void)session_manager_map_.erase(session_id);
  return ret;
}

Status SessionManager::GetVariable(SessionId session_id, const std::string &name, Tensor &val) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr inner_session = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      inner_session = it->second;
    }
  }
  return inner_session->GetVariable(name, val);
}

Status SessionManager::AddGraph(SessionId session_id, uint32_t graph_id, const Graph &graph) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr inner_session = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      inner_session = it->second;
    }
    auto compute_graph = GraphUtils::GetComputeGraph(graph);
    std::string session_graph_id = std::to_string(session_id) + "_" + std::to_string(graph_id);
    if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      GELOGW("Set graph session_graph_id attr failed.");
    } else {
      GELOGD("Set graph session_graph_id attr to [%s]", session_graph_id.c_str());
    }
  }
  return inner_session->AddGraph(graph_id, graph);
}

Status SessionManager::RunGraph(SessionId session_id, uint32_t graph_id, const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &outputs) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr inner_session = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      inner_session = it->second;
    }
  }
  return inner_session->RunGraph(graph_id, inputs, outputs);
}

Status SessionManager::RemoveGraph(SessionId session_id, uint32_t graph_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr inner_session = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      inner_session = it->second;
    }
  }
  return inner_session->RemoveGraph(graph_id);
}

bool SessionManager::HasSession(SessionId session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return false;
  }
  return session_manager_map_.find(session_id) != session_manager_map_.end();
}

Status SessionManager::GetNextSessionId(SessionId &next_session_id) const {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  static SessionId session_id = 0;

  next_session_id = session_id++;
  return SUCCESS;
}

Status SessionManager::RegisterCallBackFunc(
  SessionId session_id, const std::string &key,
  const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr inner_session = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      inner_session = it->second;
    }
  }
  return inner_session->RegisterCallBackFunc(key, callback);
}

Status SessionManager::RunGraphAsync(SessionId session_id, uint32_t graph_id, const std::vector<TensorInfo> &inputs,
                                     std::vector<TensorInfo> &outputs, std::function<void(Status)> callback) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return GE_SESSION_MANAGER_NOT_INIT;
  }
  SessionPtr inner_session = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<SessionId, SessionPtr>::iterator it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    } else {
      inner_session = it->second;
    }
  }
  return inner_session->RunGraphAsync(graph_id, inputs, outputs, callback);
}
bool SessionManager::IsGraphNeedRebuild(SessionId session_id, uint32_t graph_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT);
    return true;
  }
  SessionPtr inner_session = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      GELOGE(GE_SESSION_NOT_EXIST, "The session %lu does not exists", session_id);
      return true;
    } else {
      inner_session = it->second;
    }
  }
  return inner_session->IsGraphNeedRebuild(graph_id);
}
};  // namespace ge
