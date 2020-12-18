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

#include "./ge_local_context.h"
#include <utility>

namespace ge {
namespace {
thread_local GEThreadLocalContext thread_context;
}

GEThreadLocalContext &GetThreadLocalContext() { return thread_context; }

graphStatus GEThreadLocalContext::GetOption(const string &key, string &option) {
  auto graph_iter = graph_options_.find(key);
  if (graph_iter != graph_options_.end()) {
    option = graph_iter->second;
    return GRAPH_SUCCESS;
  }
  auto session_iter = session_options_.find(key);
  if (session_iter != session_options_.end()) {
    option = session_iter->second;
    return GRAPH_SUCCESS;
  }
  auto global_iter = global_options_.find(key);
  if (global_iter != global_options_.end()) {
    option = global_iter->second;
    return GRAPH_SUCCESS;
  }
  return GRAPH_PARAM_INVALID;
}

void GEThreadLocalContext::SetGlobalOption(map<string, string> options_map) {
  global_options_.clear();
  global_options_ = std::move(options_map);
}

void GEThreadLocalContext::SetSessionOption(map<string, string> options_map) {
  session_options_.clear();
  session_options_ = std::move(options_map);
}

void GEThreadLocalContext::SetGraphOption(map<std::string, string> options_map) {
  graph_options_.clear();
  graph_options_ = std::move(options_map);
}

map<string, string> GEThreadLocalContext::GetAllGraphOptions() const {
  return graph_options_;
}

map<string, string> GEThreadLocalContext::GetAllSessionOptions() const {
  return session_options_;
}

map<string, string> GEThreadLocalContext::GetAllGlobalOptions() const {
  return global_options_;
}

map<string, string> GEThreadLocalContext::GetAllOptions() const {
  map<string, string> options_all;
  options_all.insert(graph_options_.begin(), graph_options_.end());
  options_all.insert(session_options_.begin(), session_options_.end());
  options_all.insert(global_options_.begin(), global_options_.end());
  return options_all;
}
}  // namespace ge
