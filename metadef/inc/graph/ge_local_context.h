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
#ifndef INC_GRAPH_GE_LOCAL_CONTEXT_H_
#define INC_GRAPH_GE_LOCAL_CONTEXT_H_

#include <map>
#include <string>
#include <vector>
#include "graph/ge_error_codes.h"

using std::string;
using std::map;

namespace ge {
class GEThreadLocalContext {
 public:
  graphStatus GetOption(const string &key, string &option);
  void SetGraphOption(map<std::string, string> options_map);
  void SetSessionOption(map<std::string, string> options_map);
  void SetGlobalOption(map<std::string, string> options_map);

  map<string, string> GetAllGraphOptions() const;
  map<string, string> GetAllSessionOptions() const;
  map<string, string> GetAllGlobalOptions() const;
  map<string, string> GetAllOptions() const;

 private:
  map<string, string> graph_options_;
  map<string, string> session_options_;
  map<string, string> global_options_;
};  // class GEThreadLocalContext

GEThreadLocalContext &GetThreadLocalContext();
}  // namespace ge
#endif  // INC_GRAPH_GE_LOCAL_CONTEXT_H_
