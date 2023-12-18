/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef GE_COMMON_RUNTIME_PlUGIN_LOADER_H_
#define GE_COMMON_RUNTIME_PlUGIN_LOADER_H_

#include <memory>
#include "common/plugin/plugin_manager.h"
#include "external/ge/ge_api_error_codes.h"
#include "common/ge_visibility.h"
#include "common/ge_types.h"
#include "graph/compute_graph.h"

namespace ge {
class VISIBILITY_EXPORT RuntimePluginLoader {
 public:
  static RuntimePluginLoader &GetInstance();

  ge::graphStatus Initialize(const std::string &path_base);

 private:
  std::unique_ptr<ge::PluginManager> plugin_manager_;
};
}  // namespace ge
#endif  // GE_COMMON_RUNTIME_PlUGIN_LOADER_H_