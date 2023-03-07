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
#include "common/plugin/runtime_plugin_loader.h"
#include "framework/common/util.h"
#include "common/plugin/ge_util.h"

namespace ge {
RuntimePluginLoader &RuntimePluginLoader::GetInstance() {
  static RuntimePluginLoader instance;
  return instance;
}

ge::graphStatus RuntimePluginLoader::Initialize(const std::string &path_base) {
  std::string lib_path = "plugin/engines/runtime";
  const std::string path = path_base + lib_path;

  plugin_manager_ = ge::MakeUnique<ge::PluginManager>();
  GE_CHECK_NOTNULL(plugin_manager_);
  GE_CHK_STATUS_RET(plugin_manager_->Load(path), "[RT_Plugin][Libs]Failed, lib_paths=%s.", path.c_str());
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge