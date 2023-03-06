/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "common/plugin/opp_so_manager.h"

#include <string>

#include "common/plugin/plugin_manager.h"
#include "graph/opsproto_manager.h"
#include "common/util/error_manager/error_manager.h"
#include "common/util/mem_utils.h"
#include "framework/common/debug/log.h"
#include "mmpa/mmpa_api.h"
#include "register/op_impl_space_registry.h"
#include "register/op_impl_registry_holder_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/operator_factory_impl.h"
#include "register/shape_inference.h"
#include "graph/utils/file_utils.h"

namespace ge {
namespace {
void CloseHandle(void * const handle) {
  if (handle != nullptr) {
    if (mmDlclose(handle) != 0) {
      const char *error = mmDlerror();
      error = (error == nullptr) ? "" : error;
      GELOGE(ge::FAILED, "[Close][Handle] failed, reason:%s", error);
    }
  }
}
}

void OppSoManager::LoadOppPackage() {
  LoadOpsProtoSo();
  LoadOpMasterSo();
}

void OppSoManager::LoadOpsProtoSo() const {
  std::string ops_proto_path;
  const Status ret = PluginManager::GetOpsProtoPath(ops_proto_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get opsproto path!");
    return;
  }

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> v_path;
  PluginManager::SplitPath(ops_proto_path, v_path);
  for (auto i = 0UL; i < v_path.size(); ++i) {
    std::string path = v_path[i] + "lib/" + os_type + "/" + cpu_type + "/";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    const INT32 result = mmRealPath(path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH);
    if (result != EN_OK) {
      GELOGW("[FindSo][Check] Get path with os&cpu type [%s] failed, reason:%s", path.c_str(), strerror(errno));
      path = v_path[i];
    }
    std::vector<std::string> so_list;
    PluginManager::GetFileListWithSuffix(path, "rt2.0.so", so_list);
    for (const auto &so_path : so_list) {
      if (SaveSo(so_path) != ge::GRAPH_SUCCESS) {
        GELOGW("Save so failed!");
      }
    }
  }
}

void OppSoManager::LoadOpMasterSo() const {
  std::string op_tiling_path;
  const Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get op tiling path!");
    return;
  }

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> path_vec;
  PluginManager::SplitPath(op_tiling_path, path_vec);
  for (const auto &path : path_vec) {
    std::string root_path = path + "op_master/lib/" + os_type + "/" + cpu_type + "/";
    char_t resolved_path[MMPA_MAX_PATH] = {};
    if (mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
      GELOGW("Get path with op_master path [%s] failed, reason:%s", root_path.c_str(), strerror(errno));
      root_path = path + "op_tiling/lib/" + os_type + "/" + cpu_type + "/";
      if (mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH) != EN_OK) {
        GELOGW("Get path with op_tiling path [%s] failed, reason:%s", root_path.c_str(), strerror(errno));
        continue;
      }
    }
    std::vector<std::string> so_list;
    PluginManager::GetFileListWithSuffix(root_path, "rt2.0.so", so_list);
    for (const auto &so_path : so_list) {
      if (SaveSo(so_path) != ge::GRAPH_SUCCESS) {
        GELOGW("Save so failed!");
      }
    }
  }
}

Status OppSoManager::SaveSo(const std::string &so_path) const {
  uint32_t len;
  const auto so_data = GetBinFromFile(const_cast<std::string &>(so_path), len);
  std::string str_so_data(so_data.get(), so_data.get()+len);
  if (gert::OpImplRegistryHolderManager::GetInstance().GetOpImplRegistryHolder(str_so_data) != nullptr) {
    GELOGI("So already loaded!");
    return ge::GRAPH_SUCCESS;
  }
  void * const handle = mmDlopen(so_path.c_str(),
                                 static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                                 static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
  if (handle == nullptr) {
    GELOGW("Failed to dlopen %s! errmsg:%s", so_path.c_str(), mmDlerror());
    return ge::GRAPH_FAILED;
  }
  const std::function<void()> callback = [&handle]() {
    CloseHandle(handle);
  };
  GE_DISMISSABLE_GUARD(close_handle, callback);
  const auto om_registry_holder = ge::MakeShared<gert::OpImplRegistryHolder>();
  GE_CHECK_NOTNULL(om_registry_holder);
  size_t impl_num = 0U;
  const auto impl_funcs = om_registry_holder->GetOpImplFunctionsByHandle(handle, so_path, impl_num);
  if (impl_funcs == nullptr) {
    GELOGW("Failed to get funcs from so!");
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0U; i < impl_num; i++) {
    om_registry_holder->GetTypesToImpl()[impl_funcs[i].op_type] = impl_funcs[i].funcs;
  }
  gert::OpImplRegistryHolderManager::GetInstance().AddRegistry(str_so_data, om_registry_holder);
  auto space_registry = gert::DefaultOpImplSpaceRegistry::GetInstance().GetDefaultSpaceRegistry();
  if (space_registry == nullptr) {
    space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
    GE_CHECK_NOTNULL(space_registry);
    gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
  }
  const auto ret = space_registry->AddRegistry(om_registry_holder);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGW("Space registry add new holder failed!");
    return ge::GRAPH_FAILED;
  }
  om_registry_holder->SetHandle(handle);
  GELOGI("Save so symbol and handle in path[%s] success!", so_path.c_str());
  GE_DISMISS_GUARD(close_handle);
  return ge::GRAPH_SUCCESS;
}

OppSoManager &OppSoManager::GetInstance() {
  static OppSoManager instance;
  return instance;
}
}  // namespace ge
