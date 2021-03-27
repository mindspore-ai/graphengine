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

#include "common/dump/dump_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"

namespace {
const char *const kDumpOFF = "OFF";
const char *const kDumpoff = "off";
const char *const kDumpOn = "on";
const uint64_t kInferSessionId = 0;
}  // namespace
namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY DumpManager &DumpManager::GetInstance() {
  static DumpManager instance;
  return instance;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status DumpManager::SetDumpConf(const DumpConfig &dump_config) {
  DumpProperties dump_properties;
  std::string dump_status;
  std::string dump_path;
  std::string dump_mode;
  std::string dump_op_switch;

  if (dump_config.dump_status.empty()) {
    dump_properties_map_.emplace(kInferSessionId, dump_properties);
    GELOGI("Dump does not open");
    return SUCCESS;
  }

  dump_status = dump_config.dump_status;
  GELOGI("Dump status is %s", dump_status.c_str());
  if (dump_config.dump_status == kDumpoff || dump_config.dump_status == kDumpOFF) {
    dump_properties.ClearDumpPropertyValue();
    dump_properties_map_.emplace(kInferSessionId, dump_properties);
    return SUCCESS;
  }
  dump_properties.SetDumpStatus(dump_status);

  dump_op_switch = dump_config.dump_op_switch;
  dump_properties.SetDumpOpSwitch(dump_op_switch);
  if (dump_op_switch == kDumpoff && dump_config.dump_list.empty()) {
    dump_properties_map_.emplace(kInferSessionId, dump_properties);
    GELOGE(PARAM_INVALID, "Dump list is invalid, dump_op_switch is %s", dump_op_switch.c_str())
    return PARAM_INVALID;
  }

  if (!dump_config.dump_list.empty()) {
    for (auto model_dump : dump_config.dump_list) {
      std::string model_name = model_dump.model_name;
      GELOGI("Dump model is %s", model_name.c_str());
      std::set<std::string> dump_layers;
      for (auto layer : model_dump.layers) {
        GELOGI("Dump layer is %s in model", layer.c_str());
        dump_layers.insert(layer);
      }
      dump_properties.AddPropertyValue(model_name, dump_layers);
    }
    if (dump_op_switch == kDumpOn) {
      GELOGI("Start to dump model and single op,dump op switch is %s", dump_op_switch.c_str());
    } else {
      GELOGI("Only dump model,dump op switch is %s", dump_op_switch.c_str());
    }
  } else {
    GELOGI("Only dump single op,dump op switch is %s", dump_op_switch.c_str());
  }

  dump_path = dump_config.dump_path;
  if (dump_path.empty()) {
    GELOGE(PARAM_INVALID, "Dump path is empty");
    return PARAM_INVALID;
  }

  if (dump_path[dump_path.size() - 1] != '/') {
    dump_path = dump_path + "/";
  }
  dump_path = dump_path + CurrentTimeInStr() + "/";
  GELOGI("Dump path is %s", dump_path.c_str());
  dump_properties.SetDumpPath(dump_path);

  dump_mode = dump_config.dump_mode;
  GELOGI("Dump mode is %s", dump_mode.c_str());
  dump_properties.SetDumpMode(dump_mode);
  dump_properties_map_.emplace(kInferSessionId, dump_properties);

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY const DumpProperties &DumpManager::GetDumpProperties(
  uint64_t session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    return iter->second;
  }
  static DumpProperties default_properties;
  return default_properties;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpManager::AddDumpProperties(
  uint64_t session_id, const DumpProperties &dump_properties) {
  std::lock_guard<std::mutex> lock(mutex_);
  dump_properties_map_.emplace(session_id, dump_properties);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void DumpManager::RemoveDumpProperties(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    dump_properties_map_.erase(iter);
  }
}

}  // namespace ge
