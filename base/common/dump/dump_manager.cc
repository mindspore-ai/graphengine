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

#include "common/dump/dump_manager.h"
#include "common/ge_inner_attrs.h"

#include "common/global_variables/diagnose_switch.h"
#include "external/graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"

namespace ge {
namespace {
const char_t *const kDumpOFF = "OFF";
const char_t *const kDumpoff = "off";
const char_t *const kDumpOn = "on";
const char_t *const kDumpEnable = "1";
const char_t *const kDeviceDumpOn = "device_on";
const uint32_t kAllOverFlow = 3U;
}  // namespace

DumpManager &DumpManager::GetInstance() {
  static DumpManager instance;
  return instance;
}

bool DumpManager::NeedDoDump(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  if (dump_config.dump_status.empty() && dump_config.dump_debug.empty()) {
    infer_dump_properties_map_[kInferSessionId] = dump_properties;
    GELOGI("Dump does not open");
    return false;
  }
  GELOGI("Dump status is %s, dump debug is %s.", dump_config.dump_status.c_str(), dump_config.dump_debug.c_str());
  if (((dump_config.dump_status == kDumpoff) || (dump_config.dump_status == kDumpOFF)) &&
       (dump_config.dump_debug == kDumpoff)) {
    dump_properties.ClearDumpPropertyValue();
    infer_dump_properties_map_[kInferSessionId] = dump_properties;
    return false;
  }
  if ((dump_config.dump_status == kDumpOn) && (dump_config.dump_debug == kDumpOn)) {
    GELOGW("Not support coexistence of dump debug and dump status.");
    return false;
  }
  return true;
}

void DumpManager::SetDumpDebugConf(const DumpConfig &dump_config, DumpProperties &dump_properties) const {
  if (dump_config.dump_debug == kDumpOn) {
    GELOGI("Only do overflow detection, dump debug is %s.", dump_config.dump_debug.c_str());
    dump_properties.InitInferOpDebug();
    dump_properties.SetOpDebugMode(kAllOverFlow);
  }
}

void DumpManager::SetDumpList(const DumpConfig &dump_config, DumpProperties &dump_properties) const {
  for (const auto &model_dump : dump_config.dump_list) {
    const std::string model_name = model_dump.model_name;
    GELOGI("Dump model is %s", model_name.c_str());
    std::set<std::string> dump_layers;
    for (const auto &layer : model_dump.layers) {
      GELOGI("Dump layer is %s in model", layer.c_str());
      (void)dump_layers.insert(layer);
    }
    dump_properties.AddPropertyValue(model_name, dump_layers);
  }
}

Status DumpManager::SetNormalDumpConf(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  // device dumper only support dump all model
  std::string dump_status = dump_config.dump_status;
  if (dump_status == kDeviceDumpOn) {
    dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
    dump_status = kDumpOn;
  }
  if (dump_status == kDumpOn) {
    GELOGI("Only do normal dump process, dump status is %s", dump_config.dump_status.c_str());
    dump_properties.SetDumpStatus(dump_status);
    const std::string dump_op_switch = dump_config.dump_op_switch;
    dump_properties.SetDumpOpSwitch(dump_op_switch);
    if ((dump_op_switch == kDumpoff) && (dump_config.dump_list.empty())) {
      (void)infer_dump_properties_map_.emplace(kInferSessionId, dump_properties);
      GELOGE(PARAM_INVALID, "[Check][DumpList]Invalid, dump_op_switch is %s", dump_op_switch.c_str());
      REPORT_INNER_ERROR("E19999", "Dump list check invalid, dump_op_switch is %s", dump_op_switch.c_str());
      return PARAM_INVALID;
    }

    if (!dump_config.dump_list.empty()) {
      if (dump_op_switch == kDumpOn) {
        GELOGI("Start to dump model and single op, dump op switch is %s", dump_op_switch.c_str());
      } else {
        GELOGI("Only dump model, dump op switch is %s", dump_op_switch.c_str());
      }
      SetDumpList(dump_config, dump_properties);
    } else {
      GELOGI("Only dump single op, dump op switch is %s", dump_op_switch.c_str());
    }
    GELOGI("Dump mode is %s", dump_config.dump_mode.c_str());
    dump_properties.SetDumpMode(dump_config.dump_mode);
    GELOGI("Dump step is %s", dump_config.dump_step.c_str());
    dump_properties.SetDumpStep(dump_config.dump_step);
    diagnoseSwitch::EnableDataDump();
  }
  return SUCCESS;
}

Status DumpManager::SetDumpPath(const DumpConfig &dump_config, DumpProperties &dump_properties) const {
  std::string dump_path = dump_config.dump_path;
  if (dump_path.empty()) {
    GELOGE(PARAM_INVALID, "[Check][DumpPath]It is empty.");
    REPORT_INNER_ERROR("E19999", "Dump path check is empty.");
    return PARAM_INVALID;
  }
  if (dump_path[dump_path.size() - 1U] != '/') {
    dump_path = dump_path + "/";
  }
  dump_path = dump_path + CurrentTimeInStr() + "/";
  GELOGI("Dump path is %s", dump_path.c_str());
  dump_properties.SetDumpPath(dump_path);
  return SUCCESS;
}

Status DumpManager::SetDumpConf(const DumpConfig &dump_config) {
  DumpProperties dump_properties;
  if (!NeedDoDump(dump_config, dump_properties)) {
    diagnoseSwitch::DisableDumper();
    GELOGD("No need do dump process.");
    return SUCCESS;
  }
  SetDumpDebugConf(dump_config, dump_properties);
  GE_CHK_STATUS_RET(SetNormalDumpConf(dump_config, dump_properties), "[Init][DumpConf] failed when dump status is on.");
  GE_CHK_STATUS_RET(SetDumpPath(dump_config, dump_properties), "[Init][DumpPath] failed.");
  infer_dump_properties_map_[kInferSessionId] = dump_properties;
  return SUCCESS;
}

const DumpProperties &DumpManager::GetDumpProperties(const uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    return iter->second;
  }
  // offline infer set session id for different model, but dump properties just on kInferSessionId
  const auto infer_iter = infer_dump_properties_map_.find(kInferSessionId);
  if (infer_iter != infer_dump_properties_map_.end()) {
    return infer_iter->second;
  }
  static DumpProperties default_properties;
  return default_properties;
}

void DumpManager::AddDumpProperties(const uint64_t session_id, const DumpProperties &dump_properties) {
  const std::lock_guard<std::mutex> lock(mutex_);
  (void)dump_properties_map_.emplace(session_id, dump_properties);
}

void DumpManager::RemoveDumpProperties(const uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const std::map<uint64_t, DumpProperties>::const_iterator iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    (void)dump_properties_map_.erase(iter);
  }

  const auto infer_iter = infer_dump_properties_map_.find(session_id);
  if (infer_iter != infer_dump_properties_map_.end()) {
    (void)infer_dump_properties_map_.erase(infer_iter);
  }
}

bool DumpManager::GetCfgFromOption(const std::map<std::string, std::string> &options_all, DumpConfig &dump_cfg) {
  auto options = options_all;
  dump_cfg.dump_mode = options[OPTION_EXEC_DUMP_MODE];
  const std::string enable_flag = options[OPTION_EXEC_ENABLE_DUMP];
  if (enable_flag != kDumpEnable) {
    dump_cfg.dump_status = kDumpoff;
    dump_cfg.dump_debug = kDumpoff;
    return false;
  }
  // transfer from enable dump to dump status
  dump_cfg.dump_status = kDeviceDumpOn;
  dump_cfg.dump_debug = kDumpoff;
  dump_cfg.dump_step = options[OPTION_EXEC_DUMP_STEP];
  std::string host_pid_name = "unknown_pid";
  std::string executor_dev_id = "0";
  // pid
  if (options.find(kHostMasterPidName) != options.end()) {
    host_pid_name = options[kHostMasterPidName];
  }
  // device id
  if (options.find(kExecutorDevId) != options.end()) {
    executor_dev_id = options[kExecutorDevId];
  }
  dump_cfg.dump_path = "/var/log/npu/dump/pid" + host_pid_name + "/device" + executor_dev_id + "/";
  GELOGD("Get dump config: dump_mode[%s], dump_status[%s], dump_debug[%s], dump_path[%s], dump_step[%s]",
         dump_cfg.dump_mode.c_str(), dump_cfg.dump_status.c_str(), dump_cfg.dump_debug.c_str(),
         dump_cfg.dump_path.c_str(), dump_cfg.dump_step.c_str());
  return true;
}


}  // namespace ge
