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

#include "common/profiling/ge_profiling.h"
#include "runtime/base.h"
#include "common/profiling/profiling_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/load/graph_loader.h"
#include "init/gelib.h"
#include "framework/common/ge_inner_error_codes.h"

namespace {
const uint32_t kDeviceListIndex = 3;
const std::string kDeviceNums = "devNums";
const std::string kDeviceIdList = "devIdList";
const std::string kProfilingInit = "prof_init";
const std::string kProfilingFinalize = "prof_finalize";
const std::string kProfilingStart = "prof_start";
const std::string kProfilingStop = "prof_stop";
const std::string kProfModelSubscribe = "prof_model_subscribe";
const std::string kProfModelUnsubscribe = "prof_model_cancel_subscribe";
const std::string kRtSetDeviceRegName = "profiling";

const std::map<ProfCommandHandleType, std::string> kProfCommandTypeMap = {
    {kProfCommandhandleInit, kProfilingInit},
    {kProfCommandhandleStart, kProfilingStart},
    {kProfCommandhandleStop, kProfilingStop},
    {kProfCommandhandleFinalize, kProfilingFinalize},
    {kProfCommandhandleModelSubscribe, kProfModelSubscribe},
    {kProfCommandhandleModelUnsubscribe, kProfModelUnsubscribe}};
}  // namespace

bool TransProfConfigToParam(const ProfCommandHandleData &profCommand, vector<string> &prof_config_params) {
  prof_config_params.clear();
  prof_config_params.emplace_back(kDeviceNums);
  prof_config_params.emplace_back(std::to_string(profCommand.devNums));
  prof_config_params.emplace_back(kDeviceIdList);
  std::string devID = "";
  if (profCommand.devNums == 0) {
    GELOGW("The device num is invalid.");
    return false;
  }
  for (uint32_t i = 0; i < profCommand.devNums; i++) {
    devID.append(std::to_string(profCommand.devIdList[i]));
    if (i != profCommand.devNums - 1) {
      devID.append(",");
    }
  }

  prof_config_params.push_back(devID);
  return true;
}

bool isProfConfigValid(const uint32_t *deviceid_list, uint32_t device_nums) {
  if (deviceid_list == nullptr) {
    GELOGE(ge::PARAM_INVALID, "deviceIdList is nullptr");
    return false;
  }
  if (device_nums == 0 || device_nums > MAX_DEV_NUM) {
    GELOGE(ge::PARAM_INVALID, "The device nums: %u is invalid.", device_nums);
    return false;
  }

  // real device num
  int32_t dev_count = 0;
  rtError_t rt_err = rtGetDeviceCount(&dev_count);
  if (rt_err != RT_ERROR_NONE) {
    GELOGE(ge::INTERNAL_ERROR, "Get the Device count fail.");
    return false;
  }

  if (device_nums > static_cast<uint32_t>(dev_count)) {
    GELOGE(ge::PARAM_INVALID, "Device num(%u) is not in range 1 ~ %d.", device_nums, dev_count);
    return false;
  }

  std::unordered_set<uint32_t> record;
  for (size_t i = 0; i < device_nums; ++i) {
    uint32_t dev_id = deviceid_list[i];
    if (dev_id >= static_cast<uint32_t>(dev_count)) {
      GELOGE(ge::PARAM_INVALID, "Device id %u is not in range 0 ~ %d(exclude %d)", dev_id, dev_count, dev_count);
      return false;
    }
    if (record.count(dev_id) > 0) {
      GELOGE(ge::PARAM_INVALID, "Device id %u is duplicatedly set", dev_id);
      return false;
    }
    record.insert(dev_id);
  }
  return true;
}

ge::Status RegProfCtrlCallback(MsprofCtrlCallback func) {
  if (func == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Msprof ctrl callback is nullptr.");
    return ge::PARAM_INVALID;
  }
  if (ge::ProfilingManager::Instance().GetMsprofCallback().msprofCtrlCallback != nullptr) {
    GELOGW("Msprof ctrl callback is exist, just ignore it.");
  } else {
    GELOGI("GE register Msprof ctrl callback.");
    ge::ProfilingManager::Instance().SetMsprofCtrlCallback(func);
  }
  return ge::SUCCESS;
}

ge::Status RegProfSetDeviceCallback(MsprofSetDeviceCallback func) {
  if (func == nullptr) {
    GELOGE(ge::PARAM_INVALID, "MsprofSetDeviceCallback callback is nullptr.");
    return ge::PARAM_INVALID;
  }
  // Pass MsprofSetDeviceCallback to runtime
  GELOGI("GE pass setdevice callback to runtime.");
  ge::Status rt_ret = rtRegDeviceStateCallback(kRtSetDeviceRegName.c_str(), static_cast<rtDeviceStateCallback>(func));
  if (rt_ret != ge::SUCCESS) {
    GELOGE(rt_ret, "Pass MsprofSetDeviceCallback to runtime failed!");
    return rt_ret;
  }
  return ge::SUCCESS;
}

ge::Status RegProfReporterCallback(MsprofReporterCallback func) {
  if (func == nullptr) {
    GELOGE(ge::PARAM_INVALID, "MsprofReporterCallback callback is nullptr.");
    return ge::PARAM_INVALID;
  }
  if (ge::ProfilingManager::Instance().GetMsprofCallback().msprofReporterCallback != nullptr) {
    GELOGW("Msprof reporter callback is exist, just ignore it.");
  } else {
    GELOGI("GE register Msprof reporter callback.");
    ge::ProfilingManager::Instance().SetMsprofReporterCallback(func);
    // Pass MsprofReporterCallback to runtime
    ge::Status rt_ret = rtSetMsprofReporterCallback(func);
    if (rt_ret != ge::SUCCESS) {
      GELOGE(rt_ret, "Pass MsprofReporterCallback to runtime failed!!");
      return rt_ret;
    }
    // Pass MsprofReporterCallback to hccl
  }
  return ge::SUCCESS;
}

ge::Status ProfCommandHandle(ProfCommandHandleType type, void *data, uint32_t len) {
  if (type != kProfCommandhandleFinalize) {
    GE_CHECK_NOTNULL(data);
  }
  ProfCommandHandleData *prof_config_param = (ProfCommandHandleData *)data;
  auto iter = kProfCommandTypeMap.find(type);
  if (iter == kProfCommandTypeMap.end()) {
    GELOGW("The prof comand type is invalid.");
    return ge::PARAM_INVALID;
  }
  std::vector<string> prof_params;
  if (type == kProfCommandhandleStart || type == kProfCommandhandleStop) {
    if (!isProfConfigValid(prof_config_param->devIdList, prof_config_param->devNums)) {
      return ge::FAILED;
    }
  
    if (!TransProfConfigToParam(*prof_config_param, prof_params)) {
      GELOGE(ge::PARAM_INVALID, "Transfer profilerConfig to string vector failed");
      return ge::PARAM_INVALID;
    }
  }
  ge::GraphLoader graph_loader;
  ge::Command command;
  command.cmd_params.clear();
  command.cmd_type = iter->second;
  command.cmd_params = prof_params;
  if (type != kProfCommandhandleFinalize) {
    command.module_index = prof_config_param->profSwitch;
  }
  GELOGI("GE commandhandle execute, Command Type: %d, data type config: 0x%llx", type, command.module_index);
  if (type == kProfCommandhandleStart || type == kProfCommandhandleStop) {
    GELOGI("Profiling device nums:%s , deviceID:[%s]", prof_params[0].c_str(), prof_params[kDeviceListIndex].c_str());
  }
  ge::Status ret = graph_loader.CommandHandle(command);
  if (ret != ge::SUCCESS) {
    GELOGE(ret, "Handle profiling command failed");
    return ge::FAILED;
  }

  GELOGI("Successfully execute profiling command type: %d, command 0x%llx.", type, command.module_index);
  return ge::SUCCESS;
}

