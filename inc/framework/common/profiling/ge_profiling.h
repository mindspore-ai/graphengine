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

#ifndef INC_FRAMEWORK_COMMON_GE_PROFILING_H_
#define INC_FRAMEWORK_COMMON_GE_PROFILING_H_

#include "ge/ge_api_error_codes.h"
#include "toolchain/prof_callback.h"

const int MAX_DEV_NUM = 64;

enum ProfCommandHandleType {
  kProfCommandhandleInit = 0,
  kProfCommandhandleStart,
  kProfCommandhandleStop,
  kProfCommandhandleFinalize,
  kProfCommandhandleModelSubscribe,
  kProfCommandhandleModelUnsubscribe
};

struct ProfCommandHandleData {
  uint64_t profSwitch;
  uint32_t devNums;  // length of device id list
  uint32_t devIdList[MAX_DEV_NUM];
  uint32_t modelId;
};

GE_FUNC_VISIBILITY ge::Status RegProfCtrlCallback(MsprofCtrlCallback func);
GE_FUNC_VISIBILITY ge::Status RegProfSetDeviceCallback(MsprofSetDeviceCallback func);
GE_FUNC_VISIBILITY ge::Status RegProfReporterCallback(MsprofReporterCallback func);
GE_FUNC_VISIBILITY ge::Status ProfCommandHandle(ProfCommandHandleType type, void *data, uint32_t len);

#endif  // INC_FRAMEWORK_COMMON_GE_PROFILING_H_
