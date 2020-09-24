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

#ifndef INC_EXTERNAL_GE_GE_PROF_H_
#define INC_EXTERNAL_GE_GE_PROF_H_

#include <map>
#include <string>
#include <vector>

#include "ge/ge_api_error_codes.h"

namespace ge {
enum ProfDataTypeConfig {
  kProfAcl = 0x0001,
  kProfTaskTime = 0x0002,
  kProfAiCoreMetrics = 0x0004,
  kProfAicpuTrace = 0x0008,
  kProfModelExecute = 0x0010,
  kProfRuntimeApi = 0x0020,
  kProfRuntimeTrace = 0x0040,
  kProfScheduleTimeline = 0x0080,
  kProfScheduleTrace = 0x0100,
  kProfAiVectorCoreMetrics = 0x0200,
  kProfSubtaskTime = 0x0400,
  kProfTrainingTrace = 0x0800,
  kProfHcclTrace = 0x1000,
  kProfDataProcess = 0x2000,
  kProfTaskTrace = 0x3842,
  kProfModelLoad = 0x8000000000000000
};

enum ProfilingAicoreMetrics {
  kAicoreArithmaticThroughput = 0,
  kAicorePipeline = 1,
  kAicoreSynchronization = 2,
  kAicoreMemory = 3,
  kAicoreInternalMemory = 4,
  kAicoreStall = 5,
  kAicoreMetricsAll = 255  // only for op_trace
};

typedef struct ProfAicoreEvents ProfAicoreEvents;
typedef struct aclgrphProfConfig aclgrphProfConfig;

Status aclgrphProfInit(const char *profiler_path, uint32_t length);
Status aclgrphProfFinalize();
aclgrphProfConfig *aclgrphProfCreateConfig(uint32_t *deviceid_list, uint32_t device_nums,
                                           ProfilingAicoreMetrics aicore_metrics, ProfAicoreEvents *aicore_events,
                                           uint64_t data_type_config);
Status aclgrphProfDestroyConfig(aclgrphProfConfig *profiler_config);
Status aclgrphProfStart(aclgrphProfConfig *profiler_config);
Status aclgrphProfStop(aclgrphProfConfig *profiler_config);
}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_PROF_H_
