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
  kProfTaskTime = 0x0002,
  kProfAiCoreMetrics = 0x0004,
  kProfAicpuTrace = 0x0008,
  kProfTrainingTrace = 0x0800,
  kProfHcclTrace = 0x1000
};

enum ProfilingAicoreMetrics {
  kAicoreArithmaticThroughput = 0,
  kAicorePipeline = 1,
  kAicoreSynchronization = 2,
  kAicoreMemory = 3,
  kAicoreInternalMemory = 4,
  kAicoreStall = 5
};

typedef struct ProfAicoreEvents ProfAicoreEvents;
typedef struct aclgrphProfConfig aclgrphProfConfig;

///
/// @ingroup AscendCL
/// @brief Initialize the profiling and set profiling configuration path
/// @param [in] profiler_path: configuration path of profiling
/// @param [in] length: length of configuration path
/// @return Status result of function
///
Status aclgrphProfInit(const char *profiler_path, uint32_t length);

///
/// @ingroup AscendCL
/// @brief Finalize profiling
/// @return Status result of function
///
Status aclgrphProfFinalize();

///
/// @ingroup AscendCL
/// @brief Create data of type aclgrphProfConfig
/// @param [in] deviceid_list: device id list
/// @param [in] device_nums: device numbers
/// @param [in] aicore_metrics: type of aicore metrics
/// @param [in] aicore_events: pointer to aicore events be reserved, only support NULL now
/// @param [in] data_type_config: modules need profiling
/// @return Status result of function
///
aclgrphProfConfig *aclgrphProfCreateConfig(uint32_t *deviceid_list, uint32_t device_nums,
                                           ProfilingAicoreMetrics aicore_metrics, ProfAicoreEvents *aicore_events,
                                           uint64_t data_type_config);

///
/// @ingroup AscendCL
/// @brief  Destroy data of type aclgrphProfConfig
/// @param [in] profiler_config: config of profiling
/// @return Status result of function
///
Status aclgrphProfDestroyConfig(aclgrphProfConfig *profiler_config);

///
/// @ingroup AscendCL
/// @brief Start profiling of modules which is configured by profiler config
/// @param [in] profiler_config: config of profiling
/// @return Status result of function
///
Status aclgrphProfStart(aclgrphProfConfig *profiler_config);

///
/// @ingroup AscendCL
/// @brief Stop profiling of modules which is configured by profiler config
/// @param [in] profiler_config: config of profiling
/// @return Status result of function
///
Status aclgrphProfStop(aclgrphProfConfig *profiler_config);
}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_PROF_H_
