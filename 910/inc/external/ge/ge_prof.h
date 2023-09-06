/**
* @file ge_prof.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_GE_GE_PROF_H_
#define INC_EXTERNAL_GE_GE_PROF_H_

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

#include <map>
#include <string>
#include <vector>

#include "ge_common/ge_api_error_codes.h"

namespace ge {
enum ProfDataTypeConfig {
    kProfTaskTime       = 0x0002,
    kProfAiCoreMetrics  = 0x0004,
    kProfAicpu          = 0x0008,
    kProfL2cache        = 0x0010,
    kProfHccl           = 0x0020,
    kProfTrainingTrace  = 0x0040,
    kProfFwkScheduleL0  = 0x0200,
    kProfTaskTimeL0     = 0x0800,
    kProfFwkScheduleL1  = 0x01000000,
};

enum ProfilingAicoreMetrics {
    kAicoreArithmeticUtilization = 0,
    kAicorePipeUtilization = 1,
    kAicoreMemory = 2,
    kAicoreMemoryL0 = 3,
    kAicoreResourceConflictRatio = 4,
    kAicoreMemoryUB = 5,
    kAicoreL2Cache = 6,
    kAicorePipelineExecuteUtilization = 7
};

using ProfAicoreEvents = struct ProfAicoreEvents;
using aclgrphProfConfig = struct aclgrphProfConfig;

/**
 * @ingroup AscendCL
 * @brief Initialize the profiling and set profiling configuration path
 * @param [in] profiler_path: configuration path of profiling
 * @param [in] length: length of configuration path
 * @return Status result of function
 */
MSVP_PROF_API Status aclgrphProfInit(const char *profiler_path, uint32_t length);

/**
 * @ingroup AscendCL
 * @brief Finalize profiling
 * @return Status result of function
 */
MSVP_PROF_API Status aclgrphProfFinalize();

/**
 * @ingroup AscendCL
 * @brief Create data of type aclgrphProfConfig
 * @param [in] deviceid_list: device id list
 * @param [in] device_nums: device numbers
 * @param [in] aicore_metrics: type of aicore metrics
 * @param [in] aicore_events: pointer to aicore events be reserved, only support NULL now
 * @param [in] data_type_config: modules need profiling
 * @return Status result of function
 */
MSVP_PROF_API aclgrphProfConfig *aclgrphProfCreateConfig(uint32_t *deviceid_list, uint32_t device_nums,
    ProfilingAicoreMetrics aicore_metrics, ProfAicoreEvents *aicore_events, uint64_t data_type_config);

/**
 * @ingroup AscendCL
 * @brief  Destroy data of type aclgrphProfConfig
 * @param [in] profiler_config: config of profiling
 * @return Status result of function
 */
MSVP_PROF_API Status aclgrphProfDestroyConfig(aclgrphProfConfig *profiler_config);

/**
 * @ingroup AscendCL
 * @brief Start profiling of modules which is configured by profiler config
 * @param [in] profiler_config: config of profiling
 * @return Status result of function
 */
MSVP_PROF_API Status aclgrphProfStart(aclgrphProfConfig *profiler_config);

/**
 * @ingroup AscendCL
 * @brief Stop profiling of modules which is configured by profiler config
 * @param [in] profiler_config: config of profiling
 * @return Status result of function
 */
MSVP_PROF_API Status aclgrphProfStop(aclgrphProfConfig *profiler_config);

}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_PROF_H_
