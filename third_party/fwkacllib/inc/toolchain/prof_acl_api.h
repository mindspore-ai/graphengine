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

#ifndef MSPROF_ENGINE_PROF_ACL_API_H_
#define MSPROF_ENGINE_PROF_ACL_API_H_

#define MSVP_MAX_DEV_NUM 64
#define MSVP_PROF_API __attribute__((visibility("default")))

// DataTypeConfig
#define PROF_ACL_API                0x0001
#define PROF_TASK_TIME              0x0002
#define PROF_AICORE_METRICS         0x0004
#define PROF_AICPU_TRACE            0x0008
#define PROF_MODEL_EXECUTE          0x0010
#define PROF_RUNTIME_API            0x0020
#define PROF_RUNTIME_TRACE          0x0040
#define PROF_SCHEDULE_TIMELINE      0x0080
#define PROF_SCHEDULE_TRACE         0x0100
#define PROF_AIVECTORCORE_METRICS   0x0200
#define PROF_SUBTASK_TIME           0x0400

#define PROF_TRAINING_TRACE         0x0800
#define PROF_HCCL_TRACE             0x1000
#define PROF_DATA_PROCESS           0x2000
#define PROF_TASK_TRACE             0x3842

#define PROF_MODEL_LOAD             0x8000000000000000

// DataTypeConfig MASK
#define PROF_ACL_API_MASK                0x0001
#define PROF_TASK_TIME_MASK              0x0002
#define PROF_AICORE_METRICS_MASK         0x0004
#define PROF_AICPU_TRACE_MASK            0x0008
#define PROF_MODEL_EXECUTE_MASK          0x0010
#define PROF_RUNTIME_API_MASK            0x0020
#define PROF_RUNTIME_TRACE_MASK          0x0040
#define PROF_SCHEDULE_TIMELINE_MASK      0x0080
#define PROF_SCHEDULE_TRACE_MASK         0x0100
#define PROF_AIVECTORCORE_METRICS_MASK   0x0200
#define PROF_SUBTASK_TIME_MASK           0x0400

#define PROF_TRAINING_TRACE_MASK         0x0800
#define PROF_HCCL_TRACE_MASK             0x1000
#define PROF_DATA_PROCESS_MASK           0x2000

#define PROF_MODEL_LOAD_MASK             0x8000000000000000

#include <cstdint>
#include <string>

/**
 * @name  ProrErrorCode
 * @brief error code enum of prof_acl_apis
 */
enum ProfErrorCode {
    PROF_ERROR_NONE = 0,            // ok
    PROF_ERROR_PARAM_INVALID,       // param invalid, for example nullptr
    PROF_ERROR_REPEAT_INIT,         // profiling has already been inited
    PROF_ERROR_CONFIG_INVALID,      // config invalid, for example invalid json string
    PROF_ERROR_DIR_NO_ACCESS,       // dir is not accessable
    PROF_ERROR_FAILURE,             // failed to init or start profiling
    PROF_ERROR_NOT_INITED,          // profiling has not been inited
    PROF_ERROR_DEVICE_INVALID,      // device id invalid
    PROF_ERROR_UNSUPPORTED,         // unsupported data type or ai core metrics
    PROF_ERROR_REPEAT_START,        // profiilng has already been started
    PROF_ERROR_NOT_STARTED,         // profiling has not been started
};

/**
 * @brief transfer profiling config in acl.json to sample config
 * @param aclCfg       [IN]  profiling json string from acl.json as {"switch":"on", "result_path":"/home",...}
 * @param sampleCfg    [OUT] json string for GE as {"startCfg":[{"deviceID":"all","jobID":"1234",...}]}
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfAclCfgToSampleCfg(const std::string &aclCfg, std::string &sampleCfg);

/**
 * @name  ProfInit
 * @brief init profiling
 * @param profInitCfg [IN] config of init profiling of json format
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfInit(const std::string &profInitCfg);

/**
 * @name  ProfAicoreMetrics
 * @brief aicore metrics enum
 */
enum ProfAicoreMetrics {
    PROF_AICORE_ARITHMATIC_THROUGHPUT = 0,
    PROF_AICORE_PIPELINE = 1,
    PROF_AICORE_SYNCHRONIZATION = 2,
    PROF_AICORE_MEMORY = 3,
    PROF_AICORE_INTERNAL_MEMORY = 4,
    PROF_AICORE_STALL = 5,
    PROF_AICORE_EVENT = 255
};

/**
 * @name  ProfConfig
 * @brief struct of ProfStart
 */
struct ProfConfig {
    uint32_t devNums;                     // length of device id list
    uint32_t devIdList[MSVP_MAX_DEV_NUM]; // physical device id list
    ProfAicoreMetrics aicoreMetrics;      // aicore metric
    uint64_t dataTypeConfig;              // data type to start profiling
};

/**
 * @name  ProfStartProfiling
 * @brief start profiling
 * @param profStartCfg [IN] config to start profiling
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfStartProfiling(const ProfConfig *profStartCfg);

/**
 * @name  ProfStopConfig
 * @brief struct of ProfStop
 */
struct ProfStopConfig {
    uint64_t padding;
};

/**
 * @name  ProfStopProfiling
 * @brief stop profiling
 * @param profStopCfg [IN] config to stop profiling
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfStopProfiling(const ProfConfig *profStopCfg);

/**
 * @name  ProfFinalize
 * @brief finalize profiling task
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfFinalize();

/**
 * @name  ProfGetDataTypeConfig
 * @brief get dataTypeConfig started with of one device
 * @param deviceId          [IN] deviceId to get dataTypeConfig
 * @param dataTypeConfig    [OUT] result get
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfGetDataTypeConfig(uint32_t deviceId, uint64_t &dataTypeConfig);

#endif  // MSPROF_ENGINE_PROF_ACL_API_H_
