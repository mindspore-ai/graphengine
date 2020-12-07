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

#ifndef MSPROFILER_API_PROF_ACL_API_H_
#define MSPROFILER_API_PROF_ACL_API_H_

#define MSVP_MAX_DEV_NUM 64
#ifndef OS_TYPE
#define OS_TYPE 0
#endif // OS_TYPE


#if (OS_TYPE != LINUX)
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

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
    PROF_ERROR_REPEAT_SUBSCRIBE,    // same model id has already been subscribed
    PROF_ERROR_MODEL_ID_INVALID,    // model id does not exist or has not been subscribed
    PROF_ERROR_API_CONFLICT,        // prof ctrl api mode conflicts with subscribe mode
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
    PROF_AICORE_METRICS_COUNT,
    PROF_AICORE_NONE = 0xff,
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

namespace Msprofiler {
namespace Api {
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
 * @name  ProfStartProfiling
 * @brief start profiling
 * @param profStartCfg [IN] config to start profiling
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfStartProfiling(const ProfConfig *profStartCfg);

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

/**
 * @name  WorkMode
 * @brief profiling api work mode
 */
enum WorkMode {
    WORK_MODE_OFF,          // profiling not at work
    WORK_MODE_API_CTRL,     // profiling work on api ctrl mode, (ProfInit)
    WORK_MODE_SUBSCRIBE,    // profiling work on subscribe mode
};

/**
 * @name  ProfGetApiWorkMode
 * @brief get profiling api work mode
 * @return WorkMode
 */
MSVP_PROF_API WorkMode ProfGetApiWorkMode();

/**
 * @name  ProfSubscribeConfig
 * @brief config of subscribe api
 */
struct ProfSubscribeConfig {
    bool timeInfo;                      // subscribe op time
    ProfAicoreMetrics aicoreMetrics;    // subscribe ai core metrics
    void* fd;                           // pipe fd
};

/**
 * @name  ProfGetDataTypeConfig
 * @brief get DataTypeConfig of subscribe
 * @param profSubscribeConfig [IN] config to subscribe data
 * @return DataTypeConfig
 */
MSVP_PROF_API uint64_t ProfGetDataTypeConfig(const ProfSubscribeConfig *profSubscribeConfig);

/**
 * @name  ProfModelSubscribe
 * @brief subscribe data of one model id
 * @param modelId [IN] model id to subscribe data
 * @param devId [IN] device id of model
 * @param profSubscribeConfig [IN] config to subscribe data
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfModelSubscribe(uint32_t modelId, uint32_t devId,
                                         const ProfSubscribeConfig *profSubscribeConfig);

/**
 * @name  ProfIsModelSubscribed
 * @brief check if a model id is subscribed
 * @param modeiId [IN] modei id to check
 * @return true: subscribed, false: not
 */
MSVP_PROF_API bool ProfIsModelSubscribed(uint32_t modelId);

/**
 * @name  ProfModelUnSubscribe
 * @brief unsubscribe a model id
 * @param modeiId [IN] modei id to unsubscribe
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfModelUnSubscribe(uint32_t modelId);

/**
 * @name  ProfGetOpDescSize
 * @brief get profiling data struct size
 * @param opDescSize [OUT] bytes of profiling subscribe data struct
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfGetOpDescSize(uint32_t *opDescSize);

/**
 * @name  ProfGetOpNum
 * @brief get how many op data there are in data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param opNum [OUT] number of op in data
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfGetOpNum(const void *data, uint32_t len, uint32_t *opNum);

/**
 * @name  ProfGetModelId
 * @brief get model id of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param index [IN] index of part(op)
 * @return model id
 */
MSVP_PROF_API uint32_t ProfGetModelId(const void *data, uint32_t len, uint32_t index);

/**
 * @name  ProfGetOpType
 * @brief get op type of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param opType [OUT] op type buffer
 * @param opTypeLen [IN] buffer size of param opType
 * @param index [IN] index of part(op)
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfGetOpType(const void *data, uint32_t len, char *opType, uint32_t opTypeLen, uint32_t index);

/**
 * @name  ProfGetOpName
 * @brief get op name of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param opType [OUT] op name buffer
 * @param opTypeLen [IN] buffer size of param opName
 * @param index [IN] index of part(op)
 * @return ProfErrorCode
 */
MSVP_PROF_API int32_t ProfGetOpName(const void *data, uint32_t len, char *opName, uint32_t opNameLen, uint32_t index);

/**
 * @name  ProfGetOpStart
 * @brief get op start timestamp of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param index [IN] index of part(op)
 * @return op start timestamp (us)
 */
MSVP_PROF_API uint64_t ProfGetOpStart(const void *data, uint32_t len, uint32_t index);

/**
 * @name  ProfGetOpEnd
 * @brief get op end timestamp of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param index [IN] index of part(op)
 * @return op end timestamp (us)
 */
MSVP_PROF_API uint64_t ProfGetOpEnd(const void *data, uint32_t len, uint32_t index);

/**
 * @name  ProfGetOpDuration
 * @brief get op duration of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param index [IN] index of part(op)
 * @return op duration (us)
 */
MSVP_PROF_API uint64_t ProfGetOpDuration(const void *data, uint32_t len, uint32_t index);

/**
 * @name  ProfGetOpExecutionTime
 * @brief get op execution time of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param index [IN] index of part(op)
 * @return op execution time (us)
 */
MSVP_PROF_API uint64_t ProfGetOpExecutionTime(const void *data, uint32_t len, uint32_t index);

/**
 * @name  ProfGetOpCubeOps
 * @brief get op cube fops of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param index [IN] index of part(op)
 * @return op cube fops
 */
MSVP_PROF_API uint64_t ProfGetOpCubeOps(const void *data, uint32_t len, uint32_t index);

/**
 * @name  ProfGetOpVectorOps
 * @brief get op vector fops of specific part of data
 * @param data [IN] data read from pipe
 * @param len [IN] data length
 * @param index [IN] index of part(op)
 * @return op vector fops
 */
MSVP_PROF_API uint64_t ProfGetOpVectorOps(const void *data, uint32_t len, uint32_t index);

}   // namespace Api
}   // namespace Msprofiler

#endif  // MSPROFILER_API_PROF_ACL_API_H_
