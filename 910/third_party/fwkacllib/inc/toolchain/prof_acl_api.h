/**
 * @file prof_acl_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#ifndef MSPROFILER_API_PROF_ACL_API_H
#define MSPROFILER_API_PROF_ACL_API_H

#include <cstdint>
#include <cstddef>
#include "prof_data_config.h"

constexpr int32_t PROF_MAX_DEV_NUM = 64; // 64 : dev max number
constexpr int32_t PROF_DEFAULT_HOST_ID = PROF_MAX_DEV_NUM;

/**
 * @name  ProfAicoreMetrics
 * @brief aicore metrics enum
 */
enum ProfAicoreMetrics {
    PROF_AICORE_ARITHMETIC_UTILIZATION = 0,
    PROF_AICORE_PIPE_UTILIZATION = 1,
    PROF_AICORE_MEMORY_BANDWIDTH = 2,
    PROF_AICORE_L0B_AND_WIDTH = 3,
    PROF_AICORE_RESOURCE_CONFLICT_RATIO = 4,
    PROF_AICORE_MEMORY_UB = 5,
    PROF_AICORE_L2_CACHE = 6,
    PROF_AICORE_PIPE_EXECUTE_UTILIZATION = 7,
    PROF_AICORE_METRICS_COUNT,
    PROF_AICORE_NONE = 0xFF,
};

/**
 * @name  ProfConfig
 * @brief struct of aclprofStart/aclprofStop
 */
struct ProfConfig {
    uint32_t devNums;                       // length of device id list
    uint32_t devIdList[PROF_MAX_DEV_NUM + 1];   // physical device id list
    ProfAicoreMetrics aicoreMetrics;        // aicore metric
    uint64_t dataTypeConfig;                // data type to start profiling
};
using PROF_CONF_CONST_PTR = const ProfConfig *;

/**
 * @name  ProfSubscribeConfig
 * @brief config of subscribe api
 */
struct ProfSubscribeConfig {
    bool timeInfo;                          // subscribe op time
    ProfAicoreMetrics aicoreMetrics;        // subscribe ai core metrics
    void* fd;                               // pipe fd
};
using PROF_SUB_CONF_CONST_PTR = const ProfSubscribeConfig *;

/**
 * @name  aclprofSubscribeConfig
 * @brief config of subscribe common api
 */
struct aclprofSubscribeConfig {
    struct ProfSubscribeConfig config;
};
using ACL_PROF_SUB_CONFIG_PTR = aclprofSubscribeConfig *;
using ACL_PROF_SUB_CINFIG_CONST_PTR = const aclprofSubscribeConfig *;

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

using PROFAPI_SUBSCRIBECONFIG_CONST_PTR = const void *;
namespace Msprofiler {
namespace Api {
/**
 * @name  ProfGetOpExecutionTime
 * @brief get op execution time of specific part of data
 * @param data  [IN] data read from pipe
 * @param len   [IN] data length
 * @param index [IN] index of part(op)
 * @return op execution time (us)
 */
MSVP_PROF_API uint64_t ProfGetOpExecutionTime(const void *data, uint32_t len, uint32_t index);
}
}

#ifdef __cplusplus
extern "C" {
#endif

MSVP_PROF_API uint64_t ProfGetOpExecutionTime(const void *data, uint32_t len, uint32_t index);
MSVP_PROF_API int32_t ProfOpSubscribe(uint32_t devId, PROFAPI_SUBSCRIBECONFIG_CONST_PTR profSubscribeConfig);
MSVP_PROF_API int32_t ProfOpUnSubscribe(uint32_t devId);

using Status = int32_t;
typedef struct aclprofSubscribeConfig aclprofSubscribeConfig1;
/**
 * @ingroup AscendCL
 * @brief subscribe profiling data of graph
 * @param [in] graphId: the graph id subscribed
 * @param [in] profSubscribeConfig: pointer to config of model subscribe
 * @return Status result of function
 */
MSVP_PROF_API Status aclgrphProfGraphSubscribe(const uint32_t graphId,
    const aclprofSubscribeConfig1 *profSubscribeConfig);
/**
 * @ingroup AscendCL
 * @brief unsubscribe profiling data of graph
 * @param [in] graphId: the graph id subscribed
 * @return Status result of function
 */
MSVP_PROF_API Status aclgrphProfGraphUnSubscribe(const uint32_t graphId);

/**
 * @ingroup AscendCL
 * @brief get graph id from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 *
 * @retval graph id of subscription data
 * @retval 0 for failed
 */
MSVP_PROF_API size_t aclprofGetGraphId(const void *opInfo, size_t opInfoLen, uint32_t index);

/**
* @ingroup AscendCL
* @brief set stamp pay load
*
*
* @retval void
*/
MSVP_PROF_API int aclprofSetStampPayload(void *stamp, const int32_t type, void *value);

/**
* @ingroup AscendCL
* @brief set category and name
*
*
* @retval void
*/
MSVP_PROF_API int aclprofSetCategoryName(uint32_t category, const char *categoryName);

/**
* @ingroup AscendCL
* @brief set category to stamp
*
*
* @retval void
*/
MSVP_PROF_API int aclprofSetStampCategory(void *stamp, uint32_t category);

#ifdef __cplusplus
}
#endif

#endif  // MSPROFILER_API_PROF_ACL_API_H_
