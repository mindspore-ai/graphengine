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

#ifndef MSPROFILER_API_PROF_ACL_API_H_
#define MSPROFILER_API_PROF_ACL_API_H_

#include <cstdint>
#include <cstddef>

// DataTypeConfig
constexpr uint64_t PROF_ACL_API        = 0x00000001ULL;
constexpr uint64_t PROF_TASK_TIME      = 0x00000002ULL; // dynamic profiling hwts log
constexpr uint64_t PROF_AICORE_METRICS = 0x00000004ULL; // dynamic profiling hwts profile
constexpr uint64_t PROF_AICPU_TRACE    = 0x00000008ULL;
constexpr uint64_t PROF_L2CACHE        = 0x00000010ULL;
constexpr uint64_t PROF_HCCL_TRACE     = 0x00000020ULL;
constexpr uint64_t PROF_TRAINING_TRACE = 0x00000040ULL;
constexpr uint64_t PROF_MSPROFTX       = 0x00000080ULL;
constexpr uint64_t PROF_RUNTIME_API    = 0x00000100ULL;
constexpr uint64_t PROF_TASK_FRAMEWORK = 0x00000200ULL;
constexpr uint64_t PROF_TASK_TSFW      = 0x00000400ULL;

// system profilinig switch
constexpr uint64_t PROF_CPU                  = 0x00010000ULL;
constexpr uint64_t PROF_HARDWARE_MEMORY      = 0x00020000ULL;
constexpr uint64_t PROF_IO                   = 0x00040000ULL;
constexpr uint64_t PROF_INTER_CONNECTION     = 0x00080000ULL;
constexpr uint64_t PROF_DVPP                 = 0x00100000ULL;
constexpr uint64_t PROF_SYS_AICORE_SAMPLE    = 0x00200000ULL;
constexpr uint64_t PROF_AIVECTORCORE_SAMPLE  = 0x00400000ULL;
constexpr uint64_t PROF_INSTR                = 0x00800000ULL;

constexpr uint64_t PROF_MODEL_EXECUTE        = 0x0000001000000ULL;
constexpr uint64_t PROF_RUNTIME_TRACE        = 0x0000004000000ULL;
constexpr uint64_t PROF_SCHEDULE_TIMELINE    = 0x0000008000000ULL;
constexpr uint64_t PROF_SCHEDULE_TRACE       = 0x0000010000000ULL;
constexpr uint64_t PROF_AIVECTORCORE_METRICS = 0x0000020000000ULL;
constexpr uint64_t PROF_SUBTASK_TIME         = 0x0000040000000ULL;
constexpr uint64_t PROF_OP_DETAIL            = 0x0000080000000ULL;

constexpr uint64_t PROF_AICPU_MODEL          = 0x4000000000000000ULL;
constexpr uint64_t PROF_MODEL_LOAD           = 0x8000000000000000ULL;

constexpr uint64_t PROF_TASK_TRACE = (PROF_MODEL_EXECUTE | PROF_RUNTIME_TRACE | PROF_TRAINING_TRACE |
                                      PROF_HCCL_TRACE | PROF_TASK_TIME);

// DataTypeConfig MASK
constexpr uint64_t PROF_ACL_API_MASK        = 0x00000001ULL;
constexpr uint64_t PROF_TASK_TIME_MASK      = 0x00000002ULL;
constexpr uint64_t PROF_AICORE_METRICS_MASK = 0x00000004ULL;
constexpr uint64_t PROF_AICPU_TRACE_MASK    = 0x00000008ULL;
constexpr uint64_t PROF_L2CACHE_MASK        = 0x00000010ULL;
constexpr uint64_t PROF_HCCL_TRACE_MASK     = 0x00000020ULL;
constexpr uint64_t PROF_TRAINING_TRACE_MASK = 0x00000040ULL;
constexpr uint64_t PROF_MSPROFTX_MASK       = 0x00000080ULL;
constexpr uint64_t PROF_RUNTIME_API_MASK    = 0x00000100ULL;
constexpr uint64_t PROF_TASK_FRAMEWORK_MASK = 0x00000200ULL;
constexpr uint64_t PROF_TASK_TSFW_MASK      = 0x00000400ULL;

// system profilinig mask
constexpr uint64_t PROF_CPU_MASK                  = 0x00010000ULL;
constexpr uint64_t PROF_HARDWARE_MEMORY_MASK      = 0x00020000ULL;
constexpr uint64_t PROF_IO_MASK                   = 0x00040000ULL;
constexpr uint64_t PROF_INTER_CONNECTION_MASK     = 0x00080000ULL;
constexpr uint64_t PROF_DVPP_MASK                 = 0x00100000ULL;
constexpr uint64_t PROF_SYS_AICORE_SAMPLE_MASK    = 0x00200000ULL;
constexpr uint64_t PROF_AIVECTORCORE_SAMPLE_MASK  = 0x00400000ULL;
constexpr uint64_t PROF_INSTR_MASK                = 0x00800000ULL;

constexpr uint64_t PROF_MODEL_EXECUTE_MASK        = 0x0000001000000ULL;
constexpr uint64_t PROF_RUNTIME_TRACE_MASK        = 0x0000004000000ULL;
constexpr uint64_t PROF_SCHEDULE_TIMELINE_MASK    = 0x0000008000000ULL;
constexpr uint64_t PROF_SCHEDULE_TRACE_MASK       = 0x0000010000000ULL;
constexpr uint64_t PROF_AIVECTORCORE_METRICS_MASK = 0x0000020000000ULL;
constexpr uint64_t PROF_SUBTASK_TIME_MASK         = 0x0000040000000ULL;
constexpr uint64_t PROF_OP_DETAIL_MASK            = 0x0000080000000ULL;

constexpr uint64_t PROF_AICPU_MODEL_MASK          = 0x4000000000000000ULL;
constexpr uint64_t PROF_MODEL_LOAD_MASK           = 0x8000000000000000ULL;

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

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
