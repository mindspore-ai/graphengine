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

#ifndef __CCE_RUNTIME_BASE_H__
#define __CCE_RUNTIME_BASE_H__

#include <stdint.h>

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

// If you need export the function of this library in Win32 dll, use __declspec(dllexport)
#ifndef RTS_API
#ifdef RTS_DLL_EXPORT
#define RTS_API __declspec(dllexport)
#else
#define RTS_API
#endif
#endif

/**
 * @ingroup dvrt_base
 * @brief runtime error numbers.
 */
typedef enum tagRtError {
    RT_ERROR_NONE = 0x0,                    // success
    
    RT_ERROR_DEVICE_BASE                    = 0x07010000,
    RT_ERROR_DEVICE_NULL,
    RT_ERROR_DEVICE_NEW,
    RT_ERROR_DEVICE_ID,
    RT_ERROR_DEVICE_CHIPTYPE,
    RT_ERROR_DEVICE_DEPLOY,
    RT_ERROR_DEVICE_RETAIN,
    RT_ERROR_DEVICE_PLATFORM,
    RT_ERROR_DEVICE_LOADER,
    RT_ERROR_DEVICE_LIMIT,
    RT_ERROR_DEVICE_PROC_HANG_OUT,
    RT_ERROR_DEVICE_POWER_UP_FAIL,
    RT_ERROR_DEVICE_POWER_DOWN_FAIL,
    RT_ERROR_DEVICE_INVALID,

    RT_ERROR_DRV_BASE                       = 0x07020000,
    RT_ERROR_DRV_NULL,
    RT_ERROR_DRV_NEW,
    RT_ERROR_DRV_MEMORY,
    RT_ERROR_DRV_INPUT,
    RT_ERROR_DRV_PTRNULL,
    RT_ERROR_DRV_OPEN_AICPU,
    RT_ERROR_DRV_CLOSE_AICPU,
    RT_ERROR_DRV_SYM_AICPU,
    RT_ERROR_DRV_OPEN_TSD,
    RT_ERROR_DRV_CLOSE_TSD,
    RT_ERROR_DRV_SYM_TSD,
    RT_ERROR_DRV_SOURCE,
    RT_ERROR_DRV_REPORT,
    RT_ERROR_DRV_COMMAND,
    RT_ERROR_DRV_OCCUPY,
    RT_ERROR_DRV_ERR,

    RT_ERROR_STREAM_BASE                    = 0x07030000,
    RT_ERROR_STREAM_NULL,
    RT_ERROR_STREAM_NEW,
    RT_ERROR_STREAM_CONTEXT,
    RT_ERROR_STREAM_INVALID,
    RT_ERROR_STREAM_MODEL,
    RT_ERROR_STREAM_FUSION,
    RT_ERROR_STREAM_FULL,
    RT_ERROR_STREAM_EMPTY,
    RT_ERROR_STREAM_NOT_COMPLETE,
    RT_ERROR_STREAM_SYNC,
    RT_ERROR_STREAM_NO_CB_REG,
    RT_ERROR_STREAM_DUPLICATE,
    RT_ERROR_STREAM_NOT_EXIST,
    RT_ERROR_SQ_NO_EXIST_SQ_TO_REUSE,
    RT_ERROR_SQID_FULL,

    RT_ERROR_MODEL_BASE                     = 0x07040000,
    RT_ERROR_MODEL_NULL,
    RT_ERROR_MODEL_NEW,
    RT_ERROR_MODEL_CONTEXT,
    RT_ERROR_MODEL_ENDGRAPH,
    RT_ERROR_MODEL_STREAM,
    RT_ERROR_MODEL_EXCUTOR,
    RT_ERROR_MODEL_SETUP,
    RT_ERROR_MODEL_ID,
    RT_ERROR_MODEL_EXE_FAILED,
    RT_ERROR_END_OF_SEQUENCE,               // end of sequence

    RT_ERROR_EVENT_BASE                     = 0x07050000,
    RT_ERROR_EVENT_NULL,
    RT_ERROR_EVENT_NEW,
    RT_ERROR_EVENT_RECORDER_NULL,
    RT_ERROR_EVENT_TIMESTAMP_INVALID,
    RT_ERROR_EVENT_TIMESTAMP_REVERSAL,
    RT_ERROR_EVENT_NOT_COMPLETE,

    RT_ERROR_NOTIFY_BASE                    = 0x07060000,
    RT_ERROR_NOTIFY_NULL,
    RT_ERROR_NOTIFY_NEW,
    RT_ERROR_NOTIFY_TYPE,
    RT_ERROR_NOTIFY_NOT_COMPLETE,

    RT_ERROR_CONTEXT_BASE                   = 0x07070000,
    RT_ERROR_CONTEXT_NULL,
    RT_ERROR_CONTEXT_NEW,
    RT_ERROR_CONTEXT_DEL,
    RT_ERROR_CONTEXT_DEFAULT_STREAM_NULL,
    RT_ERROR_CONTEXT_ONLINE_STREAM_NULL,

    RT_ERROR_KERNEL_BASE                    = 0x07080000,
    RT_ERROR_KERNEL_NULL,
    RT_ERROR_KERNEL_NEW,
    RT_ERROR_KERNEL_LOOKUP,
    RT_ERROR_KERNEL_NAME,
    RT_ERROR_KERNEL_TYPE,
    RT_ERROR_KERNEL_OFFSET,
    RT_ERROR_KERNEL_DUPLICATE,
    RT_ERROR_KERNEL_UNREGISTERING,

    RT_ERROR_PROGRAM_BASE                   = 0x07090000,
    RT_ERROR_PROGRAM_NULL,
    RT_ERROR_PROGRAM_NEW,
    RT_ERROR_PROGRAM_DATA,
    RT_ERROR_PROGRAM_SIZE,
    RT_ERROR_PROGRAM_MEM_TYPE,
    RT_ERROR_PROGRAM_MACHINE_TYPE,
    RT_ERROR_PROGRAM_USEOUT,

    RT_ERROR_MODULE_BASE                    = 0x070a0000,
    RT_ERROR_MODULE_NULL,
    RT_ERROR_MODULE_NEW,

    RT_ERROR_INSTANCE_BASE                  = 0x070b0000,
    RT_ERROR_INSTANCE_NULL,
    RT_ERROR_INSTANCE_NEW,
    RT_ERROR_INSTANCE_VERSION,

    RT_ERROR_API_BASE                       = 0x070c0000,
    RT_ERROR_API_NULL,
    RT_ERROR_API_NEW,

    RT_ERROR_DATADUMP_BASE                  = 0x070d0000,
    RT_ERROR_DATADUMP_NULL,
    RT_ERROR_DATADUMP_NEW,
    RT_ERROR_DATADUMP_TIME,
    RT_ERROR_DATADUMP_FILE,
    RT_ERROR_DATADUMP_ADDRESS,
    RT_ERROR_DATADUMP_LOAD_FAILED,
    RT_ERROR_DUMP_ADDR_SET_FAILED,

    RT_ERROR_PROF_BASE                      = 0x070e0000,
    RT_ERROR_PROF_NULL,
    RT_ERROR_PROF_NEW,
    RT_ERROR_PROF_START,
    RT_ERROR_PROF_DEVICE_MEM,
    RT_ERROR_PROF_HOST_MEM,
    RT_ERROR_PROF_SET_DIR,
    RT_ERROR_PROF_OPER,
    RT_ERROR_PROF_FULL,
    RT_ERROR_PROF_NAME,

    RT_ERROR_PCTRACE_BASE                   = 0x070f0000,
    RT_ERROR_PCTRACE_NULL,
    RT_ERROR_PCTRACE_NEW,
    RT_ERROR_PCTRACE_TIME,
    RT_ERROR_PCTRACE_FILE,

    RT_ERROR_TASK_BASE                      = 0x07100000,
    RT_ERROR_TASK_NULL,
    RT_ERROR_TASK_NEW,
    RT_ERROR_TASK_TYPE,
    RT_ERROR_TASK_ALLOCATOR,

    RT_ERROR_COMMON_BASE                    = 0x07110000,
    RT_ERROR_INVALID_VALUE,             // RT_ERROR_INPUT_INVALID
    RT_ERROR_MEMORY_ADDRESS_UNALIGNED,
    RT_ERROR_SEC_HANDLE,
    RT_ERROR_OS_HANDLE,
    RT_ERROR_MUTEX_LOCK,
    RT_ERROR_MUTEX_UNLOCK,
    RT_ERROR_CALLOC,
    RT_ERROR_POOL_RESOURCE,
    RT_ERROR_TRANS_ARGS,
    RT_ERROR_METADATA,
    RT_ERROR_LOST_HEARTBEAT,
    RT_ERROR_REPORT_TIMEOUT,
    RT_ERROR_FEATURE_NOT_SUPPROT,
    RT_ERROR_MEMORY_ALLOCATION,
    RT_ERROR_MEMORY_FREE,
    RT_ERROR_INVALID_MEMORY_TYPE,

    RT_ERROR_DEBUG_BASE                     = 0x07120000,
    RT_ERROR_DEBUG_NULL,
    RT_ERROR_DEBUG_NEW,
    RT_ERROR_DEBUG_SIGNAL,
    RT_ERROR_DEBUG_OPEN,
    RT_ERROR_DEBUG_WRITE,
    RT_ERROR_DEBUG_REGISTER_FAILED,
    RT_ERROR_DEBUG_UNREGISTER_FAILED,

    RT_ERROR_ENGINE_BASE                    = 0x07130000,
    RT_ERROR_ENGINE_NULL,
    RT_ERROR_ENGINE_NEW,
    RT_ERROR_ENGINE_THREAD,

    RT_ERROR_LABEL_BASE                     = 0x07140000,
    RT_ERROR_LABEL_NULL,
    RT_ERROR_LABEL_NEW,
    RT_ERROR_LABEL_CONTEXT,
    RT_ERROR_LABEL_STREAM,
    RT_ERROR_LABEL_MODEL,
    RT_ERROR_LABEL_ALLOCATOR,
    RT_ERROR_LABEL_FREE,
    RT_ERROR_LABEL_SET,
    RT_ERROR_LABEL_ID,

    RT_ERROR_TSFW_BASE                      = 0x07150000,
    RT_ERROR_TSFW_UNKNOWN,
    RT_ERROR_TSFW_NULL_PTR,
    RT_ERROR_TSFW_ILLEGAL_AI_CORE_ID,
    RT_ERROR_TSFW_ILLEGAL_PARAM,
    RT_ERROR_TSFW_TASK_CMD_QUEUE_FULL,
    RT_ERROR_TSFW_TASK_CMD_QUEUE_EMPTY,
    RT_ERROR_TSFW_TASK_REPORT_QUEUE_FULL,
    RT_ERROR_TSFW_TASK_REPORT_QUEUE_EMPTY,
    RT_ERROR_TSFW_TASK_NODE_BUFF_ALL_OCCUPYED,
    RT_ERROR_TSFW_TASK_NODE_BUFF_ALL_FREED,
    RT_ERROR_TSFW_L2_MEM_INSUFFICIENT_SPACE,
    RT_ERROR_TSFW_L2_MALLOC_FAILED,
    RT_ERROR_TSFW_DMA_CHANNEL_ALL_OCCUPYED,
    RT_ERROR_TSFW_MEMCPY_OP_FAILED,
    RT_ERROR_TSFW_BS_SLOT_ALL_OCCUPYED,
    RT_ERROR_TSFW_TBS_SLOT_REPEAT_FREE,
    RT_ERROR_TSFW_PRIORITY_TASK_LIST_FULL,
    RT_ERROR_TSFW_PRIORITY_TASK_LIST_EMPTY,
    RT_ERROR_TSFW_NO_STREAM_LIST_NEED_TO_BE_PROCESSED,
    RT_ERROR_TSFW_REPEAT_MARK_STREAM_NEED_SERVICE,
    RT_ERROR_TSFW_SYS_DMA_CHANNEL_ALL_OCCUPAPYED,
    RT_ERROR_TSFW_NO_HBML2TASKNODE_FOUND,
    RT_ERROR_TSFW_SQNODE_NODE_SLOT_ALL_OCCUPAPYED,
    RT_ERROR_TSFW_CQNODE_NODE_SLOT_ALL_OCCUPAPYED,
    RT_ERROR_TSFW_SQNODE_NOT_ENOUGH,
    RT_ERROR_TSFW_SQNODE_SLOT_REPEAT_FREE,
    RT_ERROR_TSFW_CQNODE_SLOT_REPEAT_FREE,
    RT_ERROR_TSFW_CQ_REPORT_FAILED,
    RT_ERROR_TSFW_SYS_DMA_RESET_SUCCESS,
    RT_ERROR_TSFW_SYS_DMA_RESET_FAILED,
    RT_ERROR_TSFW_SYS_DMA_TRNSFER_FAILED,
    RT_ERROR_TSFW_SYS_DMA_MEMADDRALIGN_FAILED,
    RT_ERROR_TSFW_SYS_DMA_ERROR_QUEUE_FULL,
    RT_ERROR_TSFW_SYS_DMA_ERROR_QUEUE_EMPTY,
    RT_ERROR_TSFW_TIMER_EVENT_FULL,
    RT_ERROR_TSFW_TASK_L2_DESC_ENTRY_NOT_ENOUGH,
    RT_ERROR_TSFW_AICORE_TIMEOUT,
    RT_ERROR_TSFW_AICORE_EXCEPTION,
    RT_ERROR_TSFW_AICORE_TRAP_EXCEPTION,
    RT_ERROR_TSFW_AICPU_TIMEOUT,
    RT_ERROR_TSFW_SDMA_L2_TO_DDR_MALLOC_FAIL,
    RT_ERROR_TSFW_AICPU_EXCEPTION,
    RT_ERROR_TSFW_AICPU_DATADUMP_RSP_ERR,
    RT_ERROR_TSFW_AICPU_MODEL_RSP_ERR,
    RT_ERROR_TSFW_REPEAT_ACTIVE_MODEL_STREAM,
    RT_ERROR_TSFW_REPEAT_NOTIFY_WAIT,
    RT_ERROR_TSFW_DEBUG_INVALID_SQCQ,
    RT_ERROR_TSFW_DEBUG_WRONG_COMMAND_TYPE,
    RT_ERROR_TSFW_DEBUG_CMD_PROCESS,
    RT_ERROR_TSFW_DEBUG_INVALID_DEVICE_STATUS,
    RT_ERROR_TSFW_DEBUG_NOT_IN_DEBUG_STATUS,
    RT_ERROR_TSFW_DEBUG_INVALID_TASK_STATUS,
    RT_ERROR_TSFW_DEBUG_TASK_EMPTY,
    RT_ERROR_TSFW_DEBUG_TASK_FULL,
    RT_ERROR_TSFW_DEBUG_TASK_NOT_EXIST,
    RT_ERROR_TSFW_DEBUG_AI_CORE_FULL,
    RT_ERROR_TSFW_DEBUG_AI_CORE_NOT_EXIST,
    RT_ERROR_TSFW_DEBUG_AI_CORE_EXCEPTION,
    RT_ERROR_TSFW_DEBUG_AI_CORE_TIMEOUT,
    RT_ERROR_TSFW_DEBUG_BREAKPOINT_FULL,
    RT_ERROR_TSFW_DEBUG_READ_ERROR,
    RT_ERROR_TSFW_DEBUG_WRITE_FAIL,
    RT_ERROR_TSFW_QUEUE_FULL,
    RT_ERROR_TSFW_QUEUE_EMPTY,
    RT_ERROR_TSFW_QUEUE_ALLOC_MEM_FAIL,
    RT_ERROR_TSFW_QUEUE_DATA_SIZE_UNMATCH,
    RT_ERROR_TSFW_PCIE_DMA_INVLD_CPY_TYPE,
    RT_ERROR_TSFW_INVLD_CPY_DIR,
    RT_ERROR_TSFW_PCIE_DMA_INVLD_CQ_DES,
    RT_ERROR_TSFW_PCIE_DMA_CPY_ERR,
    RT_ERROR_TSFW_PCIE_DMA_LNK_CHN_BUSY,
    RT_ERROR_TSFW_PROFILE_BUFF_FULL,
    RT_ERROR_TSFW_PROFILE_MODE_CONFLICT,
    RT_ERROR_TSFW_PROFILE_OTHER_PID_ON,
    RT_ERROR_TSFW_SCHD_AIC_TASK_PRELOAD_FAILED,
    RT_ERROR_TSFW_TSCPU_CLOSE_FAILED,
    RT_ERROR_TSFW_EXPECT_FAIL,
    RT_ERROR_TSFW_REPEAT_MODEL_STREAM,
    RT_ERROR_TSFW_STREAM_MODEL_UNBIND,
    RT_ERROR_TSFW_MODEL_EXE_FAILED,
    RT_ERROR_TSFW_IPC_SEND_FAILED,
    RT_ERROR_TSFW_IPC_PROC_REG_FAILED,
    RT_ERROR_TSFW_STREAM_FULL,
    RT_ERROR_TSFW_END_OF_SEQUENCE,
    RT_ERROR_TSFW_SWITCH_STREAM_LABEL,
    RT_ERROR_TSFW_TRANS_SQE_FAIL,
    RT_ERROR_TSFW_RESERVED,

    RT_ERROR_SUBSCRIBE_BASE                = 0x07160000,
    RT_ERROR_SUBSCRIBE_NULL,
    RT_ERROR_SUBSCRIBE_NEW,
    RT_ERROR_SUBSCRIBE_STREAM,
    RT_ERROR_SUBSCRIBE_THREAD,
    RT_ERROR_SUBSCRIBE_GROUP,

    RT_ERROR_GROUP_BASE                    = 0x07170000,
    RT_ERROR_GROUP_NOT_SET,
    RT_ERROR_GROUP_NOT_CREATE,

    RT_ERROR_RESERVED                      = 0x07ff0000,
  }rtError_t;

/**
 * @ingroup dvrt_base
 * @brief runtime exception numbers.
 */
typedef enum tagRtExceptionType {
  RT_EXCEPTION_NONE = 0,
  RT_EXCEPTION_TS_DOWN = 1,
  RT_EXCEPTION_TASK_TIMEOUT = 2,
  RT_EXCEPTION_TASK_FAILURE = 3,
  RT_EXCEPTION_DEV_RUNNING_DOWN = 4,
  RT_EXCEPTION_STREAM_ID_FREE_FAILED = 5
} rtExceptionType;

/**
 * @ingroup dvrt_base
 * @brief Switch type.
 */
typedef enum tagRtCondition {
  RT_EQUAL = 0,
  RT_NOT_EQUAL,
  RT_GREATER,
  RT_GREATER_OR_EQUAL,
  RT_LESS,
  RT_LESS_OR_EQUAL
} rtCondition_t;

/**
 * @ingroup dvrt_base
 * @brief Data Type of Extensible Switch Task.
 */
typedef enum tagRtSwitchDataType {
  RT_SWITCH_INT32 = 0,
  RT_SWITCH_INT64 = 1,
} rtSwitchDataType_t;

typedef enum tagRtStreamFlagType {
  RT_HEAD_STREAM = 0,  // first stream
  RT_INVALID_FLAG = 0xFFFFFFFF,
} rtStreamFlagType_t;

typedef enum tagRtLimitType {
  RT_LIMIT_TYPE_LOW_POWER_TIMEOUT = 0,  // timeout for power down , ms
} rtLimitType_t;

typedef struct rtExceptionInfo {
    uint32_t taskid;
    uint32_t streamid;
    uint32_t tid;
    uint32_t deviceid;
} rtExceptionInfo;

typedef void (*rtErrorCallback)(rtExceptionType);

typedef void (*rtTaskFailCallback)(rtExceptionInfo *exceptionInfo);

/**
 * @ingroup dvrt_base
 * @brief stream handle.
 */
typedef void *rtStream_t;

/**
 * @ingroup dvrt_base
 * @brief runtime event handle.
 */
typedef void *rtEvent_t;

/**
 * @ingroup dvrt_base
 * @brief label handle.
 */
typedef void *rtLabel_t;

/**
 * @ingroup profiling_base
 * @brief runtime handle.
 */
RTS_API rtError_t rtSetProfDirEx(const char *profDir, const char *address, const char *jobCtx);

/**
 * @ingroup profiling_base
 * @brief init profiler object.
 */
RTS_API rtError_t rtProfilerInit(const char *profdir, const char *address, const char *job_ctx);

/**
 * @ingroup profiling_base
 * @brief config rts profiler.
 */
RTS_API rtError_t rtProfilerConfig(uint16_t type);

/**
 * @ingroup profiling_base
 * @brief start rts profiler.
 */
RTS_API rtError_t rtProfilerStart(uint64_t profConfig, int32_t numsDev, uint32_t* deviceList);

/**
 * @ingroup profiling_base
 * @brief stop rts profiler.
 */
RTS_API rtError_t rtProfilerStop(uint64_t profConfig, int32_t numsDev, uint32_t* deviceList);

/**
 * @ingroup profiling_base
 * @brief ts send keypoint profiler log.
 */
RTS_API rtError_t rtProfilerTrace(uint64_t id, bool notify, uint32_t flags, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief Returns the last error from a runtime call.
 */
RTS_API rtError_t rtGetLastError();

/**
 * @ingroup dvrt_base
 * @brief Returns the last error from a runtime call.
 */
RTS_API rtError_t rtPeekAtLastError();

/**
 * @ingroup dvrt_base
 * @brief register callback for error code
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtSetExceptCallback(rtErrorCallback callback);

/**
 * @ingroup dvrt_base
 * @brief register callback for task fail
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtSetTaskFailCallback(rtTaskFailCallback callback);

/**
 * @ingroup dvrt_base
 * @brief notify handle.
 */
typedef void *rtNotify_t;

/**
 * @ingroup dvrt_base
 * @brief create label instance
 * @param [out]    label   created label
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreate(rtLabel_t *label);

/**
 * @ingroup dvrt_base
 * @brief set label and stream instance
 * @param [in] label   set label
 * @param [in] stream  set stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSet(rtLabel_t label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief destroy label instance
 * @param [in] label   label to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelDestroy(rtLabel_t label);

/**
 * @ingroup dvrt_base
 * @brief label switch instance
 * @param [in] ptr  address to get value compared
 * @param [in] condition
 * @param [in] value  to compare
 * @param [in] true_label   goto label
 * @param [in] stream  to submit label_switch task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSwitch(void *ptr, rtCondition_t condition, uint32_t value, rtLabel_t trueLabel,
                                rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief goto label instance
 * @param [in] label   goto label
 * @param [in] stream  to submit label_goto task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelGoto(rtLabel_t label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief name label instance
 * @param [in] label  instance
 * @param [in] name  label name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNameLabel(rtLabel_t label, const char *name);

/**
 * @ingroup dvrt_base
 * @brief label switch by index
 * @param [in] ptr  index value ptr
 * @param [in] max  index max value
 * @param [in] labelInfoPtr  label content info ptr
 * @param [in] stream  set stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSwitchByIndex(void *ptr, uint32_t max, void *labelInfoPtr, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief stream goto label
 * @param [in] label  goto label
 * @param [in] stream  stream  to submit label_goto task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelGotoEx(rtLabel_t label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [in] label  model label list
 * @param [in] labelNumber  label number
 * @param [in] dst  device ptr
 * @param [in] dstMax  dst size
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelListCpy(rtLabel_t *label, uint32_t labelNumber, void *dst, uint32_t dstMax);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [out] label  created label handle
 * @param [in] stream  label bind stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreateEx(rtLabel_t *label, rtStream_t stream);

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif

#endif  // __CCE_RUNTIME_BASE_H__
