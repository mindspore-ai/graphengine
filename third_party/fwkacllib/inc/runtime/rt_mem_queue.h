/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: mbuf and queue interface
 */

#ifndef CCE_RUNTIME_RT_MEM_QUEUE_H
#define CCE_RUNTIME_RT_MEM_QUEUE_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define RT_MQ_MAX_NAME_LEN 128 // same as driver's
#define RT_MQ_DEPTH_MIN 2U
#define RT_MQ_MODE_PUSH 1
#define RT_MQ_MODE_PULL 2
#define RT_MQ_MODE_DEFAULT RT_MQ_MODE_PUSH

typedef struct tagMemQueueAttr {
    char name[RT_MQ_MAX_NAME_LEN];
    uint32_t depth;
    uint32_t workMode;
    uint32_t flowCtrlDropTime;
    bool flowCtrlFlag;
    bool overWriteFlag;
} rtMemQueueAttr_t;

typedef struct tagMemQueueShareAttr {
    uint32_t manage : 1;
    uint32_t read : 1;
    uint32_t write : 1;
    uint32_t rsv : 29;
} rtMemQueueShareAttr_t;

typedef struct tagMemQueueBuffInfo {
    void *addr;
    size_t len;
} rtMemQueueBuffInfo;

typedef struct tagMemQueueBuff {
    void *contextAddr;
    size_t contextLen;
    rtMemQueueBuffInfo *buffInfo;
    uint32_t buffCount;
} rtMemQueueBuff_t;


typedef enum tagMemQueueQueryCmd {
    RT_MQ_QUERY_QUE_ATTR_OF_CUR_PROC = 0, // input is qid(4bytes), output is rtMemQueueShareAttr_t
    RT_MQ_QUERY_QUES_OF_CUR_PROC = 1,
    RT_MQ_QUERY_CMD_MAX = 2
} rtMemQueueQueryCmd_t;

#define RT_MQ_EVENT_QS_MSG 27 // same as driver's

#define RT_MQ_SCHED_PRIORITY_LEVEL0 0 // same as driver's
#define RT_MQ_SCHED_PRIORITY_LEVEL1 1
#define RT_MQ_SCHED_PRIORITY_LEVEL2 2
#define RT_MQ_SCHED_PRIORITY_LEVEL3 3
#define RT_MQ_SCHED_PRIORITY_LEVEL4 4
#define RT_MQ_SCHED_PRIORITY_LEVEL5 5
#define RT_MQ_SCHED_PRIORITY_LEVEL6 6
#define RT_MQ_SCHED_PRIORITY_LEVEL7 7

/* Events can be released between different systems. This parameter specifies the destination type of events
   to be released. The destination type is defined based on the CPU type of the destination system. */
#define RT_MQ_DST_ENGINE_ACPU_DEVICE 0            // device AICPU, same as driver's
#define RT_MQ_DST_ENGINE_ACPU_HOST 1              // Host AICPU
#define RT_MQ_DST_ENGINE_CCPU_DEVICE 2           // device CtrlCPU
#define RT_MQ_DST_ENGINE_CCPU_HOST 3             // Host CtrlCPU
#define RT_MQ_DST_ENGINE_DCPU_DEVICE 4          // device DataCPU
#define RT_MQ_DST_ENGINE_TS_CPU 5                 // device TS CPU
#define RT_MQ_DST_ENGINE_DVPP_CPU 6               // device DVPP CPU

#define RT_MQ_SCHED_EVENT_QS_MSG 25 // same as driver's EVENT_QS_MSG

/* When the destination engine is AICPU, select a policy.
   ONLY: The command is executed only on the local AICPU.
   FIRST: The local AICPU is preferentially executed. If the local AICPU is busy, the remote AICPU can be used. */
#define RT_SCHEDULE_POLICY_ONLY 0 // same as driver's schedule_policy
#define RT_SCHEDULE_POLICY_FIRST 1 // same as driver's schedule_policy


typedef struct tagEschedEventSummary {
    int32_t pid; // dst PID
    uint32_t grpId;
    int32_t eventId; // only RT_MQ_SCHED_EVENT_QS_MSG is supported
    uint32_t subeventId;
    uint32_t msgLen;
    char *msg;
    uint32_t dstEngine; // dst system cpu type
    int32_t policy; // RT_SCHEDULE_POLICY_ONLY or RT_SCHEDULE_POLICY_FIRST
} rtEschedEventSummary_t;

typedef struct tagEschedEventReply {
    char *buf;
    uint32_t bufLen;
    uint32_t replyLen; // output, ack msg len, same with msgLen in halEschedAckEvent
} rtEschedEventReply_t;

#define RT_DEV_PROCESS_CP1 0
#define RT_DEV_PROCESS_CP2 1
#define RT_DEV_PROCESS_DEV_ONLY 2
#define RT_DEV_PROCESS_QS 3
#define RT_DEV_PROCESS_SIGN_LENGTH 49

typedef struct tagBindHostpidInfo {
    int32_t hostPid;
    uint32_t vfid;
    uint32_t chipId;
    int32_t mode; // online:0, offline:1
    int32_t cpType; // type of custom-process, see RT_DEV_PROCESS_XXX
    uint32_t len; // lenth of sign
    char sign[RT_DEV_PROCESS_SIGN_LENGTH]; // sign of hostpid
} rtBindHostpidInfo_t;

#define RT_MEM_BUFF_MAX_CFG_NUM 64

typedef struct {
    uint32_t cfgId;    // cfg id, start from 0
    uint32_t totalSize;  // one zone total size
    uint32_t blkSize;  // blk size, 2^n (0, 2M]
    uint32_t maxBufSize; // max size can alloc from zone
    uint32_t pageType;  // page type, small page / huge page
    int32_t elasticEnable; // elastic enable
    int32_t elasticRate;
    int32_t elasticRateMax;
    int32_t elasticHighLevel;
    int32_t elasticLowLevel;
} rtMemZoneCfg_t;

typedef struct {
    rtMemZoneCfg_t cfg[RT_MEM_BUFF_MAX_CFG_NUM];
}rtMemBuffCfg_t;

typedef void *rtMbufPtr_t;

/**
 * @ingroup rt_mem_queue
 * @brief init queue schedule
 * @param [in] device   the logical device id
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueueInitQS(int32_t device);

/**
 * @ingroup rt_mem_queue
 * @brief create mbuf queue
 * @param [in] device   the logical device id
 * @param [in] rtMemQueueAttr   attribute of queue
 * @param [out] qid  queue id
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueueCreate(int32_t device, const rtMemQueueAttr_t *queueAttr, uint32_t *qid);

/**
 * @ingroup rt_mem_queue
 * @brief destroy mbuf queue
 * @param [in] device   the logical device id
 * @param [in] qid  queue id
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueueDestroy(int32_t device, uint32_t qid);

/**
 * @ingroup rt_mem_queue
 * @brief destroy mbuf queue init
 * @param [in] device   the logical device id
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueueInit(int32_t device);

/**
 * @ingroup rt_mem_queue
 * @brief enqueu mbuf
 * @param [in] device   the logical device id
 * @param [in] qid  queue id
 * @param [in] mbuf   enqueue mbuf
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueueEnQueue(int32_t device, uint32_t qid, void *mbuf);


/**
 * @ingroup rt_mem_queue
 * @brief enqueu mbuf
 * @param [in] device   the logical device id
 * @param [in] qid  queue id
 * @param [out] mbuf   dequeue mbuf
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf);

/**
 * @ingroup rt_mem_queue
 * @brief enqueu peek
 * @param [in] device   the logical device id
 * @param [in] qid  queue id
 * @param [out] bufLen   length of mbuf in queue
 * @param [in] timeout  peek timeout  (ms), -1: wait all the time until peeking success
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueuePeek(int32_t device, uint32_t qid, size_t *bufLen, int32_t timeout);

/**
 * @ingroup rt_mem_queue
 * @brief enqueu  buff
 * @param [in] device   the logical device id
 * @param [in] qid  queue id
 * @param [in] inBuf   enqueue buff
 * @param [in] timeout  enqueue timeout  (ms), -1: wait all the time until enqueue success
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueueEnQueueBuff(int32_t device, uint32_t qid, rtMemQueueBuff_t *inBuf, int32_t timeout);

/**
 * @ingroup rt_mem_queue
 * @brief enqueu  buff
 * @param [in] device   the logical device id
 * @param [in] qid  queue id
 * @param [out] outBuf   dequeue buff
 * @param [in] timeout  dequeue timeout  (ms), -1: wait all the time until dequeue success
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtMemQueueDeQueueBuff(int32_t device, uint32_t qid, rtMemQueueBuff_t *outBuf, int32_t timeout);


/**
* @ingroup rt_mem_queue
* @brief  query queue status
* @param [in] device: the logical device id
* @param [in] cmd: query cmd
* @param [in] inBuff: input buff
* @param [in] inLen: the length of input
* @param [in|out] outBuff: output buff
* @param [in|out] outLen: the length of output
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMemQueueQuery(int32_t device, rtMemQueueQueryCmd_t cmd, const void *inBuff, uint32_t inLen,
    void *outBuff, uint32_t *outLen);

/**
* @ingroup rt_mem_queue
* @brief  grant queue
* @param [in] device: logic devid
* @param [in] qid: queue id
* @param [in] pid: pid
* @param [in] attr: queue share attr
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMemQueueGrant(int32_t device, uint32_t qid, int32_t pid, rtMemQueueShareAttr_t *attr);

/**
* @ingroup rt_mem_queue
* @brief  attach queue
* @param [in] device: logic devid
* @param [in] qid: queue id
* @param [in] timeOut: timeOut
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMemQueueAttach(int32_t device, uint32_t qid, int32_t timeOut);

/**
* @ingroup rt_mem_queue
* @brief  Commit the event to a specific process
* @param [in] device: logic devid
* @param [in] event: event summary info
* @param [out] ack: event reply info
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtEschedSubmitEventSync(int32_t device, rtEschedEventSummary_t *event,
                                          rtEschedEventReply_t *ack);

/**
* @ingroup rt_mem_queue
* @brief  query device proccess id
* @param [in] info: see struct rtBindHostpidInfo_t
* @param [out] devPid: device proccess id
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtQueryDevPid(rtBindHostpidInfo_t *info, int32_t *devPid);

/**
* @ingroup rt_mem_queue
* @brief device buff init
* @param [in] cfg, init cfg
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMbufInit(rtMemBuffCfg_t *cfg);

/**
* @ingroup rt_mem_queue
* @brief alloc buff
* @param [out] buff: buff addr alloced
* @param [in]  size: The amount of memory space requested
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size);

/**
* @ingroup rt_mem_queue
* @brief free buff
* @param [in] buff: buff addr to be freed
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMbufFree(rtMbufPtr_t mbuf);

/**
* @ingroup rt_mem_queue
* @brief get Data addr of Mbuf
* @param [in] mbuf: Mbuf addr
* @param [out] buf: Mbuf data addr
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **buf);

/**
* @ingroup rt_mem_queue
* @brief get total Buffer size of Mbuf
* @param [in] mbuf: Mbuf addr
* @param [out] totalSize: total buffer size of Mbuf
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *totalSize);

/**
* @ingroup rt_mem_queue
* @brief Get the address and length of its user_data from the specified Mbuf
* @param [in] mbuf: Mbuf addr
* @param [out] priv: address of its user_data
* @param [out]  size: length of its user_data
* @return RT_ERROR_NONE for ok
*/
RTS_API rtError_t rtMbufGetPrivInfo (rtMbufPtr_t mbuf,  void **priv, uint64_t *size);

// mem group
typedef struct {
    uint64_t maxMemSize; // max buf size in grp, in KB. = 0 means no limit
} rtMemGrpConfig_t;

typedef struct {
    uint32_t admin : 1;     // admin permission, can add other proc to grp
    uint32_t read : 1;     // read only permission
    uint32_t write : 1;    // read and write permission
    uint32_t alloc : 1;    // alloc permission (have read and write permission)
    uint32_t rsv : 28;
} rtMemGrpShareAttr_t;

#define RT_MEM_GRP_QUERY_GROUPS_OF_PROCESS 1  // query process all grp

typedef struct {
    int32_t pid;
} rtMemGrpQueryByProc_t; // cmd: GRP_QUERY_GROUPS_OF_PROCESS

typedef union {
    rtMemGrpQueryByProc_t grpQueryByProc; // cmd: GRP_QUERY_GROUPS_OF_PROCESS
} rtMemGrpQueryInput_t;

#define RT_MEM_GRP_NAME_LEN 32  // it must be same as driver define BUFF_GRP_NAME_LEN

typedef struct {
    char groupName[RT_MEM_GRP_NAME_LEN];  // group name
    rtMemGrpShareAttr_t attr; // process in group attribute
} rtMemGrpOfProc_t; // cmd: GRP_QUERY_GROUPS_OF_PROCESS

typedef struct {
    rtMemGrpOfProc_t *groupsOfProc; // cmd: GRP_QUERY_GROUPS_OF_PROCESS
    size_t maxNum; // max number of result
    size_t resultNum; // if the number of results exceeds 'maxNum', only 'maxNum' results are filled in buffer
} rtMemGrpQueryOutput_t;

/**
* @ingroup rt_mem_queue
* @brief create mem group
* @attention null
* @param [in] name, group name
* @param [in] cfg, group cfg
* @return   0 for success, others for fail
*/
RTS_API rtError_t rtMemGrpCreate(const char *name, const rtMemGrpConfig_t *cfg);

/**
* @ingroup rt_mem_queue
* @brief add process to group
* @param [in] name, group name
* @param [in] pid, process id
* @param [in] attr, process permission in group
* @return   0 for success, others for fail
*/
RTS_API rtError_t rtMemGrpAddProc(const char *name, int32_t pid, const rtMemGrpShareAttr_t *attr);

/**
* @ingroup rt_mem_queue
* @brief attach proccess to check permission in group
* @param [in] name, group name
* @param [in] timeout, time out ms
* @return   0 for success, others for fail
*/
RTS_API rtError_t rtMemGrpAttach(const char *name, int32_t timeout);

/**
* @ingroup rt_mem_queue
* @brief buff group query
* @param [in] cmd, cmd type
* @param [in] input, query input
* @param [in|out] output, query output
* @return   0 for success, others for fail
*/
RTS_API rtError_t rtMemGrpQuery(int32_t cmd, const rtMemGrpQueryInput_t *input, rtMemGrpQueryOutput_t *output);

#if defined(__cplusplus)
}
#endif
#endif // CCE_RUNTIME_RT_MEM_QUEUE_H
