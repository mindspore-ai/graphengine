/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: event.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_EVENT_H
#define CCE_RUNTIME_EVENT_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define RT_IPCINT_MSGLEN_MAX     (0x8U)

typedef enum rtEventWaitStatus {
    EVENT_STATUS_COMPLETE = 0,
    EVENT_STATUS_NOT_READY = 1,
    EVENT_STATUS_MAX = 2,
} rtEventWaitStatus_t;

typedef enum rtEventStatus {
    RT_EVENT_INIT = 0,
    RT_EVENT_RECORDED = 1,
} rtEventStatus_t;

typedef struct tagIpcIntNoticeInfo {
    uint32_t ntcPid;
    uint16_t ntcGrpId;
    uint16_t ntcTid;
    uint16_t ntcSubEventId;
    uint8_t  ntcEventId;
    uint8_t  msgLen;
    uint8_t  msg[RT_IPCINT_MSGLEN_MAX];
    uint16_t phyDevId;
} rtIpcIntNoticeInfo_t;

typedef enum {
    RT_STREAM_ID = 0,
    RT_EVENT_ID,
    RT_MODEL_ID,
    RT_NOTIFY_ID,
    RT_CMO_ID,
    RT_CNT_NOTIFY_ID,    /* add start ascend910d */
    RT_INVALID_ID,
} rtIdType_t;

/**
 * @ingroup event_flags
 * @brief event op bit flags
 */
#define RT_EVENT_DDSYNC_NS    0x01U
#define RT_EVENT_STREAM_MARK  0x02U
#define RT_EVENT_DDSYNC       0x04U
#define RT_EVENT_TIME_LINE    0x08U
#define RT_EVENT_MC2          0x10U             // RT_EVENT_MC2 does not support OR with other flags

#define RT_EVENT_DEFAULT (RT_EVENT_DDSYNC | RT_EVENT_TIME_LINE | RT_EVENT_STREAM_MARK)
#define RT_EVENT_WITH_FLAG (RT_EVENT_DDSYNC_NS)

#define RT_NOTIFY_FLAG_DEFAULT (0x00U)
#define RT_NOTIFY_FLAG_DOWNLOAD_TO_DEV (0x01U)  // RT_NOTIFY_FLAG_DOWNLOAD_TO_DEV does not support OR with other flags
#define RT_NOTIFY_FLAG_SHR_ID_SHADOW (0x1U << 6)
#define RT_NOTIFY_FLAG_MAX \
    (RT_NOTIFY_FLAG_DOWNLOAD_TO_DEV |RT_NOTIFY_FLAG_SHR_ID_SHADOW)
/**
 * @ingroup notify_flags
 * @brief notify op bit flags
 */
#define RT_NOTIFY_DEFAULT   0x00U
#define RT_NOTIFY_MC2       0x01U               // RT_NOTIFY_MC2 does not support OR with other flags
#define RT_DMS_MAX_EVENT_NAME_LENGTH 256
#define RT_DMS_MAX_EVENT_DATA_LENGTH 32
#define RT_DMS_MAX_EVENT_RESV_LENGTH 32
#define RT_DSM_EVENT_FILTER_FLAG_PID (1UL << 3)
#define RT_MAX_RECORD_PA_NUM_PER_DEV    20U

typedef struct {
    uint64_t ptr;
    uint64_t len;
} rtMemRepairAddr;

typedef struct {
    uint32_t devid;
    uint32_t count;
    rtMemRepairAddr repairAddr[RT_MAX_RECORD_PA_NUM_PER_DEV];
} rtMemUceInfo;

typedef struct tagDmsEventFilter {
    uint64_t filterFlag;
    uint32_t eventId;
    unsigned char severity;
    unsigned char nodeType;
    unsigned char resv[RT_DMS_MAX_EVENT_RESV_LENGTH]; /**< reserve 32byte */
} rtDmsEventFilter;

typedef struct tagDmsFaultEvent {
    uint64_t alarmRaisedTime;
    uint32_t eventId;
    int32_t tgid;
    int32_t eventSerialNum;
    int32_t notifySerialNum;
    uint16_t deviceId;
    uint16_t nodeType;
    uint16_t subNodeType;
    unsigned char nodeId;
    unsigned char subNodeId;
    unsigned char severity;
    unsigned char assertion;
    char eventName[RT_DMS_MAX_EVENT_NAME_LENGTH];
    char additionalInfo[RT_DMS_MAX_EVENT_DATA_LENGTH];
    unsigned char osId;
    unsigned char resv[RT_DMS_MAX_EVENT_RESV_LENGTH]; /* reserve 32byte */
} rtDmsFaultEvent;

typedef struct tagNotifyPhyInfo {
    uint32_t phyId;  /* phy id */
    uint32_t tsId;   /* ts id */
    uint32_t idType; /* SHR_ID_NOTIFY_TYPE */
    uint32_t shrId;  /* notify id */
    uint32_t flag;   /* RT_NOTIFY_FLAG_SHR_ID_SHADOW for remote id or shadow node */
    uint32_t rsv[3];
} rtNotifyPhyInfo;

/**
 * @ingroup dvrt_event
 * @brief create event instance
 * @param [in|out] event   created event
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEventCreate(rtEvent_t *evt);

/**
 * @ingroup dvrt_event
 * @brief create event instance with flag
 * @param [in|out] event   created event  flag event op flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEventCreateWithFlag(rtEvent_t *evt, uint32_t flag);

/**
 * @ingroup dvrt_event
 * @brief create event instance with flag for single mode
 * @param [in|out] event  created event  flag event op flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEventCreateExWithFlag(rtEvent_t *evt, uint32_t flag);

/**
 * @ingroup dvrt_event
 * @brief destroy event instance
 * @param [in] evt   event to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEventDestroy(rtEvent_t evt);

/**
 * @ingroup dvrt_event
 * @brief synchronize destroy event instance
 * @param [in] evt   event to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEventDestroySync(rtEvent_t evt);

/**
 * @ingroup dvrt_event
 * @brief get event id
 * @param [in] evt event to be get
 * @param [in|out] event_id   event_id id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetEventID(rtEvent_t evt, uint32_t *evtId);

/**
 * @ingroup dvrt_event
 * @brief event record
 * @param [int] event   event to record
 * @param [int] stm   stream handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEventRecord(rtEvent_t evt, rtStream_t stm);

/**
 * @ingroup dvrt_event
 * @brief event reset
 * @param [int] event   event to reset
 * @param [int] stm   stream handle
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtEventReset(rtEvent_t evt, rtStream_t stm);

/**
 * @ingroup dvrt_event
 * @brief wait event to be complete
 * @param [in] evt   event to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEventSynchronize(rtEvent_t evt);

/**
 * @ingroup dvrt_event
 * @brief wait event to be complete
 * @param [in] evt   event to wait
 * @param [in] timeout event wait timeout
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_EVENT_SYNC_TIMEOUT for timeout
 */
RTS_API rtError_t rtEventSynchronizeWithTimeout(rtEvent_t evt, const int32_t timeout);

/**
 * @ingroup dvrt_event
 * @brief Queries an event's status
 * @param [in] evt   event to query
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_EVENT_NOT_COMPLETE for not complete
 */
RTS_API rtError_t rtEventQuery(rtEvent_t evt);

/**
 * @ingroup dvrt_event
 * @brief Queries an event's wait status
 * @param [in] evt   event to query
 * @param [in out] EVENT_WAIT_STATUS status
 * @return EVENT_STATUS_COMPLETE for complete
 * @return EVENT_STATUS_NOT_READY for not complete
 */
RTS_API rtError_t rtEventQueryWaitStatus(rtEvent_t evt, rtEventWaitStatus_t *status);

/**
 * @ingroup dvrt_event
 * @brief Queries an event's status
 * @param [in] evt   event to query
 * @param [in out] rtEventStatus_t status
 * @return RT_EVENT_RECORDED  for recorded
 * @return RT_EVENT_INIT for not recorded
 */
RTS_API rtError_t rtEventQueryStatus(rtEvent_t evt, rtEventStatus_t *status);

/**
 * @ingroup dvrt_event
 * @brief computes the elapsed time between events.
 * @param [in] timeInterval   time between start and end in ms
 * @param [in] startEvent  starting event
 * @param [in] endEvent  ending event
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtEventElapsedTime(float32_t *timeInterval, rtEvent_t startEvent, rtEvent_t endEvent);

/**
 * @ingroup dvrt_event
 * @brief get the elapsed time from a event after event recorded.
 * @param [in] timeStamp   time in ms
 * @param [in] evt  event handle
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtEventGetTimeStamp(uint64_t *timeStamp, rtEvent_t evt);

/**
 * @ingroup dvrt_event
 * @brief name an event
 * @param [in] evt  event to be named
 * @param [in] name  identification name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input of event, name
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtNameEvent(rtEvent_t evt, const char_t *name);

/**
 * @ingroup
 * @brief get fault event .
 * @param [in] deviceId device id
 * @param [in] filter filter condition:PID
 * @param [in] len output length
 * @param [out] dmsEvent return dms event struct array
 * @param [out] eventCount return event count
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetFaultEvent(const int32_t deviceId, rtDmsEventFilter *filter, rtDmsFaultEvent *dmsEvent,
    uint32_t len, uint32_t *eventCount);

/**
* @ingroup dvrt_mem
* @brief Get memUceInfo.
* @attention Only support ONLINE scene.
* @param [in] deviceId device id.
* @param [out] memUceInfo Returned memUceInfo.
* @return DRV_ERROR_NONE : success
* @return DV_ERROR_XXX : fail
*/
RTS_API rtError_t rtGetMemUceInfo(const uint32_t deviceId, rtMemUceInfo *memUceInfo);

/**
* @ingroup dvrt_mem
* @brief Repair Ucemem.
* @attention Only support ONLINE scene.
* @param [in] deviceId device id.
* @param [out] memUceInfo Returned memUceInfo.
* @return DRV_ERROR_NONE : success
* @return DV_ERROR_XXX : fail
*/
RTS_API rtError_t rtMemUceRepair(const uint32_t deviceId, rtMemUceInfo *memUceInfo);

/**
 * @ingroup dvrt_event
 * @brief Create a notify
 * @param [in] device_id  device id
 * @param [in|out] notify_   notify to be created
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNotifyCreate(int32_t deviceId, rtNotify_t *notify);


/**
 * @ingroup dvrt_event
 * @brief Create a notify
 * @param [in] device_id  device id
 * @param [in|out] notify_   notify to be created
 * @param [in] flag  flag notify flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNotifyCreateWithFlag(int32_t deviceId, rtNotify_t *notify, uint32_t flag);

/**
 * @ingroup dvrt_event
 * @brief Destroy a notify
 * @param [in] notify_   notify to be destroyed
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtNotifyDestroy(rtNotify_t notify);

/**
 * @ingroup dvrt_event
 * @brief Record a notify
 * @param [in] notify_ notify to be recorded
 * @param [in] stream_  input stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_STREAM_CONTEXT for stream is not in current ctx
 */
RTS_API rtError_t rtNotifyRecord(rtNotify_t notify, rtStream_t stm);

/**
 * @ingroup dvrt_event
 * @brief Reset a notify
 * @param [in] notify_   notify to be destroyed
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNotifyReset(rtNotify_t notify);

/**
 * @ingroup dvrt_event
 * @brief Resource clean
 * @param [in] devId   deviceId
 * @param [in] type   typeId
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtResourceClean(int32_t devId, rtIdType_t type);

/**
 * @ingroup dvrt_event
 * @brief Wait for a notify
 * @param [in] notify   notify to be wait
 * @param [in] stm   input stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_STREAM_CONTEXT for stream is not in current ctx
 */
RTS_API rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stm);

/**
 * @ingroup dvrt_event
 * @brief Wait for a notify with time out
 * @param [in] notify notify to be wait
 * @param [in] stm  input stream
 * @param [in] timeOut  input timeOut
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_STREAM_CONTEXT for stream is not in current ctx
 */
RTS_API rtError_t rtNotifyWaitWithTimeOut(rtNotify_t notify, rtStream_t stm, uint32_t timeOut);

/**
 * @ingroup dvrt_event
 * @brief get notify id
 * @param [in] notify_ notify to be get
 * @param [in|out] notify_id   notify id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetNotifyID(rtNotify_t notify, uint32_t *notifyId);

/**
 * @ingroup dvrt_event
 * @brief Get notify phy info
 * @param [in] notify the created/opened notify
 * @param [out] phyDevId phy device id
 * @param [out] tsId ts id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNotifyGetPhyInfo(rtNotify_t notify, uint32_t *phyDevId, uint32_t *tsId);

/**
 * @ingroup dvrt_event
 * @brief Get notify phy and pod info
 * @param [in] notify the created/opened notify
 * @param [out] phyDevId phy device id
 * @param [out] tsId ts id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNotifyGetPhyInfoExt(rtNotify_t notify, rtNotifyPhyInfo *notifyInfo);

/**
 * @ingroup dvrt_event
 * @brief Set a notify to IPC notify
 * @param [in] notify   notify to be set to IPC notify
 * @param [in] name   identification name
 * @param [in] len   length of name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtIpcSetNotifyName(rtNotify_t notify, char_t *name, uint32_t len);

/**
 * @ingroup dvrt_event
 * @brief Open IPC notify
 * @param [out] notify the opened notify
 * @param [in] name   identification name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtIpcOpenNotify(rtNotify_t *notify, const char_t *name);

/**
 * @ingroup dvrt_event
 * @brief Open IPC notify
 * @param [out] notify the opened notify
 * @param [in] name   identification name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtIpcOpenNotifyWithFlag(rtNotify_t *notify, const char_t *name, uint32_t flag);

/**
 * @ingroup dvrt_event
 * @brief Get the physical address corresponding to notify
 * @param [in] notify notify to be queried
 * @param [in] devAddrOffset  device physical address offset
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtNotifyGetAddrOffset(rtNotify_t notify, uint64_t *devAddrOffset);

/**
 * @ingroup dvrt_event
 * @brief Ipc set notify pid
 * @param [in] name name to be queried
 * @param [in] pid  process id
 * @param [in] num  length of pid[]
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtSetIpcNotifyPid(const char_t *name, int32_t pid[], int32_t num);

/**
 * @ingroup dvrt_event
 * @brief Create an ipc interrupt notice task
 * @param [in] ipcIntNoticeInfo  ipcIntNotice info to be sent
 * @param [in] stm  stream handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtIpcIntNotice(const rtIpcIntNoticeInfo_t * const ipcIntNoticeInfo, rtStream_t stm);

#if defined(__cplusplus)
}
#endif

#endif  // CCE_RUNTIME_EVENT_H
