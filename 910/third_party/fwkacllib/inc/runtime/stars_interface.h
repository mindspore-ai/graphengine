/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: stars interface
 */

#ifndef CCE_RUNTIME_STARS_INTERFACE_H
#define CCE_RUNTIME_STARS_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef RTS_API
#ifdef RTS_DLL_EXPORT
#define RTS_API __declspec(dllexport)
#else
#define RTS_API
#endif
#endif

#define STARS_EVENT_RECORD 0x4
#define STARS_EVENT_WAIT 0x5
#define STARS_SCH_WR_VAL 0x8
#define STARS_SCH_RDMA 0x10
#define STARS_SCH_SDMA 0xb

#define STARS_DEVICE_NAME_LENGTH_MAX    32
#define CHIP_NUM                        4
#define CPU_DIE_NUM                     2
#define STARS_DEV_NUM                   (CHIP_NUM * CPU_DIE_NUM)
#define STARS_WR_VAL_LEN_MAX            8

/* MPI 下发的数据结构 */
typedef struct stars_trans_parm {
    unsigned int opcode; /* event record/event wait/rdma/sdma opcode */
    unsigned int wr_cqe_flag;
    void *trans_parms;
    uint32_t parms_len; /* trans_parms结构的长度 */
} stars_trans_parm_t;

/* SDMA SQE的数据结构 */
typedef struct sdma_trans_parm {
    unsigned int length;
    unsigned short s_substreamid;
    unsigned short d_substreamid;
    unsigned long long s_addr;
    unsigned long long d_addr;
    unsigned int s_stride_len;
    unsigned int d_stride_len;
    unsigned int stride_num;
    unsigned char opcode; /* commom addr transfer/memset/HBM cache opcode */
} sdma_trans_parm_t;

typedef enum {
    STARS_EVENT_TYPE_INNODE,  /* the event_id is used in a node */
    STARS_EVENT_TYPE_OUTNODE, /* the event_id is used between nodes */
} STARS_EVENT_TYPE;

typedef struct event_id_type {
    unsigned int event_id;          /* first event index of all obtained eventids */
    uint64_t offset;                /* phys addr offset */
} event_id_type_t;

/* EVENT SQE的数据结构 */
typedef struct event_trans_parm {
    STARS_EVENT_TYPE    type; /* in-node：use id; between nodes：use event addr */
    unsigned int        event_id;
    uint64_t            event_addr;
} event_trans_parm_t;

typedef enum {
    STARS_QP_CMD_TYPE_SQ_DB,
    STARS_QP_CMD_TYPE_RQ_DB,
    STARS_QP_CMD_TYPE_SRQ_DB,
    STARS_QP_CMD_TYPE_CQ_DB,
    STARS_QP_CMD_TYPE_CQ_DB_NTF,
} STARS_QP_CMD_TYPE;

typedef struct rdma_trans_parm {
    unsigned int            task_count;     /* number of wr */
    unsigned int            qp_num;         /* qp->qp_num */
    unsigned int            db_value;
    STARS_QP_CMD_TYPE       cmd;            /* only STARS_QP_CMD_TYPE_SQ_DB is supported */
    unsigned short          streamid;       /* sva is not support, streamid can be ignored */
    unsigned short          substreamid;    /* sva is not support, substreamid can be ignored */
    unsigned char           hac_functionId;
    unsigned char           sl;             /* service level */
    unsigned long           rdma_addr;
} rdma_trans_parm_t;

typedef struct rdma_output_status {
    uint64_t wr_id;
    unsigned int byte_len;
    unsigned int imm_data;
    unsigned char status;
    unsigned short slid;
    unsigned int qp_num;
    unsigned int cmpl_cnt;
} rdma_output_status_t;
typedef struct wr_val_trans_parm {
    unsigned int        va_enable;                      /* destination address type, VA:1 or PA:0 */
    uintptr_t           d_addr;                         /* destination address */
    unsigned int        length;                         /* data length to be written, in bytes, max 32 */
    unsigned int        event_id;                       /* if used for event flag write, use this instead of d_addr */
    unsigned int        wr_data[STARS_WR_VAL_LEN_MAX];  /* input data */
} wr_val_trans_parm_t;

typedef struct stars_cqe_output {
    bool                    is_rdma_task;
    rdma_output_status_t    rdma_cqe;
} stars_cqe_output_t;

typedef struct stars_wait_output {
    unsigned int        out_pos;
    unsigned int        out_num; /* 数组长度 */
    stars_cqe_output_t  *cqe_output;
} stars_wait_output_t;

/* 用户态获取Stars设备信息的数据结构 */
typedef struct device_info {
    char                name[STARS_DEVICE_NAME_LENGTH_MAX];
    unsigned int        chip_index;
    unsigned int        die_index;
    unsigned int        io_mode;   /* rdma die */
} device_info_t;

typedef struct stars_info {
    struct device_info  dev_info[STARS_DEV_NUM];
    unsigned int        dev_num;
} stars_info_t;

/**
 * @ingroup rt_stars
 * @brief stars设备初始化
 * @param [in]  dev_id            设备id
 * @return ACL_RT_SUCCESS         成功
 * @return 其他错误码              失败
 */
RTS_API int stars_dev_init(int dev_id);

/**
 * @ingroup rt_stars
 * @brief stars设备去初始化
 * @param [in]  dev_id            设备去初始化
 * @return ACL_RT_SUCCESS         成功
 * @return 其他错误码              失败
 */
RTS_API int stars_dev_deinit(int dev_id);

/**
 * @ingroup rt_stars
 * @brief 获取记录的stars设备信息
 * @param [in]  info              stars设备信息的指针
 * @param [out] info              获取到的stars设备信息
 * @return ACL_RT_SUCCESS         成功
 * @return 其他错误码              失败
 */
RTS_API int stars_get_info(stars_info_t *info);

/**
 * @ingroup rt_stars
 * @brief 分配一个stars rtsq通道
 * @param [in]  dev_id              设备id
 * @param [in]  pool_id             die id
 * @param [out]  stars handle
 * @return stars handle             stars句柄
 */
RTS_API void *stars_get_handle(int dev_id, unsigned int pool_id);

/**
 * @ingroup rt_stars
 * @brief 释放通道
 * @param [in]  phandle           stars句柄
 * @return ACL_RT_SUCCESS         成功
 * @return 其他错误码              失败
 */
RTS_API int stars_release_handle(void *phandle);

/**
 * @ingroup rt_stars
 * @brief stars下发各类任务（SDMA/RDMA/EVENT）
 * @param [in]  phandle              stars句柄
 * @param [in]  trans_parm           sqe数据指针
 * @param [in]  task_count           sqe的数量
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_send_task(void *phandle, stars_trans_parm_t *trans_parm, unsigned int task_count);

/**
 * @ingroup rt_stars
 * @brief 等待stars通道发送完成
 * @param [in]  phandle              stars句柄
 * @param [in]  task_count           要解析的cqe的数量（send接口中标注wr_cqe_flag为1的task的数量）
 * @param [in]  output               根据不同sch的需要，要返回的数据(由调用者传入空间地址，实现者传入数据)
 * @param [out]  task_cmpl_count     本次wait到的cqe的数量
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_wait_cqe(void *phandle, unsigned int task_count, unsigned int *task_cmpl_count, void *output);

/**
 * @ingroup rt_ucx
 * @brief 等待stars通道发送完成
 * @param [in]  dev_id               设备id
 * @param [in]  count                表示需要申请的event ID的个数, 非连续
 * @param [in]  event_id             用户传入的event_id结构体指针
 * @param [in]  pool_id              表示需求申请的event id所在pool id
 * @param [out]  event_id            获取到的event_id的信息
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_get_event_id(int dev_id, unsigned int count, event_id_type_t *event_id, unsigned int pool_id);

/**
 * @ingroup rt_ucx
 * @brief 释放event id
 * @param [in]  dev_id               设备id
 * @param [in]  count                释放event id的个数
 * @param [in]  event_id             用户传入的event_id结构体指针
 * @param [in]  event_id             用户获得的event_id结构体指针
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_release_event_id(int dev_id, unsigned int count, event_id_type_t *event_id);

/**
 * @ingroup rt_stars
 * @brief 获取event table的物理地址
 * @param [in]  dev_id               设备id
 * @param [out] addr                 event table物理地址
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_get_event_table_addr(int dev_id, uint64_t *addr);

/**
 * @ingroup rt_stars
 * @brief 获取pasid
 * @param [in]  dev_id               设备id
 * @param [out] pasid                返回的pasid
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_get_pasid(int dev_id, unsigned int *pasid);

/**
 * @ingroup rt_stars
 * @brief 获取rdma加速器往io adapter 写cqe的地址,此地址为4个POE io寄存器的首地址，每个POE地址偏移为64Byte
 * @param [in]  dev_id               设备id
 * @param [in]  die_id               die id
 * @param [out] addr                 cqe地址
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_get_rdma_cq_addr(int dev_id, unsigned int die_id, uint64_t *addr);

/**
 * @ingroup rt_stars
 * @brief 为用户进程虚拟地址PIN页表
 * @param [in]  dev_id               设备id
 * @param [in]  vma                  虚拟地址
 * @param [in]  size                 PIN内存大小
 * @param [out] cookie               返回的cookie指针
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_pin_umem(int dev_id, void *vma, unsigned int size, uint64_t *cookie);

/**
 * @ingroup rt_stars
 * @brief 为用户进程虚拟地址PIN页表
 * @param [in] dev_id                设备id
 * @param [in] cookie                pin的cookie
 * @return ACL_RT_SUCCESS            成功
 * @return 其他错误码                 失败
 */
RTS_API int stars_unpin_umem(int dev_id, uint64_t cookie);

#if defined(__cplusplus)
}
#endif
#endif  // CCE_RUNTIME_STARS_INTERFACE_H
