/**
 * @file cce/cce.h
 *
 * Copyright(C), 2017 - 2017, Huawei Tech. Co., Ltd. ALL RIGHTS RESERVED.
 *
 * @brief library api header file.
 *
 * @version 1.0
 *
 */
#ifndef __CCE_API_H__
#define __CCE_API_H__

#include "cce/cce.h"
namespace cce {
typedef enum tagCceCRedOp
{
    CCE_RED_OP_SUM        = 0,
    CCE_RED_OP_PROD       = 1,
    CCE_RED_OP_MAX        = 2,
    CCE_RED_OP_Min        = 3,
    CCE_RED_OP_RESERVED
}ccReduceOp_t;

#ifndef DAVINCI_LITE
ccStatus_t ccVectorReduce(const void *src1,
                          const void *src2,
                          uint64_t count,
                          ccDataType_t datatype,
                          ccReduceOp_t op,
                          rtStream_t streamId,
                          const void *dst );
#endif
};//cce

#endif /* __CCE_H__ */
