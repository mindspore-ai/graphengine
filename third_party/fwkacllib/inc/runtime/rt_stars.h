/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef __CCE_RUNTIME_STARS_H
#define __CCE_RUNTIME_STARS_H

#include "base.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

/**
 * @ingroup rt_stars
 * @brief launch stars task.
 * used for send star sqe directly.
 * @param [in] taskSqe     stars task sqe
 * @param [in] sqeLen      stars task sqe length
 * @param [in] stream      associated stream
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtStarsTaskLaunch(const void *taskSqe, uint32_t sqeLen, rtStream_t stream);

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif // __CCE_RUNTIME_STARS_H
