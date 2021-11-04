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

#ifndef CCE_RUNTIME_RT_FFTS_PLUS_H
#define CCE_RUNTIME_RT_FFTS_PLUS_H

#include "base.h"
#include "rt_ffts_plus_define.h"
#include "rt_stars_define.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#pragma pack(push)
#pragma pack (1)

typedef struct tagFftsPlusTaskInfo {
    const rtFftsPlusSqe_t *fftsPlusSqe;
    const void *descBuf;           // include total context
    size_t      descBufLen;        // the length of descBuf
} rtFftsPlusTaskInfo_t;

#pragma pack(pop)

RTS_API rtError_t rtGetAddrAndPrefCntWithHandle(void *handle, const void *devFunc, void **addr, uint32_t *prefetchCnt);

RTS_API rtError_t rtFftsPlusTaskLaunch(rtFftsPlusTaskInfo_t *fftsPlusTaskInfo, rtStream_t stream);

RTS_API rtError_t rtFftsPlusTaskLaunchWithFlag(rtFftsPlusTaskInfo_t *fftsPlusTaskInfo, rtStream_t stream,
                                               uint32_t flag);

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif // CCE_RUNTIME_RT_FFTS_PLUS_H