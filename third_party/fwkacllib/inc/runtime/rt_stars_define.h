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

#ifndef CCE_RUNTIME_RT_STARS_DEFINE_H
#define CCE_RUNTIME_RT_STARS_DEFINE_H

#include "base.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#pragma pack(push)
#pragma pack (1)

typedef struct tagStarsSqeHeader {
    uint8_t type : 6;
    uint8_t l1Lock : 1;
    uint8_t l1Unlock : 1;

    uint8_t ie : 2;
    uint8_t preP : 2;
    uint8_t postP : 2;
    uint8_t wrCqe : 1;
    uint8_t reserved : 1;

    uint16_t blockDim;

    uint16_t rtStreamId;
    uint16_t taskId;
} rtStarsSqeHeader_t;

// ffts+ type
typedef enum tagFftsPlusType {
    RT_FFTS_PLUS_TYPE_RES1 = 2,   // Reserved
    RT_FFTS_PLUS_TYPE_RES2 = 3,   // Reserved
    RT_FFTS_PLUS_TYPE = 4,        // FFTS+ mode
} rtFftsPlusType_t;

// ffts+ sqe
typedef struct tagFftsPlusSqe {
    // 0-7 bytes
    rtStarsSqeHeader_t sqeHeader;
    // 8-11 bytes
    uint16_t fftsType : 3;
    uint16_t reserved1 : 9;
    uint16_t wrrRatio : 4;
    uint16_t reserved2;
    // 12-15 bytes
    uint16_t sqeIndex;
    uint8_t  kernelCredit;
    uint8_t  reserved4;
    // 16-23 bytes
    uint32_t stackPhyBaseL;
    uint32_t stackPhyBaseH;
    // 24-31 bytes
    uint16_t  totalContextNum;
    uint16_t  readyContextNum;
    uint16_t  preloadContextNum;
    uint16_t  reserved5;
    // 32-35 bytes
    uint16_t  reserved6;
    uint16_t  prefetchOstNum : 5;
    uint16_t  reserved9 : 3;
    uint16_t  cmaintOstNum : 5;
    uint16_t  reserved10 : 3;
    // 36-39 bytes
    uint16_t  aicPrefetchLower : 5;
    uint16_t  reserved11 : 3;
    uint16_t  aicPrefetchUpper : 5;
    uint16_t  reserved12 : 3;
    uint16_t  aivPrefetchLower : 5;
    uint16_t  Reserved13 : 3;
    uint16_t  aivPrefetchUpper : 5;
    uint16_t  Reserved14 : 3;
    // 40-47 bytes
    uint32_t contextAddressBaseL;
    uint32_t contextAddressBaseH : 17;
    uint32_t reserved15 : 15;
    // 48-63 bytes
    uint32_t reserved16[4];
} rtFftsPlusSqe_t;

#pragma pack(pop)

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif // CCE_RUNTIME_RT_STARS_DEFINE_H