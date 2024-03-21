/**
* @file acl_dump.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_DUMP_H_
#define INC_EXTERNAL_ACL_DUMP_H_

#include <stdint.h>

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define ACL_DUMP_API __declspec(dllexport)
#else
#define ACL_DUMP_API __attribute__((visibility("default")))
#endif

#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_DUMP_MAX_FILE_PATH_LENGTH    4096
typedef struct acldumpChunk  {
    char       fileName[ACL_DUMP_MAX_FILE_PATH_LENGTH];   // file name, absolute path
    uint32_t   bufLen;                           // dataBuf length
    uint32_t   isLastChunk;                      // is last chunk. 0: not 1: yes
    int64_t    offset;                           // Offset in file. -1: append write
    int32_t    flag;                             // flag
    uint8_t    dataBuf[0];                       // data buffer
} acldumpChunk;

ACL_DUMP_API aclError acldumpRegCallback(int32_t (* const messageCallback)(const acldumpChunk *, int32_t),
    int32_t flag);
ACL_DUMP_API void acldumpUnregCallback();

#ifdef __cplusplus
}
#endif

#endif
