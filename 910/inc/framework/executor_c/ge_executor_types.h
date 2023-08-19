/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_C_TYPES_H_
#define INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_C_TYPES_H_
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "vector.h"
#if defined(__cplusplus)
extern "C" {
#endif
typedef uint32_t Status;
#define SUCCESS  0

typedef struct {
  void *modelData;
  uint64_t modelLen;
  int32_t priority;
} ModelData;

typedef struct {
  void *data;
  uint64_t length;
} DataBuffer;

typedef struct {
  DataBuffer *dataBuffer;
} DataBlob;

typedef struct {
  Vector blobs; // type : DataBlob
  uint64_t* io_addr;
  uint32_t ioa_size;
} DataSet;

typedef DataSet InputData;
typedef DataSet OutputData;

typedef struct {
  void *stream;
  void *workPtr;
  size_t workSize;
  size_t mpamId;
  size_t aicQos;
  size_t aicOst;
  size_t mecTimeThreshHold;
} ExecHandleDesc;


#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_EXECUTOR_GE_EXECUTOR_C_TYPES_H_