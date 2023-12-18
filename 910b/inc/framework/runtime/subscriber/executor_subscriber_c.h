/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_C_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_C_H_
#include "exe_graph/runtime/base_type.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  kExecuteStart,
  kExecuteEnd,
  kModelStart,
  kModelEnd,
  kExecuteEventEnd
} ExecutorEvent;

typedef void (*SubscriberFunc)(int type, void *arg, ExecutorEvent event, const void *node, KernelStatus result);
typedef struct {
  SubscriberFunc callback;
  void *arg;
} ExecutorSubscriber;
#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_C_H_
