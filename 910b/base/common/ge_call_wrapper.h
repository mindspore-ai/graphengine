/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef GE_GE_CALL_WRAPPER_H_
#define GE_GE_CALL_WRAPPER_H_

#include "framework/common/debug/ge_log.h"
#include "common/util.h"

/*lint --emacro((773),GE_TIMESTAMP_START)*/
/*lint -esym(773,GE_TIMESTAMP_START)*/
#define GE_TIMESTAMP_START(stage) const uint64_t startUsec_##stage = ge::GetCurrentTimestamp()

#define GE_TIMESTAMP_END(stage, stage_name)                                           \
  do {                                                                                \
    const uint64_t endUsec_##stage = ge::GetCurrentTimestamp();                       \
    GELOGI("[GEPERFTRACE] The time cost of %s is [%lu] micro second.", (stage_name),  \
            (endUsec_##stage - startUsec_##stage));                                   \
  } while (false)

#define GE_TIMESTAMP_EVENT_END(stage, stage_name)                                     \
  do {                                                                                \
    const uint64_t endUsec_##stage = ge::GetCurrentTimestamp();                       \
    GEEVENT("[GEPERFTRACE] The time cost of %s is [%lu] micro second.", (stage_name), \
            (endUsec_##stage - startUsec_##stage));                                   \
  } while (false)

#define GE_TIMESTAMP_CALLNUM_START(stage)                 \
  uint64_t startUsec_##stage = ge::GetCurrentTimestamp(); \
  uint64_t call_num_of##stage = 0U;                       \
  uint64_t time_of##stage = 0U

#define GE_TIMESTAMP_RESTART(stage) (startUsec_##stage = ge::GetCurrentTimestamp())

#define GE_TIMESTAMP_ADD(stage)                                    \
  time_of##stage += ge::GetCurrentTimestamp() - startUsec_##stage; \
  call_num_of##stage++

#define GE_TIMESTAMP_CALLNUM_END(stage, stage_name)                                                                \
  GELOGI("[GEPERFTRACE] The time cost of %s is [%lu] micro second, call num is %lu", (stage_name), time_of##stage, \
          call_num_of##stage)

#define GE_TIMESTAMP_CALLNUM_EVENT_END(stage, stage_name)                                                           \
  GEEVENT("[GEPERFTRACE] The time cost of %s is [%lu] micro second, call num is %lu", (stage_name), time_of##stage, \
          call_num_of##stage)

#define RETURN_IF_ERROR_WITH_TIMESTAMP_NAME(var_name, prefix, func, ...)  \
  do {                                                                    \
    GE_TIMESTAMP_START(var_name);                                         \
    const auto ret_inner_macro = (func)(__VA_ARGS__);                     \
    GE_TIMESTAMP_END(var_name, #prefix "::" #func);                       \
    if (ret_inner_macro != ge::SUCCESS) {                                 \
      GELOGE(ret_inner_macro, "[Process][" #prefix "_" #func "] failed"); \
      return ret_inner_macro;                                             \
    }                                                                     \
  } while (false)

#define RETURN_IF_ERROR_WITH_PERF_TIMESTAMP_NAME(var_name, prefix, func, ...) \
  do {                                                                        \
    GE_TIMESTAMP_START(var_name);                                             \
    const auto ret_inner_macro = (func)(__VA_ARGS__);                         \
    GE_TIMESTAMP_EVENT_END(var_name, #prefix "::" #func);                     \
    if (ret_inner_macro != ge::SUCCESS) {                                     \
      GELOGE(ret_inner_macro, "[Process][" #prefix "_" #func "] failed");     \
      return ret_inner_macro;                                                 \
    }                                                                         \
  } while (false)

#define JOIN_NAME_INNER(a, b) a##b
#define JOIN_NAME(a, b) JOIN_NAME_INNER(a, b)
#define COUNTER_NAME(a) JOIN_NAME(a, __COUNTER__)
#define GE_RUN(prefix, func, ...) \
  RETURN_IF_ERROR_WITH_TIMESTAMP_NAME(COUNTER_NAME(ge_timestamp_##prefix), prefix, func, __VA_ARGS__)
#define GE_RUN_PERF(prefix, func, ...) \
  RETURN_IF_ERROR_WITH_PERF_TIMESTAMP_NAME(COUNTER_NAME(ge_timestamp_##prefix), prefix, func, __VA_ARGS__)

#endif  // GE_GE_CALL_WRAPPER_H_
