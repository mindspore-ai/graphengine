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

#ifndef GE_GE_CALL_WRAPPER_H_
#define GE_GE_CALL_WRAPPER_H_
#include "framework/common/debug/ge_log.h"

#define RUN_WITH_TIMESTAMP_NAME(var_name, prefix, func, ...)           \
  do {                                                                 \
    GE_TIMESTAMP_START(var_name);                                      \
    auto ret_inner_macro = func(__VA_ARGS__);                          \
    GE_TIMESTAMP_END(var_name, #prefix "::" #func)                     \
    if (ret_inner_macro != ge::SUCCESS) {                              \
      GELOGE(ret_inner_macro, "Failed to process " #prefix "_" #func); \
      return ret_inner_macro;                                          \
    }                                                                  \
  } while (0)

#define JOIN_NAME_INNER(a, b) a##b
#define JOIN_NAME(a, b) JOIN_NAME_INNER(a, b)
#define COUNTER_NAME(a) JOIN_NAME(a, __COUNTER__)
#define GE_RUN(prefix, func, ...) \
  RUN_WITH_TIMESTAMP_NAME(COUNTER_NAME(ge_timestamp_##prefix), prefix, func, __VA_ARGS__)

#endif  // GE_GE_CALL_WRAPPER_H_
