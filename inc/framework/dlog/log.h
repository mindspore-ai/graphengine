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

#ifndef INC_FRAMEWORK_DLOG_LOG_H_
#define INC_FRAMEWORK_DLOG_LOG_H_

#include <string>
#if !defined(__ANDROID__) && !defined(ANDROID)
#include "toolchain/slog.h"
#else
#include <android/log.h>
#endif

#ifdef _MSC_VER
#define FUNC_NAME __FUNCTION__
#else
#define FUNC_NAME __PRETTY_FUNCTION__
#endif

#if !defined(__ANDROID__) && !defined(ANDROID)
#define DAV_LOGI(MOD_NAME, fmt, ...) dlog_info(static_cast<int>(GE), "%s:" #fmt, __FUNCTION__, ##__VA_ARGS__)
#define DAV_LOGW(MOD_NAME, fmt, ...) dlog_warn(static_cast<int>(GE), "%s:" #fmt, __FUNCTION__, ##__VA_ARGS__)
#define DAV_LOGE(MOD_NAME, fmt, ...) dlog_error(static_cast<int>(GE), "%s:" #fmt, __FUNCTION__, ##__VA_ARGS__)
#define DAV_LOGD(MOD_NAME, fmt, ...) dlog_debug(static_cast<int>(GE), "%s:" #fmt, __FUNCTION__, ##__VA_ARGS__)
#define DAV_EVENT(MOD_NAME, fmt, ...) dlog_event(static_cast<int>(GE), "%s:" #fmt, __FUNCTION__, ##__VA_ARGS__)
#else
#define DAV_LOGI(MOD_NAME, fmt, ...) \
  __android_log_print(ANDROID_LOG_INFO, MOD_NAME, "%s %s(%d)::" #fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define DAV_LOGW(MOD_NAME, fmt, ...) \
  __android_log_print(ANDROID_LOG_WARN, MOD_NAME, "%s %s(%d)::" #fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define DAV_LOGE(MOD_NAME, fmt, ...) \
  __android_log_print(ANDROID_LOG_ERROR, MOD_NAME, "%s %s(%d)::" #fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define DAV_LOGD(MOD_NAME, fmt, ...) \
  __android_log_print(ANDROID_LOG_DEBUG, MOD_NAME, "%s %s(%d)::" #fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define DAV_EVENT(MOD_NAME, fmt, ...) \
  __android_log_print(ANDROID_LOG_DEBUG, MOD_NAME, "%s %s(%d)::" #fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif

#define DLOG_DECLARE(level) \
  void Log_##level(const char *mod_name, const char *func, const char *file, int line, const char *format, ...)

namespace ge {
DLOG_DECLARE(INFO);
DLOG_DECLARE(WARNING);
DLOG_DECLARE(ERROR);
}  // namespace ge

#endif  // INC_FRAMEWORK_DLOG_LOG_H_
