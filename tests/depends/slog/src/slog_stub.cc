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

#include "toolchain/slog.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

void dav_log(int module_id, const char *fmt, ...) {}

void DlogErrorInner(int module_id, const char *fmt, ...) { dav_log(module_id, fmt); }

void DlogWarnInner(int module_id, const char *fmt, ...) { dav_log(module_id, fmt); }

void DlogInfoInner(int module_id, const char *fmt, ...) { dav_log(module_id, fmt); }

void DlogDebugInner(int module_id, const char *fmt, ...) { dav_log(module_id, fmt); }

void DlogEventInner(int module_id, const char *fmt, ...) { dav_log(module_id, fmt); }

void DlogInner(int moduleId, int level, const char *fmt, ...) { dav_log(moduleId, fmt); }

void DlogWithKVInner(int moduleId, int level, KeyValue *pstKVArray, int kvNum, const char *fmt, ...) {
  dav_log(moduleId, fmt);
}

int dlog_getlevel(int module_id, int *enable_event) { return DLOG_DEBUG; }
