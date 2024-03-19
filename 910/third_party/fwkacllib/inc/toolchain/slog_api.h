/**
 * @file slog_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef SLOG_API_H
#define SLOG_API_H
#include "slog.h"
#ifdef __cplusplus
#ifndef LOG_CPP
extern "C" {
#endif
#endif // __cplusplus

typedef struct tagKV {
    char *kname;
    char *value;
} KeyValue;

/**
 * @ingroup slog
 * @brief Internal log interface, other modules are not allowed to call this interface
 */
DLL_EXPORT void DlogErrorInner(int moduleId, const char *fmt, ...);
DLL_EXPORT void DlogWarnInner(int moduleId, const char *fmt, ...);
DLL_EXPORT void DlogInfoInner(int moduleId, const char *fmt, ...);
DLL_EXPORT void DlogDebugInner(int moduleId, const char *fmt, ...);
DLL_EXPORT void DlogEventInner(int moduleId, const char *fmt, ...);
DLL_EXPORT void DlogInner(int moduleId, int level, const char *fmt, ...);
DLL_EXPORT void DlogWithKVInner(int moduleId, int level, KeyValue *pstKVArray, int kvNum, const char *fmt, ...);

#ifdef __cplusplus
#ifndef LOG_CPP
}
#endif // LOG_CPP
#endif // __cplusplus

#ifdef LOG_CPP
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup slog
 * @brief Internal log interface, other modules are not allowed to call this interface
 */
DLL_EXPORT void DlogInnerForC(int moduleId, int level, const char *fmt, ...);
DLL_EXPORT void DlogWithKVInnerForC(int moduleId, int level, KeyValue *pstKVArray, int kvNum, const char *fmt, ...);

#ifdef __cplusplus
}
#endif
#endif // LOG_CPP
#endif