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
#ifndef AICPU_CUST_CPU_UTILS_H
#define AICPU_CUST_CPU_UTILS_H
#include "cpu_context.h"
namespace aicpu {
#define CUST_KERNEL_LOG_DEBUG(ctx, fmt, ...) \
    CustCpuKernelUtils::CustLogDebug(        \
        ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define CUST_KERNEL_LOG_INFO(ctx, fmt, ...) \
    CustCpuKernelUtils::CustLogInfo(        \
        ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define CUST_KERNEL_LOG_WARNING(ctx, fmt, ...) \
    CustCpuKernelUtils::CustLogWarning(        \
        ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define CUST_KERNEL_LOG_ERROR(ctx, fmt, ...) \
    CustCpuKernelUtils::CustLogError(        \
        ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)

class CustCpuKernelUtils {
public:
    static void CustLogDebug(aicpu::CpuKernelContext &ctx, const char *fmt, ...);
    static void CustLogWarning(aicpu::CpuKernelContext &ctx, const char *fmt, ...);
    static void CustLogInfo(aicpu::CpuKernelContext &ctx, const char *fmt, ...);
    static void CustLogError(aicpu::CpuKernelContext &ctx, const char *fmt, ...);

private:
    static void WriteCustLog(aicpu::CpuKernelContext &ctx, uint32_t level, const char *fmt, va_list v);
    static void SafeWrite(aicpu::CpuKernelContext &ctx, char *msg, size_t len);
};
}  // namespace aicpu
#endif