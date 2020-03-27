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

#include <cce/dnn.h>
#include <securec.h>

#define EVENT_LENTH 10

rtError_t rtCtxSetCurrent(rtContext_t ctx) { return RT_ERROR_NONE; }

rtError_t rtGetStreamId(rtStream_t stream, int32_t *streamId) {
  *streamId = 0;
  return RT_ERROR_NONE;
}

rtError_t rtCtxGetCurrent(rtContext_t *ctx) {
  int x = 1;
  *ctx = (void *)x;
  return RT_ERROR_NONE;
}

rtError_t rtCtxSetDryRun(rtContext_t ctx, rtDryRunFlag_t enable, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtEventGetTimeStamp(uint64_t *time, rtEvent_t event) {
  *time = 12345;
  return RT_ERROR_NONE;
}

rtError_t rtEventCreate(rtEvent_t *event) {
  *event = new int[EVENT_LENTH];
  return RT_ERROR_NONE;
}
rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtEventSynchronize(rtEvent_t event) { return RT_ERROR_NONE; }

rtError_t rtEventDestroy(rtEvent_t event) {
  delete[](int *) event;
  return RT_ERROR_NONE;
}

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type) {
  *devPtr = new uint8_t[size];
  return RT_ERROR_NONE;
}

rtError_t rtMemset(void *devPtr, uint64_t destMax, uint32_t value, uint64_t count) { return RT_ERROR_NONE; }

rtError_t rtFree(void *devPtr) {
  delete[](uint8_t *) devPtr;
  return RT_ERROR_NONE;
}

rtError_t rtMallocHost(void **hostPtr, uint64_t size) {
  *hostPtr = new uint8_t[size];
  return RT_ERROR_NONE;
}

rtError_t rtFreeHost(void *hostPtr) {
  delete[](uint8_t *) hostPtr;
  return RT_ERROR_NONE;
}

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority) {
  *stream = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtStreamDestroy(rtStream_t stream) {
  if (stream != nullptr) {
    delete (uint32_t *)stream;
  }
  return RT_ERROR_NONE;
}

rtError_t rtSetDevice(int32_t device) { return RT_ERROR_NONE; }

rtError_t rtStreamSynchronize(rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind) {
#ifdef OTQT_UT
  if (destMax == 12 && count == 12) {  // UTEST_kernelinfo_manager.all_success special treatment
    memcpy_s(dst, destMax, src, count);
  }
#endif
  return RT_ERROR_NONE;
}
rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                        rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event) { return RT_ERROR_NONE; }

rtError_t rtGetDeviceCount(int32_t *count) {
  *count = 1;
  return RT_ERROR_NONE;
}

rtError_t rtDeviceReset(int32_t device) { return RT_ERROR_NONE; }

rtError_t rtEventElapsedTime(float *time, rtEvent_t start, rtEvent_t end) {
  *time = 10.0f;
  return RT_ERROR_NONE;
}
rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName, const void *devFunc) {
  return RT_ERROR_NONE;
}

rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
                             uint32_t funcMode) {
  return RT_ERROR_NONE;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle) { return RT_ERROR_NONE; }

rtError_t rtKernelConfigTransArg(const void *ptr, uint64_t size, uint32_t flag, void **arg) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc,
                         rtStream_t stream) {
  return RT_ERROR_NONE;
}
rtError_t rtSetupArgument(const void *arg, uint32_t size, uint32_t offset) { return RT_ERROR_NONE; }
rtError_t rtLaunch(const void *stubFunc) { return RT_ERROR_NONE; }
rtError_t rtDevBinaryUnRegister(void *handle) { return RT_ERROR_NONE; }
rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtSetProfDir(char *profdir) { return RT_ERROR_NONE; }

rtError_t rtSetProfDirEx(char *profdir, char *address, char *job_ctx) { return RT_ERROR_NONE; }

rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t *aiCoreMemorySize) { return RT_ERROR_NONE; }

rtError_t rtSetKernelReportCallback(rtKernelReportCallback callBack) {
  rtKernelInfo rt_kernel_info = {0};
  rt_kernel_info.arg_size = 12;
  rt_kernel_info.task_offset = 100;
  rt_kernel_info.arg = (void *)100;
  rt_kernel_info.module_addr = (void *)100;
  rt_kernel_info.module_size = 100;

  rtStream_t stream;
  callBack(stream, &rt_kernel_info);
  return RT_ERROR_NONE;
}

rtError_t rtMemAdvise(void *ptr, uint64_t size, uint32_t advise) { return RT_ERROR_NONE; }

/// @ingroup rt_kernel
/// @brief start fusion kernels.
/// @param [in] stream   stream for fusion kernels
/// @return RT_ERROR_NONE for ok, errno for failed
rtError_t rtKernelFusionStart(rtStream_t stream) { return RT_ERROR_NONE; }

/// @ingroup rt_kernel
/// @brief end fusion kernels.
/// @param [in] stream   stream for fusion kernels
/// @return RT_ERROR_NONE for ok, errno for failed
rtError_t rtKernelFusionEnd(rtStream_t stream) { return RT_ERROR_NONE; }
rtError_t rtMemGetInfo(size_t *free, size_t *total) {
  *free = 512UL * 1024UL * 1024UL;
  *total = 1024UL * 1024UL * 1024UL;
  return RT_ERROR_NONE;
}

rtError_t rtMemAllocManaged(void **ptr, uint64_t size, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtMemFreeManaged(void *ptr) { return RT_ERROR_NONE; }

rtError_t rtMetadataRegister(void *handle, const char *metadata) { return RT_ERROR_NONE; }
rtError_t rtSetTaskGenCallback(rtTaskGenCallback callback) { return RT_ERROR_NONE; }

rtError_t rtModelCreate(rtModel_t *model, uint32_t flag) {
  *model = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtModelDestroy(rtModel_t model) {
  delete model;
  return RT_ERROR_NONE;
}

rtError_t rtModelBindStream(rtModel_t model, rtStream_t stream, uint32_t flag) { return RT_ERROR_NONE; }
rtError_t rtModelUnbindStream(rtModel_t model, rtStream_t stream) { return RT_ERROR_NONE; }
rtError_t rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtGetFunctionByName(const char *stubName, void **stubFunc) {
  *(char **)stubFunc = "func";
  return RT_ERROR_NONE;
}

rtError_t rtQueryFunctionRegistered(const char *stubName) { return RT_ERROR_NONE; }

rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunchEx(void *args, uint32_t argsSize, uint32_t flags, rtStream_t stream_) { return RT_ERROR_NONE; }

rtError_t rtCpuKernelLaunch(const void *soName, const void *kernelName, uint32_t blockDim, const void *args,
                            uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtModelGetTaskId(void *handle, uint32_t *taskid) {
  *taskid = 0;
  return RT_ERROR_NONE;
}
rtError_t rtEndGraph(rtModel_t model, rtStream_t stream) { return RT_ERROR_NONE; }
rtError_t rtProfilerStop(void) { return RT_ERROR_NONE; }

rtError_t rtSetDvfsProfile(DvfsProfileMode mode) { return RT_ERROR_NONE; }

rtError_t rtUnsetDvfsProfile() { return RT_ERROR_NONE; }

rtError_t rtGetDvfsProfile(DvfsProfileMode *pmode) { return RT_ERROR_NONE; }

rtError_t rtCtxDestroy(rtContext_t ctx) { return RT_ERROR_NONE; }

rtError_t rtProfilerInit(const char *profdir, const char *address, const char *job_ctx) { return RT_ERROR_NONE; }

rtError_t rtProfilerStart(void) { return RT_ERROR_NONE; }

rtError_t rtLabelCreate(rtLabel_t *label) { return RT_ERROR_NONE; }

rtError_t rtLabelDestroy(rtLabel_t label) { return RT_ERROR_NONE; }

rtError_t rtLabelSet(rtLabel_t label, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtLabelSwitch(void *ptr, rtCondition_t condition, uint32_t value, rtLabel_t trueLabel, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelGoto(rtLabel_t label, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtInvalidCache(uint64_t base, uint32_t len) { return RT_ERROR_NONE; }

rtError_t rtModelLoadComplete(rtModel_t model) { return RT_ERROR_NONE; }

rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags) {
  *stream = new uint32_t;
  return RT_ERROR_NONE;
}

rtError_t rtFlushCache(uint64_t base, uint32_t len) { return RT_ERROR_NONE; }

rtError_t rtProfilerTrace(uint64_t id, bool notify, uint32_t flags, rtStream_t stream_) { return RT_ERROR_NONE; }

rtError_t rtMemSetRC(const void *devPtr, uint64_t size, uint32_t readCount) { return RT_ERROR_NONE; }

rtError_t rtStreamSwitch(void *ptr, rtCondition_t condition, int64_t value, rtStream_t true_stream, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamSwitchEx(void *ptr, rtCondition_t condition, void *value_ptr, rtStream_t true_stream,
                           rtStream_t stream, rtSwitchDataType_t dataType) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamActive(rtStream_t active_stream, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtEventReset(rtEvent_t event, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtGetDevice(int32_t *device) { return RT_ERROR_NONE; }

rtError_t rtDatadumpInfoLoad(const void *dumpInfo, uint32_t length) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                 rtSmDesc_t *smDesc, rtStream_t stream_, uint32_t flags) {
  return RT_ERROR_NONE;
}

rtError_t rtCpuKernelLaunchWithFlag(const void *soName, const void *kernelName, uint32_t coreDim, const void *args,
                                    uint32_t argsSize, rtL2Ctrl_t *l2ctrl, rtStream_t stream_, uint32_t flags) {
  return RT_ERROR_NONE;
}