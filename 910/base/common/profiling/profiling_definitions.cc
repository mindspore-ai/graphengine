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

#include "common/profiling_definitions.h"
#include "external/ge/ge_api_error_codes.h"
#include "external/graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "runtime/dev.h"
#include "mmpa/mmpa_api.h"
#include "graph/def_types.h"
namespace ge {
namespace profiling {
ProfilingContext &ProfilingContext::GetInstance()  {
  static ProfilingContext pc;
  return pc;
}
ProfilingContext::ProfilingContext() {
  if (!IsDumpToStdEnabled()) {
    return;
  }
  Init();
}
ProfilingContext::~ProfilingContext() = default;
void ProfilingContext::Init() {
  if (inited_) {
    return;
  }
  profiler_ = Profiler::Create();
  if (profiler_ == nullptr) {
    return;
  }
  str_index_ = kProfilingIndexEnd;
  strings_to_index_.clear();

  profiler_->RegisterString(kAclCompileAndExecute, "AclCompileAndExecute");
  profiler_->RegisterString(kAclMatchOpModel, "AclMatchOpModel");
  profiler_->RegisterString(kAclMatchStaticOpModel, "AclMatchStaticOpModel");
  profiler_->RegisterString(kAclMatchDynamicOpModel, "AclMatchDynamicOpModel");
  profiler_->RegisterString(kAclExecuteAsync, "AclExecuteAsync");
  profiler_->RegisterString(kAclBuildOpModel, "AclBuildOpModel");
  profiler_->RegisterString(kAclLoadSingleOp, "AclLoadSingleOp");
  profiler_->RegisterString(kInferShape, "InferShape");
  profiler_->RegisterString(kTiling, "Tiling");
  profiler_->RegisterString(kUpdateShape, "UpdateShape");
  profiler_->RegisterString(kConstPrepare, "ConstPrepare");
  profiler_->RegisterString(kInitHybridExecuteArgs, "InitHybridExecuteArgs");
  profiler_->RegisterString(kInitInferShapeContext, "InitInferShapeContext");
  profiler_->RegisterString(kDestroyInferShapeContext, "DestroyInferShapeContext");
  profiler_->RegisterString(kResetSubgraphExecutor, "ResetSubgraphExecutor");
  profiler_->RegisterString(kCommitInferShapeTask, "CommitInferShapeTask");
  profiler_->RegisterString(kDeviceToHost, "DeviceToHost");
  profiler_->RegisterString(kPrepareTask, "PrepareTask");
  profiler_->RegisterString(kLaunchTask, "LaunchTask");
  profiler_->RegisterString(kCommitTilingTask, "CommitTilingTask");
  profiler_->RegisterString(kAtomic, "Atomic");
  profiler_->RegisterString(kKernelLaunchPrepare, "KernelLaunchPrepare");
  profiler_->RegisterString(kRtKernelLaunch, "rtKernelLaunch");
  profiler_->RegisterString(kRtEventCreateRecord, "rtEventCreate&Record");
  profiler_->RegisterString(kRtEventSync, "rtEventSync");
  profiler_->RegisterString(kRtStreamSync, "rtStreamSync");
  profiler_->RegisterString(kRtEventDestroy, "rtEventDestroy");
  profiler_->RegisterString(kOpExecute, "OpExecute");
  profiler_->RegisterString(kModelExecute, "ModelExecute");
  profiler_->RegisterString(kAllocMem, "AllocMemory");
  profiler_->RegisterString(kCopyH2D, "CopyH2D");
  profiler_->RegisterString(kPrepareNode, "PrepareNode");
  profiler_->RegisterString(kWaitForPrepareDone, "WaitForPrepareDone");
  profiler_->RegisterString(kPropgateOutputs, "PropgateOutputs");
  profiler_->RegisterString(kOnNodeDoneCallback, "OnNodeDoneCallback");
  profiler_->RegisterString(kValidateInputTensor, "ValidateInputTensor");
  profiler_->RegisterString(kAfterExecuted, "AfterExecuted");
  profiler_->RegisterString(kRtEventSychronize, "RtEventSychronize");
  profiler_->RegisterString(kInferShapeWaitDependShape, "InferShapeWaitDependShape");
  profiler_->RegisterString(kInferShapeWaitInputTensor, "InferShapeWaitInputTensor");
  profiler_->RegisterString(kInferShapeCallInferFunc, "CallInferFunc");
  profiler_->RegisterString(kInferShapePropgate, "PropgateOutputShape");
  profiler_->RegisterString(kSelectBranch, "SelectBranch");
  profiler_->RegisterString(kExecuteSubGraph, "ExecuteSubGraph");
  profiler_->RegisterString(kInitSubGraphExecutor, "InitSubGraphExecutor");
  profiler_->RegisterString(kSelectBin, "SelectBin");
  profiler_->RegisterString(kFindCompileCache, "FindCompileCache");
  profiler_->RegisterString(kAddCompileCache, "AddCompileCache");
  profiler_->RegisterString(kFuzzCompileOp, "FuzzCompileOp");
  profiler_->RegisterString(kCalcRuningParam, "CalcRuningParam");
  profiler_->RegisterString(kGenTask, "GenTask");
  profiler_->RegisterString(kRegisterBin, "RegisterBin");

  // FFTS Plus
  profiler_->RegisterString(kFftsPlusPreThread, "FftsPlusPreThread");
  profiler_->RegisterString(kFftsPlusNodeThread, "FftsPlusNodeThread");
  profiler_->RegisterString(kFftsPlusInferShape, "FftsPlusInferShape");
  profiler_->RegisterString(kOpFftsCalculateV2, "OpFftsCalculateV2");
  profiler_->RegisterString(kInitThreadRunInfo, "InitThreadRunInfo");
  profiler_->RegisterString(kFftsPlusGraphSchedule, "FftsPlusGraphSchedule");
  profiler_->RegisterString(kKnownGetAddrAndPrefCnt, "rtGetAddrAndPrefCntWithHandle");
  profiler_->RegisterString(kKernelGetAddrAndPrefCnt, "rtKernelGetAddrAndPrefCnt");
  profiler_->RegisterString(kUpdateAddrAndPrefCnt, "UpdateAddrAndPrefCnt");
  profiler_->RegisterString(kInitOpRunInfo, "InitOpRunInfo");
  profiler_->RegisterString(kGetAutoThreadParam, "GetAutoThreadParam");
  profiler_->RegisterString(kAllocateOutputs, "AllocateOutputs");
  profiler_->RegisterString(kAllocateWorkspaces, "AllocateWorkspaces");
  profiler_->RegisterString(kInitTaskAddrs, "InitTaskAddrs");
  profiler_->RegisterString(kInitThreadRunParam, "InitThreadRunParam");
  profiler_->RegisterString(kUpdateTaskAndCache, "UpdateTaskAndCache");
  profiler_->RegisterString(kFftsPlusTaskLaunch, "rtFftsPlusTaskLaunch");

  inited_ = true;
}

int64_t ProfilingContext::RegisterStringHash(const uint64_t hash_id, const std::string &str)  {
  if (profiler_ == nullptr) {
    return -1;
  }
  const std::lock_guard<std::mutex> lock(strings_to_index_mutex_);
  auto &idx = strings_to_index_[str];
  if (idx == 0) {
    idx = str_index_;
    str_index_++;
    profiler_->RegisterStringHash(idx, hash_id, str);
  }
  GELOGD("[Register][Strhash]hash_id=%lu element=%s, idx=%ld.", hash_id, str.c_str(), idx);
  return idx;
}

int64_t ProfilingContext::RegisterString(const std::string &str) {
  if (profiler_ == nullptr) {
    return -1;
  }
  const std::lock_guard<std::mutex> lock(strings_to_index_mutex_);
  auto &index = strings_to_index_[str];
  if (index == 0) {
    index = str_index_;
    str_index_++;
    profiler_->RegisterString(index, str);
  }
  GELOGD("[Register][element]element=%s, idx=%ld.", str.c_str(), index);
  return index;
}

void ProfilingContext::UpdateHashByStr(const std::string &str, const uint64_t hash) {
  if (profiler_ == nullptr) {
    return;
  }
  const auto &index = strings_to_index_[str];
  if (index != 0) {
    profiler_->UpdateHashByIndex(index, hash);
  } else {
    // update operation can not add new element, strings_to_index_ size can not increase in this function
    (void) strings_to_index_.erase(str);
  }
  GELOGD("[Update][hash] element=%s, hash=%lu, index=%ld", str.c_str(), hash, index);
}

Status ProfilingContext::QueryHashId(const std::string &src_str, uint64_t &hash_id) {
  // when some profiling data size exceeds the specified size, query its hashId instead.
  MsprofHashData hash_data{};
  hash_data.dataLen = src_str.size();
  hash_data.data = reinterpret_cast<uint8_t *>(const_cast<char_t *>(src_str.c_str()));
  const int32_t ret = MsprofReportData(static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
                                       static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_HASH),
                                       &hash_data, static_cast<uint32_t>(sizeof(MsprofHashData)));
  if (ret != 0) {
    GELOGW("[Profiling] Query hash id of long string failed, src string is %s", src_str.c_str());
    return FAILED;
  }
  hash_id = hash_data.hashId;
  return SUCCESS;
}

void ProfilingContext::UpdateElementHashId() {
  const std::lock_guard<std::mutex> lock(strings_to_index_mutex_);
  // event type has not been register hash id
  size_t idx = 0UL;
  for (; idx < static_cast<size_t>(kProfilingIndexEnd); idx++) {
    const auto &str = profiler_->GetStringHashes()[idx].str;
    uint64_t hash_id = 0UL;
    const Status ret = QueryHashId(str, hash_id);
    if (ret != SUCCESS) {
      GELOGW("[Get][QueryHashId]Failed, ret[0x%X]", ret);
    }
    profiler_->UpdateHashByIndex(static_cast<int64_t>(idx), hash_id);
    GELOGD("[Update][hash] hash=%lu, index=%ld", hash_id, idx);
  }
  // other registered strings need update hash id
  for (; idx < (static_cast<size_t>(kProfilingIndexEnd) + GetRegisterStringNum()); idx++) {
    const auto &str = profiler_->GetStringHashes()[idx].str;
    uint64_t hash_id = 0UL;
    const Status ret = QueryHashId(str, hash_id);
    if (ret != SUCCESS) {
      GELOGW("[Get][QueryHashId]Failed, ret[0x%X]", ret);
    }
    UpdateHashByStr(str, hash_id);
  }
}

bool ProfilingContext::IsDumpToStdEnabled() {
  const uint32_t kProfilingStdLengh = 128U;
  char_t profiling_to_std_out[kProfilingStdLengh] = {};
  const bool enabled = (mmGetEnv("GE_PROFILING_TO_STD_OUT", &profiling_to_std_out[0U],
                                 kProfilingStdLengh) == EN_OK);
  return enabled;
}
}
}
