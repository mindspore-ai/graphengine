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

#ifndef AIR_CXX_PROFILING_DEFINITIONS_H
#define AIR_CXX_PROFILING_DEFINITIONS_H
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include "graph/profiler.h"
#include "external/ge/ge_api_types.h"
#include "toolchain/prof_callback.h"
namespace ge {
namespace profiling {
enum {
  kAclCompileAndExecute,
  kAclMatchOpModel,
  kAclMatchStaticOpModel,
  kAclMatchDynamicOpModel,
  kAclExecuteAsync,
  kAclLoadSingleOp,
  kAclBuildOpModel,
  kInferShape,
  kTiling,
  kUpdateShape,
  kConstPrepare,
  kInitHybridExecuteArgs,
  kInitInferShapeContext,
  kDestroyInferShapeContext,
  kResetSubgraphExecutor,
  kCommitInferShapeTask,
  kDeviceToHost,
  kPrepareTask,
  kLaunchTask,
  kCommitTilingTask,
  kAtomic,
  kKernelLaunchPrepare,
  kRtKernelLaunch,
  kRtEventCreateRecord,
  kRtEventSync,
  kRtEventDestroy,
  kRtStreamSync,
  kOpExecute,
  kModelExecute,
  kAllocMem,
  kCopyH2D,
  kPrepareNode,
  kWaitForPrepareDone,
  kPropgateOutputs,
  kOnNodeDoneCallback,
  kValidateInputTensor,
  kAfterExecuted,
  kRtEventSychronize,
  kInferShapeWaitDependShape,
  kInferShapeWaitInputTensor,
  kInferShapeCallInferFunc,
  kInferShapePropgate,
  // v2 control node
  kSelectBranch,
  kExecuteSubGraph,
  kInitSubGraphExecutor,
  // fuzz compile
  kSelectBin,
  kFindCompileCache,
  kAddCompileCache,
  kFuzzCompileOp,
  kCalcRuningParam,
  kGenTask,
  kRegisterBin,

  // FFTS Plus
  kFftsPlusPreThread,
  kFftsPlusNodeThread,
  kFftsPlusInferShape,
  kOpFftsCalculateV2,
  kInitThreadRunInfo,
  kFftsPlusGraphSchedule,
  kKnownGetAddrAndPrefCnt,
  kKernelGetAddrAndPrefCnt,
  kUpdateAddrAndPrefCnt,
  kInitOpRunInfo,
  kGetAutoThreadParam,
  kAllocateOutputs,
  kAllocateWorkspaces,
  kInitTaskAddrs,
  kInitThreadRunParam,
  kUpdateTaskAndCache,
  kFftsPlusTaskLaunch,

  // Add new definitions here
  kProfilingIndexEnd
};
constexpr uint64_t kInvalidHashId = 0UL;

class ProfilingContext {
 public:
  static bool IsDumpToStdEnabled();
  static ProfilingContext &GetInstance();
  ProfilingContext();
  ~ProfilingContext();

  /*
   * 还有一种思路是`IsEnabled`只判断profiler_是否为空指针，不再设置单独的enabled标记位，这样可以少一个标记位。
   * 但是这么做就意味着，profiler_实例在未使能profiling时，必须是空指针状态。
   * 为了性能考虑，profiling机制在编译和加载时，就会调用`RegisterString`，向profiler_注册字符串，后续执行时，只会使用注册好的index了。
   * 因此存在一种场景：编译时并未使能profiling（因为编译时间很长，使能profiling也无法真实反应执行时的耗时状态），
   * 因此编译时注册字符串的动作并没有生效。在执行时，动态的打开了profiling，这种场景下，执行时无法拿到注册后字符串
   */
  bool IsEnabled() const noexcept {
    return enabled_ && (profiler_ != nullptr);
  }
  void SetEnable() noexcept {
    enabled_ = true;
  }
  void SetDisable() noexcept {
    enabled_ = false;
  }

  void RecordCurrentThread(const int64_t element, const int64_t event, const EventType et,
                           const std::chrono::time_point<std::chrono::system_clock> time_point) {
    if (IsEnabled()) {
      profiler_->RecordCurrentThread(element, event, et, time_point);
    }
  }

  void RecordCurrentThread(const int64_t element, const int64_t event, const EventType et) {
    RecordCurrentThread(element, event, et, std::chrono::system_clock::now());
  }

  const Profiler *GetProfiler() const {
    return profiler_.get();
  }

  void Dump(std::ostream &out_stream) const {
    if (IsEnabled()) {
      profiler_->Dump(out_stream);
    } else {
      out_stream << "Profiling not enable, skip to dump" << std::endl;
    }
  }

  void DumpToStdOut() const {
    Dump(std::cout);
  }

  void Reset() {
    if (IsEnabled()) {
      profiler_->Reset();
    }
  }

  int64_t RegisterString(const std::string &str);
  int64_t RegisterStringHash(const uint64_t hash_id, const std::string &str);
  void UpdateElementHashId();
  static Status QueryHashId(const std::string &src_str, uint64_t &hash_id);
  size_t GetRegisterStringNum() const {
    return strings_to_index_.size();
  }

  void Init();

 private:
  void UpdateHashByStr(const std::string &str, const uint64_t hash);

 private:
  bool inited_;
  bool enabled_;
  int64_t str_index_;
  std::unordered_map<std::string, int64_t> strings_to_index_;
  std::mutex strings_to_index_mutex_;
  std::unique_ptr<Profiler> profiler_;
};

class ScopeProfiler {
 public:
  ScopeProfiler(const int64_t element, const int64_t event) : element_(element), event_(event) {
    if (ProfilingContext::GetInstance().IsEnabled()) {
      start_trace_ = std::chrono::system_clock::now();
    }
  }
  ~ScopeProfiler() {
    if (ProfilingContext::GetInstance().IsEnabled()) {
      ProfilingContext::GetInstance().RecordCurrentThread(element_, event_, EventType::kEventStart, start_trace_);
      ProfilingContext::GetInstance().RecordCurrentThread(element_, event_, EventType::kEventEnd);
    }
  }
  void SetElement(const int64_t element) {
    element_ = element;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_trace_;
  int64_t element_;
  int64_t event_;
};
}  // namespace profiling
}  // namespace ge
#define PROFILING_START(element, event)                                                  \
  ge::profiling::ProfilingContext::GetInstance().RecordCurrentThread((element), (event), \
                                                                     ge::profiling::EventType::kEventStart)
#define PROFILING_END(element, event)                                                    \
  ge::profiling::ProfilingContext::GetInstance().RecordCurrentThread((element), (event), \
                                                                     ge::profiling::EventType::kEventEnd)
#define PROFILING_SCOPE(element, event) ge::profiling::ScopeProfiler profiler((element), (event))
#define PROFILING_SCOPE_CONST(element, event) const ge::profiling::ScopeProfiler profiler((element), (event))
#define PROFILING_SCOPE_ELEMENT(element) profiler.SetElement((element))
#endif  // AIR_CXX_PROFILING_DEFINITIONS_H
