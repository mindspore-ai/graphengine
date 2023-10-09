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
#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_GLOBAL_PROFILER_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_SUBSCRIBER_GLOBAL_PROFILER_H_

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <fstream>
#include "mmpa/mmpa_api.h"
#include "built_in_subscriber_definitions.h"
#include "common/debug/ge_log.h"
#include "framework/common/ge_visibility.h"
#include "runtime/subscriber/executor_subscriber_c.h"
#include "runtime/base.h"
#include "toolchain/prof_api.h"
#include "common/checker.h"

namespace gert {
constexpr uint32_t kTensorInfoBytes = 44UL;
constexpr uint32_t kTensorInfoBytesWithCap = 56U;
constexpr size_t kMaxContextIdNum =
    (static_cast<size_t>(MSPROF_ADDTIONAL_INFO_DATA_LENGTH) - sizeof(uint32_t) - sizeof(uint64_t)) / sizeof(uint32_t);
struct ProfilingData {
  uint64_t name_idx;
  uint64_t type_idx;
  ExecutorEvent event;
  std::chrono::time_point<std::chrono::system_clock> timestamp;
  int64_t thread_id;
};
enum class GeProfInfoType {
  // model level
  kModelExecute = MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE + 1,  // 模型执行
  kModelLoad,         // 模型加载
  kInputCopy,         // input拷贝
  kOutputCopy,        // output拷贝
  kModelLevelEnd,
  // node level
  // todo
  kInferShape = MSPROF_REPORT_NODE_GE_API_BASE_TYPE + 1,
  kCompatibleInferShape,
  kTiling,
  kCompatibleTiling,
  kStreamSync,
  kStepInfo,
  kNodeLevelEnd,
  // acl level
  kIsGraphNeedRebuild = MSPROF_REPORT_ACL_GRAPH_BASE_TYPE + 1,
  kRemoveGraph,
  kAddGraph,
  kBuildGraph,
  kRunGraphAsync,
  kGEInitialize,
  kGEFinalize,
  kAclLevelEnd
};

struct ContextIdInfoWrapper {
  MsprofAdditionalInfo context_id_info;
  std::string op_name;
};

extern const std::unordered_map<std::string, GeProfInfoType> kNamesToProfTypes;
class GlobalProfiler {
 public:
  GlobalProfiler() = default;
  void Record(uint64_t name_idx, uint64_t type_idx, ExecutorEvent event,
              std::chrono::time_point<std::chrono::system_clock> timestamp) {
    auto index = count_.fetch_add(1, std::memory_order_relaxed);
    if (index >= kProfilingDataCap) {
      return;
    }
    thread_local static auto tid = static_cast<int64_t>(mmGetTid());
    records_[index] = {name_idx, type_idx, event, timestamp, tid};
  }
  void Dump(std::ostream &out_stream, std::vector<std::string> &idx_to_str) const;
  size_t GetCount() const {
    return count_.load();
  }

 private:
  std::atomic<size_t> count_{0UL};
  ProfilingData records_[kProfilingDataCap];
};

struct ProfFusionMemSize {
  uint64_t input_mem_size{0UL};
  uint64_t output_mem_size{0UL};
  uint64_t weight_mem_size{0UL};
  uint64_t workspace_mem_size{0UL};
};

class VISIBILITY_EXPORT GlobalProfilingWrapper {
 public:
  GlobalProfilingWrapper(const GlobalProfilingWrapper &) = delete;
  GlobalProfilingWrapper(GlobalProfilingWrapper &&) = delete;
  GlobalProfilingWrapper &operator=(const GlobalProfilingWrapper &) = delete;
  GlobalProfilingWrapper &operator=(GlobalProfilingWrapper &&) = delete;

  static GlobalProfilingWrapper *GetInstance() {
    static GlobalProfilingWrapper global_prof_wrapper;
    return &global_prof_wrapper;
  }

  static void OnGlobalProfilingSwitch(void *ins, uint64_t enable_flags);

  void Init(const uint64_t enable_flags);

  void Free() {
    global_profiler_.reset(nullptr);
    SetEnableFlags(0UL);
  }

  GlobalProfiler *GetGlobalProfiler() const {
    return global_profiler_.get();
  }

  void SetEnableFlags(const uint64_t enable_flags) {
    enable_flags_.store(enable_flags);
  }

  uint64_t GetRecordCount() {
    if (global_profiler_ == nullptr) {
      return 0UL;
    }
    return global_profiler_->GetCount();
  }

  uint64_t GetEnableFlags() const {
    return enable_flags_.load();
  }

  bool IsEnabled(ProfilingType profiling_type) {
    return enable_flags_.load() & BuiltInSubscriberUtil::EnableBit<ProfilingType>(profiling_type);
  }

  void DumpAndFree(std::ostream &out_stream) {
    Dump(out_stream);
    Free();
  }
  void Dump(std::ostream &out_stream) {
    if (global_profiler_ != nullptr) {
      global_profiler_->Dump(out_stream, idx_to_str_);
    }
  }
  void Record(uint64_t name_idx, uint64_t type_idx, ExecutorEvent event,
              std::chrono::time_point<std::chrono::system_clock> timestamp) {
    if (global_profiler_ != nullptr) {
      global_profiler_->Record(name_idx, type_idx, event, timestamp);
    }
  }

  uint64_t RegisterString(const std::string &name);

  const std::vector<std::string> &GetIdxToStr() const {
    return idx_to_str_;
  }
  uint32_t GetProfModelId() const;
  ge::Status RegisterExtendProfType(const std::string &name, const uint32_t idx) const;
  void IncProfModelId();
  void RegisterBuiltInString();
  ge::Status RegisterProfType() const;
  static ge::Status ReportEvent(const uint64_t item_id, const uint32_t request_id, const GeProfInfoType type,
                             MsprofEvent &prof_single_event);
  static ge::Status ReportApiInfo(const uint64_t begin_time, const uint64_t end_time, const uint64_t item_id,
                                  const uint32_t api_type);
  static ge::Status ReportApiInfoModelLevel(const uint64_t begin_time, const uint64_t end_time, const uint64_t item_id,
                                            const uint32_t api_type);
  static ge::Status ReportTaskMemoryInfo(const std::string &node_name);
  static ge::Status ReportTensorInfo(const uint32_t tid, const bool is_aging, const ge::TaskDescInfo &task_desc_info);
  static void BuildNodeBasicInfo(const ge::OpDescPtr &op_desc, const uint32_t block_dim,
                                 const std::pair<uint64_t, uint64_t> &op_name_and_type_hash, const uint32_t task_type,
                                 MsprofCompactInfo &node_basic_info);
  static void BuildCompactInfo(const uint64_t prof_time, MsprofCompactInfo &node_basic_info);
  static void BuildApiInfo(const std::pair<uint64_t, uint64_t> &prof_time, const uint32_t api_type,
                           const uint64_t item_id, MsprofApi &api);
  static void BuildContextIdInfo(const uint64_t prof_time, const std::vector<uint32_t> &context_ids,
                                 const size_t op_name, const std::string &op_name_str,
                                 std::vector<ContextIdInfoWrapper> &infos);

  static ge::Status ReportGraphIdMap(const uint64_t prof_time, const uint32_t tid,
                                     const std::pair<uint32_t, uint32_t> graph_id_and_model_id, const bool is_aging,
                                     const size_t model_name = 0UL);
  static ge::Status ProfileStepTrace(const uint64_t index_id, const uint32_t model_id, const uint16_t tag_id,
                                     const rtStream_t stream);
  static void BuildSingleProfTensorInfo(const uint32_t tid, const ge::TaskDescInfo &task_desc_info, const size_t index,
                                        const uint32_t tensor_num, MsprofAdditionalInfo &tensor_info);

  static void BuildFusionOpInfo(const ProfFusionMemSize &mem_size, const std::vector<std::string> &origin_op_names,
                                const size_t op_name, std::vector<MsprofAdditionalInfo> &infos);

  static unsigned int ReportLogicStreamInfo(
      const uint64_t timestamp, const uint32_t tid,
      const std::unordered_map<uint32_t, std::vector<uint32_t>> &logic_stream_ids_to_physic_stream_ids,
      const uint16_t is_aging);

 private:
  GlobalProfilingWrapper();
  static void BuildSingleContextIdInfo(const uint64_t prof_time, const vector<uint32_t> &context_ids,
                                       const size_t index, const size_t context_id_num, MsprofAdditionalInfo &info);

  static void BuildProfFusionInfoBase(const ProfFusionMemSize &mem_size, const size_t fusion_op_num,
                                      const size_t op_name, ProfFusionOpInfo *prof_fusion_data);

 private:
  std::unique_ptr<GlobalProfiler> global_profiler_{nullptr};
  std::atomic<uint64_t> enable_flags_{0UL};
  uint64_t str_idx_{0UL};
  bool is_builtin_string_registered_{false};
  std::vector<std::string> idx_to_str_;
  std::mutex register_mutex_;
  std::mutex mutex_;
  // rt2流程acl会给推理场景生成一个model id，但是静态子图没有办法获取这个model id，生成的davinci model
  // 的model id为uint32_t的最大值，因此这种情况下，需要给它一个model id且生成model id的逻辑需要与acl一致
  std::atomic_uint32_t model_id_generator_{std::numeric_limits<uint32_t>::max() / 2U};
};

class ScopeProfiler {
 public:
  ScopeProfiler(const size_t element, const size_t event) : element_(element), event_(event) {
    if (GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kGeHost)) {
      start_trace_ = std::chrono::system_clock::now();
    }
  }

  void SetElement(const size_t element) {
    element_ = element;
  }

  ~ScopeProfiler() {
    if (GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kGeHost)) {
      GlobalProfilingWrapper::GetInstance()->Record(element_, event_, kExecuteStart, start_trace_);
      GlobalProfilingWrapper::GetInstance()->Record(element_, event_, kExecuteEnd, std::chrono::system_clock::now());
    }
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_trace_;
  size_t element_;
  size_t event_;
};

class GraphProfilingReporter {
 public:
  explicit GraphProfilingReporter(const GeProfInfoType api_id) : graphApi_(api_id) {
    if (GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kTaskTime)) {
      start_time_ = MsprofSysCycleTime();
    }
  }

  ~GraphProfilingReporter() {
    if (GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kTaskTime)) {
      const uint64_t end_time_ = MsprofSysCycleTime();
      MsprofApi api{};
      api.beginTime = start_time_;
      api.endTime = end_time_;
      thread_local static auto tid = mmGetTid();
      api.threadId = static_cast<uint32_t>(tid);
      api.level = MSPROF_REPORT_ACL_LEVEL;
      api.type = static_cast<uint32_t>(graphApi_);
      (void)MsprofReportApi(true, &api);
    }
  }

 private:
  uint64_t start_time_ = 0UL;
  const GeProfInfoType graphApi_;
};

class VISIBILITY_EXPORT ProfilerRegistry {
 public:
  ProfilerRegistry(const ProfilerRegistry &) = delete;
  ProfilerRegistry(ProfilerRegistry &&) = delete;
  ProfilerRegistry &operator=(const ProfilerRegistry &) = delete;
  ProfilerRegistry &operator=(ProfilerRegistry &&) = delete;

  static ProfilerRegistry &GetInstance();

  void SaveRegistryType(const std::string &type, const bool launch_flag);
  bool IsProfLaunchType(const std::string &kernel_type, const bool launch_flag = true);
  bool IsProfDavinciModelExecuteType(const std::string &kernel_type);
 private:
  ProfilerRegistry() noexcept = default;
  std::vector<std::string> register_prof_launch_type_{};
  std::vector<std::string> register_prof_non_launch_type_{};
  std::mutex mutex_;
};

class ProfLaunchTypeRegistry {
 public:
  explicit ProfLaunchTypeRegistry(const std::string &type, const bool launch_flag) noexcept {
    ProfilerRegistry::GetInstance().SaveRegistryType(type, launch_flag);
  }
};
}  // namespace gert

#define GE_PROFILING_START(event)                                                             \
  std::chrono::time_point<std::chrono::system_clock> event##start_time;                       \
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kGeHost)) { \
    event##start_time = std::chrono::system_clock::now();                                     \
  }

#define GE_PROFILING_END(name_idx, type_idx, event)                                                         \
  do {                                                                                                      \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kGeHost)) {             \
      gert::GlobalProfilingWrapper::GetInstance()->Record(name_idx, type_idx, ExecutorEvent::kExecuteStart, \
                                                          event##start_time);                               \
      gert::GlobalProfilingWrapper::GetInstance()->Record(name_idx, type_idx, ExecutorEvent::kExecuteEnd,   \
                                                          std::chrono::system_clock::now());                \
    }                                                                                                       \
  } while (false)

#define CANN_PROFILING_API_END(item_id, info_type, event)                                                          \
  do {                                                                                                             \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {                  \
      const auto end_time = MsprofSysCycleTime();                                                                  \
      gert::GlobalProfilingWrapper::GetInstance()->ReportApiInfo(event##begin_time, end_time, item_id, info_type); \
    }                                                                                                              \
  } while (false)

#define CANN_PROFILING_API_START(event)                                                         \
  uint64_t event##begin_time;                                                                   \
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) { \
    event##begin_time = MsprofSysCycleTime();                                                   \
  }

#define CANN_PROFILING_MODEL_API_END(item_id, info_type, end_time, event)                                          \
  do {                                                                                                             \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {                  \
      gert::GlobalProfilingWrapper::GetInstance()->ReportApiInfoModelLevel(                                        \
      event##begin_time, end_time, item_id, info_type);                                                            \
    }                                                                                                              \
  } while (false)

#define CANN_PROFILING_MODEL_API_START(event)                                                   \
  uint64_t event##begin_time = 0UL;                                                             \
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) { \
    event##begin_time = MsprofSysCycleTime();                                                   \
  }

#define CANN_PROFILING_EVENT_START(item_id, request_id, info_type, single_event)                     \
  do {                                                                                               \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {    \
      (void)gert::GlobalProfilingWrapper::ReportEvent(item_id, request_id, info_type, single_event); \
    }                                                                                                \
  } while (false)

#define CANN_PROFILING_EVENT_END(item_id, request_id, info_type, single_event)                    \
  do {                                                                                            \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) { \
      const uint64_t prof_time = MsprofSysCycleTime();                                            \
      (single_event).timeStamp = prof_time;                                                       \
      (void)MsprofReportEvent(true, &(single_event));                                             \
    }                                                                                             \
  } while (false)

#define CANN_PROFILING_INFER_EVENT_START(request_id, info_type, single_event)                                        \
  do {                                                                                                               \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {                    \
      (void)gert::GlobalProfilingWrapper::ReportEvent(gert::GlobalProfilingWrapper::GetInstance()->GetProfModelId(), \
                                                      request_id, info_type, single_event);                          \
    }                                                                                                                \
  } while (false)

#define CANN_PROFILING_INFER_EVENT_END(request_id, info_type, single_event)                       \
  do {                                                                                            \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) { \
      const uint64_t prof_time = MsprofSysCycleTime();                                            \
      (single_event).timeStamp = prof_time;                                                       \
      (void)MsprofReportEvent(true, &(single_event));                                             \
    }                                                                                             \
  } while (false)

#define CANN_PROFILING_GRAPH_ID(prof_time, tid, graph_id, model_id, is_aging, model_name)                   \
  do {                                                                                                      \
    if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {           \
      (void)gert::GlobalProfilingWrapper::ReportGraphIdMap(prof_time, tid, {graph_id, model_id}, is_aging,  \
                                                           model_name);                                     \
    }                                                                                                       \
  } while (false)

#define CANN_PROFILING_STEP_TRACE(item_id, request_id, tag_id, stream)                         \
  do {                                                                                         \
    (void)gert::GlobalProfilingWrapper::ProfileStepTrace(request_id, item_id, tag_id, stream); \
  } while (false)
#define REGISTER_PROF_TYPE(type) const gert::ProfLaunchTypeRegistry type##prof_type_registry(#type, true)
#define REGISTER_PROF_NON_LAUNCH_TYPE(type) const gert::ProfLaunchTypeRegistry type##prof_type_registry(#type, false)
#define GE_ASSERT_MSPROF_OK(v, ...) \
  GE_ASSERT((((v) == MSPROF_ERROR_NONE) || ((v) == MSPROF_ERROR_UNINITIALIZE)), __VA_ARGS__)
#define RT2_PROFILING_SCOPE(element, event) gert::ScopeProfiler profiler((element), event)
#define RT2_PROFILING_SCOPE_CONST(element, event) const gert::ScopeProfiler profiler((element), (event))
#define RT2_PROFILING_SCOPE_ELEMENT(element) profiler.SetElement(element)
#define GRAPH_PROFILING_REG(api_id) const gert::GraphProfilingReporter profilingReporter(api_id)
#endif
