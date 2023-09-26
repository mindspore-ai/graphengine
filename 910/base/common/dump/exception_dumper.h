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

#ifndef GE_COMMON_DUMP_EXCEPTION_DUMPER_H_
#define GE_COMMON_DUMP_EXCEPTION_DUMPER_H_

#include <vector>
#include <mutex>

#include "graph/op_desc.h"
#include "framework/common/ge_types.h"
#include "runtime/base.h"
#include "common/dump/dump_properties.h"
#include "exe_graph/runtime/dfx_info_filler.h"
#include "common/dump/kernel_tracing_utils.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
struct ExtraOpInfo {
  std::string node_info;
  std::string tiling_data;
  std::string args_before_execute;
  uint32_t tiling_key{0U};
  uintptr_t args{0U};
  size_t args_size{0UL};
  std::vector<void *> input_addrs;
  std::vector<void *> output_addrs;
  std::vector<uint64_t> input_sizes;
  std::vector<uint64_t> output_sizes;
  bool is_host_args{false};
  std::vector<std::pair<uintptr_t, int64_t>> workspace_info{};
  void DebugLogString() {
    std::stringstream ss;
    ss << "Node Info: " << node_info;
    GELOGD("%s", ss.str().c_str());
    ss.str("");
    ss << "Tiling key: " << tiling_key;
    ss << "Tiling data: " << tiling_data;
    GELOGD("%s", ss.str().c_str());
    ss.str("");
    ss << "Args before execute: " << args_before_execute;
    ss << "Args addr: " << args;
    ss << "Args size: " << args_size;
    GELOGD("%s", ss.str().c_str());
    ss.str("");
    for (const auto &ele : workspace_info) {
      ss << "Workspace addr: " << ele.first;
      ss << "Workspace size: " << ele.second;
    }
    GELOGD("%s", ss.str().c_str());
  }
};

class ExecutorExceptionDumpInfoWrapper : public gert::ExceptionDumpInfoWrapper {
 public:
  ExecutorExceptionDumpInfoWrapper() = default;
  explicit ExecutorExceptionDumpInfoWrapper(ge::ExtraOpInfo *dump_unit) : dump_unit_(dump_unit) {}

  void SetTilingData(uintptr_t addr, size_t size) override {
    std::stringstream ss;
    gert::PrintHex(reinterpret_cast<uint8_t *>(addr), size, ss);
    dump_unit_->tiling_data = ss.str();
  }

  void SetTilingKey(uint32_t key) override { dump_unit_->tiling_key = key; }

  void SetHostArgs(uintptr_t addr, size_t size) override {
    std::stringstream ss;
    ss << "args before execute: ";
    gert::PrintHex(reinterpret_cast<void **>(addr), size / sizeof(void *), ss);
    dump_unit_->args_before_execute = ss.str();
    dump_unit_->args = addr;
    dump_unit_->args_size = size;
    dump_unit_->is_host_args = true;
  }

  void SetDeviceArgs(uintptr_t addr, size_t size) override {
    (void)addr;
    (void)size;
  }

  void AddWorkspace(uintptr_t addr, int64_t bytes) override {
    dump_unit_->workspace_info.emplace_back(addr, bytes);
  }

 private:
  ExtraOpInfo *dump_unit_;
};

class ExceptionDumper {
 public:
  ExceptionDumper() noexcept = default;
  ~ExceptionDumper();

  void SaveDumpOpInfo(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id);
  void SaveDumpOpInfo(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const uint32_t task_id,
                      const uint32_t stream_id);
  void SaveInputOutputInfo(const bool is_input, const OpDescPtr &op, OpDescInfo &op_desc_info) const;
  Status DumpExceptionInfo(const std::vector<rtExceptionInfo> &exception_infos);
  void LogExceptionTvmOpInfo(const OpDescInfo &op_desc_info) const;
  void LogExceptionArgs(const OpDescInfo &op_desc_info) const;
  bool GetOpDescInfo(const uint32_t stream_id, const uint32_t task_id, OpDescInfo &op_desc_info,
                     const uint32_t context_id = UINT32_MAX, const uint32_t thread_id = UINT32_MAX);
  OpDescInfo *MutableOpDescInfo(const uint32_t task_id, const uint32_t stream_id);

  static Status DumpDevMem(const ge::char_t * const file, const void * const addr, const int64_t size);

  static void Reset(ExtraOpInfo &extra_op_info);

  Status DumpNodeInfo(const OpDescInfo &op_desc_info, const std::string &file_path,
                      const bool is_exception, const bool is_ffts_plus,
                      const ge::DumpProperties &dump_properties) const;

  void Clear() {
    const std::lock_guard<std::mutex> lock(mutex_);
    op_desc_info_.clear();
  }

 private:
  void RefreshAddrs(OpDescInfo &op_desc_info) const;
  void SaveOpDescInfo(const OpDescPtr &op, OpDescInfo &op_desc_info, const OpDescInfoId &id) const;
  Status DumpExceptionInput(const OpDescInfo &op_desc_info, const std::string &dump_file,
                            const bool is_exception, const ge::DumpProperties &dump_properties) const;
  Status DumpExceptionOutput(const OpDescInfo &op_desc_info, const std::string &dump_file,
                             const bool is_exception, const ge::DumpProperties &dump_properties) const;
  Status DumpExceptionWorkspace(const OpDescInfo &op_desc_info, const std::string &dump_file,
                                const bool is_exception, const ge::DumpProperties &dump_properties) const;
  void SaveDumpOpInfoHelper(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id);
  std::mutex mutex_;
  std::vector<OpDescInfo> op_desc_info_;
  size_t op_desc_info_idx_{0UL};
};
}  // namespace ge

#endif // GE_COMMON_DUMP_EXCEPTION_DUMPER_H_
