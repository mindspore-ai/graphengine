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

namespace ge {
struct ExtraOpInfo {
  std::string node_info;
  std::string tiling_data;
  std::string args_before_execute;
  bool has_memory_log{false};
  uint32_t tiling_key{0U};
  uintptr_t args{0U};
  size_t args_size{0UL};
  std::vector<void *> input_addrs;
  std::vector<void *> output_addrs;
  std::vector<void *> space_addrs;
  std::vector<int64_t> workspace_bytes{};
  bool is_host_args{false};
};

class ExceptionDumper {
 public:
  ExceptionDumper() noexcept = default;
  ~ExceptionDumper();

  void SaveDumpOpInfo(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id);
  void SaveDumpOpInfo(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const uint32_t task_id,
                      const uint32_t stream_id);
  Status DumpExceptionInfo(const std::vector<rtExceptionInfo> &exception_infos);
  void LogExceptionTvmOpInfo(const OpDescInfo &op_desc_info) const;
  void LogExceptionArgs(const OpDescInfo &op_desc_info) const;
  bool GetOpDescInfo(const uint32_t stream_id, const uint32_t task_id, OpDescInfo &op_desc_info,
                     const uint32_t context_id = UINT32_MAX, const uint32_t thread_id = UINT32_MAX);
  OpDescInfo *MutableOpDescInfo(const uint32_t task_id, const uint32_t stream_id);

  static Status DumpDevMem(const ge::char_t * const file, const void * const addr, const int64_t size);

  static void Reset(ExtraOpInfo &extra_op_info);

  const std::vector<OpDescInfo> &GetSavedOpDescInfo() const {
    return op_desc_info_;
  }

  void Clear() {
    const std::lock_guard<std::mutex> lock(mutex_);
    op_desc_info_.clear();
  }

 private:
  void RefreshAddrs(OpDescInfo &op_desc_info) const;
  void SaveOpDescInfo(const OpDescPtr &op, OpDescInfo &op_desc_info, const OpDescInfoId &id) const;
  Status DumpExceptionInput(const OpDescInfo &op_desc_info, const std::string &dump_file) const;
  Status DumpExceptionOutput(const OpDescInfo &op_desc_info, const std::string &dump_file) const;
  Status DumpExceptionWorkspace(const OpDescInfo &op_desc_info, const std::string &dump_file) const;
  void SaveDumpOpInfoHelper(const OpDescPtr &op, const ExtraOpInfo &extra_op_info, const OpDescInfoId &id);
  std::mutex mutex_;
  std::vector<OpDescInfo> op_desc_info_;
  size_t op_desc_info_idx_{0UL};
};
}  // namespace ge

#endif // GE_COMMON_DUMP_EXCEPTION_DUMPER_H_
