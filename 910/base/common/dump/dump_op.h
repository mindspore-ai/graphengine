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

#ifndef GE_COMMON_DUMP_DUMP_OP_H_
#define GE_COMMON_DUMP_DUMP_OP_H_

#include <string>

#include "graph/op_desc.h"
#include "common/dump/dump_properties.h"
#include "proto/op_mapping.pb.h"
#include "runtime/stream.h"
#include "runtime/mem.h"

namespace ge {
class DumpOp {
 public:
  DumpOp() = default;
  ~DumpOp();

  void SetDumpInfo(const DumpProperties &dump_properties, const OpDescPtr &op_desc,
                   const std::vector<uintptr_t> &input_addrs, const std::vector<uintptr_t> &output_addrs,
                   rtStream_t const stream);
  Status LaunchDumpOp(bool is_single_op_dump);
  void SetLoopAddr(const uintptr_t global_step, const uintptr_t loop_per_iter, const uintptr_t loop_cond);
  void SetDynamicModelInfo(const std::string &dynamic_model_name, const std::string &dynamic_om_name,
                           const uint32_t dynamic_model_id);
  void SetTaskId(uint32_t task_id) {
    task_id_ = task_id;
  }
  void SetWorkspaceAddrs(const std::vector<uint64_t> &workspace_addr) {
    space_addrs_.assign(workspace_addr.cbegin(), workspace_addr.cend());
  }
  void SetStreamId(const uint32_t stream_id) {
    stream_id_ = stream_id;
  }
  Status UpdateAddrs(const std::vector<uintptr_t> &input_addrs,
                     const std::vector<uintptr_t> &output_addrs);
 private:
  Status ExecutorDumpOp();
  void DumpWorkspace(toolkit::aicpu::dump::Task &task);
  Status DumpOutput(toolkit::aicpu::dump::Task &task);
  Status DumpInput(toolkit::aicpu::dump::Task &task);
  Status SetDumpModelName();
  Status ProtoMallocAndMemcpy(const size_t proto_size, const std::string &proto_msg);
  Status LaunchDump(toolkit::aicpu::dump::Task &task);

  DumpProperties dump_properties_;
  OpDescPtr op_desc_;
  std::vector<uintptr_t> input_addrs_;
  std::vector<uintptr_t> output_addrs_;
  std::vector<uint64_t> space_addrs_;

  void *proto_dev_mem_ = nullptr;
  void *proto_size_dev_mem_ = nullptr;
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info_;
  rtStream_t stream_;
  uintptr_t global_step_ = 0U;
  uintptr_t loop_per_iter_ = 0U;
  uintptr_t loop_cond_ = 0U;
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  std::string dynamic_model_name_;
  std::string dynamic_om_name_;
  std::uint32_t dynamic_model_id_;
};
}  // namespace ge

#endif  // GE_COMMON_DUMP_DUMP_OP_H_
