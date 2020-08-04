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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_DUMPER_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_DUMPER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "proto/ge_ir.pb.h"
#include "proto/op_mapping_info.pb.h"
#include "runtime/mem.h"
#include "task_info/task_info.h"

namespace ge {
class DataDumper {
 public:
  DataDumper()
      : model_name_(),
        model_id_(0),
        runtime_param_(),
        dev_mem_load_(nullptr),
        dev_mem_unload_(nullptr),
        op_list_(),
        input_map_(),
        load_flag_(false),
        device_id_(0),
        global_step_(0),
        loop_per_iter_(0),
        loop_cond_(0) {}

  ~DataDumper();

  void SetModelName(const std::string &model_name) { model_name_ = model_name; }

  void SetModelId(uint32_t model_id) { model_id_ = model_id; }

  void SetMemory(const RuntimeParam &runtime_param) { runtime_param_ = runtime_param; }

  void SetDeviceId(uint32_t device_id) { device_id_ = device_id; }

  void SetLoopAddr(void *global_step, void *loop_per_iter, void *loop_cond);

  void SaveDumpInput(const std::shared_ptr<Node> &node);

  // args is device memory stored first output addr
  void SaveDumpTask(uint32_t task_id, uint32_t stream_id, const std::shared_ptr<OpDesc> &op_desc, uintptr_t args);
  void SaveEndGraphId(uint32_t task_id, uint32_t stream_id);

  void SetOmName(const std::string &om_name) { om_name_ = om_name; }

  Status LoadDumpInfo();

  Status UnloadDumpInfo();

 private:
  void ReleaseDevMem(void **ptr) noexcept;

  void PrintCheckLog(string &dump_list_key);

  std::string model_name_;

  // for inference data dump
  std::string om_name_;

  uint32_t model_id_;
  RuntimeParam runtime_param_;
  void *dev_mem_load_;
  void *dev_mem_unload_;

  struct InnerDumpInfo;
  struct InnerInputMapping;

  std::vector<InnerDumpInfo> op_list_;
  uint32_t end_graph_task_id_ = 0;
  uint32_t end_graph_stream_id_ = 0;
  std::multimap<std::string, InnerInputMapping> input_map_;
  bool load_flag_;
  uint32_t device_id_;
  uintptr_t global_step_;
  uintptr_t loop_per_iter_;
  uintptr_t loop_cond_;

  Status DumpOutput(const InnerDumpInfo &inner_dump_info, aicpu::dump::Task &task);
  Status DumpInput(const InnerDumpInfo &inner_dump_info, aicpu::dump::Task &task);
  Status ExecuteLoadDumpInfo(aicpu::dump::OpMappingInfo &op_mapping_info);
  void SetEndGraphIdToAicpu(uint32_t task_id, uint32_t stream_id, aicpu::dump::OpMappingInfo &op_mapping_info);
  Status ExecuteUnLoadDumpInfo(aicpu::dump::OpMappingInfo &op_mapping_info);
};
struct DataDumper::InnerDumpInfo {
  uint32_t task_id;
  uint32_t stream_id;
  std::shared_ptr<OpDesc> op;
  uintptr_t args;
  bool is_task;
  int input_anchor_index;
  int output_anchor_index;
  std::vector<int64_t> dims;
  int64_t data_size;
};

struct DataDumper::InnerInputMapping {
  std::shared_ptr<OpDesc> data_op;
  int input_anchor_index;
  int output_anchor_index;
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_DUMPER_H_
