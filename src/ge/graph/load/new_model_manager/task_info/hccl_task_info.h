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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_

#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include "common/opskernel/ge_task_info.h"
#include "graph/load/new_model_manager/task_info/task_info.h"
#include "graph/manager/util/hcom_util.h"
namespace ge {
class HcclTaskInfo : public TaskInfo {
 public:
  HcclTaskInfo()
      : davinci_model_(nullptr),
        hccl_type_(""),
        input_data_addr_(nullptr),
        output_data_addr_(nullptr),
        count_(0),
        data_type_(HCCL_DATA_TYPE_INT8),
        op_type_(HCCL_REP_OP_SUM),
        root_id_(0),
        id_(0),
        workspace_addr_(nullptr),
        workspace_mem_size_(0),
        hccl_stream_list_(),
        ops_kernel_store_(nullptr),
        private_def_(nullptr),
        private_def_len_(0) {}

  ~HcclTaskInfo() override;

  ge::Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  ge::Status Distribute() override;

  uint32_t GetTaskID() override { return id_; }

 private:
  ge::Status SetAddrs(const std::string &hccl_type, const std::shared_ptr<OpDesc> &op);

  void TransToGETaskInfo(GETaskInfo &ge_task);

  void GetPrivateDefByTaskDef(const domi::TaskDef &task);

  void ReuseStream(int64_t stream_num, DavinciModel *davinci_model);

  ge::Status CreateStream(int64_t stream_num, DavinciModel *davinci_model);

  DavinciModel *davinci_model_;
  string hccl_type_;
  void *input_data_addr_;
  void *output_data_addr_;
  int32_t count_;
  hcclDataType_t data_type_;
  hcclRedOp_t op_type_;
  int64_t root_id_;
  uint32_t id_;
  void *workspace_addr_;
  uint64_t workspace_mem_size_;
  vector<rtStream_t> hccl_stream_list_;
  void *ops_kernel_store_;
  void *private_def_;
  uint32_t private_def_len_;
  static std::mutex hccl_follow_stream_mutex_;
  static uint32_t max_node_of_hccl_stream_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_
