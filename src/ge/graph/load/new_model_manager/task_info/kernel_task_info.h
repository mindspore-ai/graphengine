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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/load/new_model_manager/task_info/task_info.h"
#include "graph/op_desc.h"
namespace ge {
class KernelTaskInfo : public TaskInfo {
 public:
  friend class DavinciModel;

  KernelTaskInfo()
      : ctx_(),
        stub_func_(nullptr),
        args_(nullptr),
        sm_desc_(nullptr),
        flowtable_(nullptr),
        block_dim_(0),
        args_size_(0),
        flowtable_size_(0),
        task_id_(0),
        so_name_(""),
        kernel_name_(""),
        kernel_type_(cce::ccKernelType::CCE_AI_CORE),
        dump_flag_(RT_KERNEL_DEFAULT),
        dump_args_(nullptr),
        davinci_model_(nullptr) {}

  ~KernelTaskInfo() override {
    davinci_model_ = nullptr;
    stub_func_ = nullptr;
    sm_desc_ = nullptr;
    flowtable_ = nullptr;
    args_ = nullptr;
  }

  Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) override;

  Status Distribute() override;

  Status Release() override;

  cce::ccOpContext *GetCtx() override { return &ctx_; }

  uint32_t GetTaskID() override { return task_id_; }

  uintptr_t GetDumpArgs() override {
    auto ret = reinterpret_cast<uintptr_t >(dump_args_);
    return ret;
  }

  cce::ccOpContext ctx_;

 private:
  Status InitTVMTask(DavinciModel *davinci_model, uint16_t offset, const domi::KernelDef &kernel_def);

  Status InitAICPUCustomTask(const std::map<uint32_t, std::shared_ptr<OpDesc>> &op_list, uint32_t op_index,
                             const domi::KernelDef &kernel_def);

  Status InitCceTask(DavinciModel *davinci_model, const domi::KernelDef &kernel_def);

  Status InitAicpuTask(const std::map<uint32_t, OpDescPtr> &op_list, uint32_t op_index,
                       const domi::KernelDef &kernel_def);

  Status StoreInputOutputTensor(const std::vector<void *> &input_data_addrs,
                                const std::vector<void *> &output_data_addrs,
                                const std::vector<::tagCcAICPUTensor> &input_descs,
                                const std::vector<::tagCcAICPUTensor> &output_descs);

  Status SetContext(const domi::KernelDef &kernel_def);

  Status UpdateCceArgs(std::string &sm_desc, std::string &flowtable, DavinciModel *davinci_model,
                       const domi::KernelDef &kernel_def);

  Status SetFlowtable(std::string &flowtable, const domi::KernelDef &kernel_def);

  static void FreeRtMem(void **ptr);

  void *stub_func_;
  void *args_;
  void *sm_desc_;
  void *flowtable_;
  uint32_t block_dim_;
  uint32_t args_size_;
  uint32_t flowtable_size_;
  uint32_t task_id_;
  std::string so_name_;
  std::string kernel_name_;
  cce::ccKernelType kernel_type_;
  uint32_t dump_flag_;
  void *dump_args_;
  DavinciModel *davinci_model_;

  struct AICPUCustomInfo {
    void *input_descs = nullptr;
    void *input_addrs = nullptr;
    void *output_descs = nullptr;
    void *output_addrs = nullptr;
    void *attr_handle = nullptr;
  } custom_info_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_
