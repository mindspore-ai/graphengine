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

#ifndef GE_GRAPH_PASSES_AICPU_CONSTANT_FOLDING_PASS_H_
#define GE_GRAPH_PASSES_AICPU_CONSTANT_FOLDING_PASS_H_

#include <vector>
#include <string>

#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/passes/folding_pass.h"

namespace ge {
class AicpuConstantFoldingPass : public FoldingPass {
 public:
  Status Run(ge::NodePtr &node) override;

 private:
  enum AddrType { kData = 0, kSummary = 1, kTypeEnd };

  struct AddrAndType {
    uint64_t input_addr;
    AddrType attr_type;
  } __attribute__((packed));

  struct DataPtrInfo {
    uint64_t release_flag;
    uint64_t data_size;
    uint64_t src_ptr;
    uint64_t dst_ptr;
  } __attribute__((packed));
  bool CheckInput(const ge::NodePtr &node, vector<ConstGeTensorPtr> &weight_vec);
  Status GetInputAddrs(const vector<ConstGeTensorPtr> &weight_vec, vector<AddrAndType> &input_addrs);
  Status GetOutputAddrs(const OpDescPtr &node_desc, vector<uint64_t> &output_addrs);
  Status GenerateTaskForLaunch(STR_FWK_OP_KERNEL &aicpu_task, void *&task_buf) const;
  Status GenerateDataPtrInfo(const vector<uint64_t> &output_addrs, vector<DataPtrInfo> &data_vec,
                             vector<uint64_t> &data_infos);
  Status GenerateGeTensor(const OpDescPtr &node_desc, const vector<DataPtrInfo> &data_vec,
                          vector<GeTensorPtr> &outputs);
  Status UpdateWorkSpaceAddr(string &task_info, STR_FWK_OP_KERNEL &task) const;
  Status UpdateInputAndOutputAddr(const vector<uint64_t> &io_addrs, STR_FWK_OP_KERNEL &task) const;
  Status UpdateSingleOpAddr(string &task_info, const vector<AddrAndType> &input_addrs,
                            const vector<uint64_t> &outputs_addr_vec, STR_FWK_OP_KERNEL &task);
  Status UpdateMemCopyAddr(string &task_info, const vector<uint64_t> &data_infos, vector<uint64_t> &internal_addrs,
                           STR_FWK_OP_KERNEL &task);
  Status LaunchSingleOpRunTask(const NodePtr &node, const vector<AddrAndType> &input_addrs,
                               const vector<uint64_t> &output_addrs);
  Status LaunchMemCopyTask(const vector<uint64_t> &data_infos);
  void ReleaseMemory(const vector<AddrAndType> &input_addrs, const vector<uint64_t> &output_addrs,
                     const vector<DataPtrInfo> &data_vec) const;
  Status KernelLaunch(void *aicpu_task) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_AICPU_CONSTANT_FOLDING_PASS_H_
