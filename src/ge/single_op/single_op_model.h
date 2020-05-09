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

#ifndef GE_SINGLE_OP_SINGLE_OP_MODEL_H_
#define GE_SINGLE_OP_SINGLE_OP_MODEL_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/helper/model_helper.h"
#include "graph/load/new_model_manager/davinci_model_parser.h"
#include "single_op/single_op.h"
#include "single_op/stream_resource.h"

namespace ge {
struct SingleOpModelParam {
  uint64_t base_addr = 0;
  uint64_t memory_size = 0;
  uint64_t weight_addr = 0;
  uint64_t weight_size = 0;

  uint8_t *mem_base = nullptr;
  uint8_t *weight_base = nullptr;

  std::map<uintptr_t, int> addr_mapping_;
  int64_t core_type = 0;
};

class SingleOpModel {
 public:
  SingleOpModel(const std::string &model_name, const void *model_data, uint32_t model_size);
  ~SingleOpModel() = default;

  Status Init();
  Status BuildOp(StreamResource &resource, SingleOp &single_op);

 private:
  Status InitModel();
  Status ParseInputsAndOutputs();
  Status SetInputsAndOutputs(SingleOp &single_op);

  Status InitModelMem(StreamResource &resource);

  Status ParseInputNode(const OpDescPtr &op_desc);
  void ParseOutputNode(const OpDescPtr &op_desc);

  Status BuildTaskList(SingleOp &single_op);
  Status BuildKernelTask(const domi::KernelDef &kernel_def, SingleOp &single_op, OpTask **task);

  static void ParseOpModelParams(ModelHelper &model_helper, SingleOpModelParam &param);
  void ParseArgTable(TbeOpTask *task, SingleOp &op);

  std::string model_name_;
  const void *ori_model_data_;
  uint32_t ori_model_size_;

  ModelHelper model_helper_;

  map<uint32_t, OpDescPtr> op_list_;
  SingleOpModelParam model_params_;

  std::vector<ptrdiff_t> input_offset_list_;
  std::vector<size_t> input_sizes_;
  std::vector<ptrdiff_t> output_offset_list_;
  std::vector<size_t> output_sizes_;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_SINGLE_OP_MODEL_H_
