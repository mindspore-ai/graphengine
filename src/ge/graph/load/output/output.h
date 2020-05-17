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

#ifndef GE_GRAPH_LOAD_OUTPUT_OUTPUT_H_
#define GE_GRAPH_LOAD_OUTPUT_OUTPUT_H_

#include <string>
#include <vector>

#include "common/debug/log.h"
#include "common/op/attr_define.h"
#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "common/op/op_parser_util.h"
#include "common/types.h"
#include "common/util.h"
#include "common/ge_types.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/op_desc.h"

namespace ge {
using std::string;
using std::vector;

// The base class for all op
class Output {
 public:
  Output(const OpDescPtr &op_desc, DavinciModel *model);
  virtual ~Output();

  ///
  /// @ingroup domi
  /// @brief Initialize input/output params
  /// @return Status
  ///
  virtual Status Init();

  ///
  /// @ingroup domi
  /// @brief Copy Op Output to user space.
  /// @brief when model running, Add one DataOp as input node, Add one Output Op as output node.
  /// @return Status
  ///
  virtual Status CopyResult(OutputData &rslt, uint32_t data_begin, uint32_t &data_index, bool support_mem_share);

  ///
  /// @ingroup domi
  /// @brief Trans Output data to fp16
  /// @return Status
  ///
  Status SetDataBuf(DataBuffer &data_buf, uint32_t &data_count, size_t i, bool support_mem_share);

  ///
  /// @ingroup domi
  /// @brief Get Output data and size.
  /// @return void
  ///
  void GetOutputData(vector<void *> &v_data_addr, vector<int64_t> &v_data_size);

  // Copy assignment operator and copy constructor are deleted
  Output &operator=(const Output &output) = delete;
  Output(const Output &output) = delete;

 protected:
  // Model's base address
  uint8_t *base_;
  uint8_t *var_base_;
  uint64_t logic_base_;
  uint64_t logic_var_base_;
  // The DavinciModel which ops belong to
  DavinciModel *model_;

  ConstOpDescPtr op_desc_;

  // Input descriptions
  size_t input_num_;
  vector<void *> v_input_data_addr_;  // init as:buf_base + op_def_->input(i));
  vector<int64_t> v_input_size_;
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_OUTPUT_OUTPUT_H_
