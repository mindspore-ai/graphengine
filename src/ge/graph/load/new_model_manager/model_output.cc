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

#include "graph/load/new_model_manager/model_output.h"

#include <memory>
#include <string>

#include "common/debug/log.h"
#include "common/op/ge_op_utils.h"
#include "graph/load/new_model_manager/davinci_model.h"
#include "graph/load/output/output.h"

namespace ge {
Status ModelOutput::CopyResult(DavinciModel *model, OpDescPtr op_desc, OutputData &rslt, uint32_t &data_index,
                               bool support_mem_share) {
  uint32_t data_begin = data_index;
  std::shared_ptr<Output> model_output = MakeShared<Output>(op_desc, model);
  if (model_output == nullptr) {
    return INTERNAL_ERROR;
  }

  if (model_output->Init() != SUCCESS) {
    return INTERNAL_ERROR;
  }

  return model_output->CopyResult(rslt, data_begin, data_index, support_mem_share);
}
}  // namespace ge
