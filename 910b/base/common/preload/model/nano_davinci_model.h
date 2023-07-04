/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_
#define GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_
#include "common/preload/model/pre_davinci_model.h"

namespace ge {
class NanoDavinciModel : public PreDavinciModel {
 public:
  NanoDavinciModel() = default;
  virtual ~NanoDavinciModel() = default;
  Status Init() override;
  Status DoPartitionProcess() override;
  Status InitZeroCopyInfo();
  Status GenZeroCopyTable(const OpDescPtr &op_desc, uint32_t &search_id, const bool is_input);
  std::unordered_map<int64_t, uint32_t> GetZeroCopyInfo() { return zero_copy_offset_to_ids_; }
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_