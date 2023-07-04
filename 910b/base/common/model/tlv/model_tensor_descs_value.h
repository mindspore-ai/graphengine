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

#ifndef MODEL_TENSOR_DESCS_VALUE_
#define MODEL_TENSOR_DESCS_VALUE_

#include "base_tlv_block.h"
#include "graph/def_types.h"
#include "framework/common/ge_model_inout_types.h"

namespace ge {
class ModelTensorDesc {
public:
  friend class ModelIntroduction;
  size_t Size();
  bool Serilize(uint8_t ** const addr, size_t &left_size);
private:
  ModelTensorDescBaseInfo base_info;
  std::string name;
  std::vector<int64_t> dims;
  std::vector<int64_t> dimsV2;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
};

class ModelTensorDescsValue : public BaseTlvBlock {
public:
  friend class ModelIntroduction;
  size_t Size() override;
  bool Serilize(uint8_t ** const addr, size_t &left_size) override;
  bool NeedSave() override;
  virtual ~ModelTensorDescsValue() = default;
private:
  uint32_t tensor_desc_size = 0U;
  std::vector<ModelTensorDesc> descs;
};
}
#endif