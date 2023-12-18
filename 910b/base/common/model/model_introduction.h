/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef INC_FRAMEWORK_COMMON_HELPER_MODEL_INTRODUCTION_H_
#define INC_FRAMEWORK_COMMON_HELPER_MODEL_INTRODUCTION_H_

#include "common/model/tlv/base_tlv_block.h"
#include "common/model/tlv/model_tensor_descs_value.h"
#include "common/model/tlv/vec_int_value.h"
#include "common/model/tlv/vec_int_int_value.h"
#include "common/model/tlv/vec_str_value.h"
#include "framework/common/ge_types.h"
#include "ge/ge_api_error_codes.h"
#include "common/model/ge_model.h"

namespace ge {
struct ModelDesc {
  ModelTensorDescsValue inputDesc;
  ModelTensorDescsValue outputDesc;
  vecIntValue dynamicBatch;
  vecIntIntValue dynamicHW;
  vecIntIntValue dynamicDims;
  vecStrValue dynamicOutputShape;
  vecStrValue dataNameOrder;
};

class ModelIntroduction {
 public:
  Status Init(const GeModelPtr &ge_model);
  std::shared_ptr<uint8_t> Data();
  uint32_t DataSize();
  ~ModelIntroduction() = default;

 private:
  Status ConstructInputInfo();
  Status ConstructOutputInfo();
  Status ConstructDynamicInfo();
  void ConstructNameOrder();
  void ConstructDynamicOutShape();

  static Status CreateOutput(const uint32_t index, const OpDescPtr &op_desc, ModelTensorDesc &output);
  static Status CreateInputDimsInfo(const OpDescPtr &op_desc, ModelTensorDesc &model_tensor_desc);
  Status GetDynamicInfoFromCase(int32_t &dynamic_type, std::vector<std::vector<int64_t>> &batch_info);
  void TlvBlockSize(BaseTlvBlock &tlv_block);
  static Status SaveTlvBlock(BaseTlvBlock &tlv_block, const ModelDescType type, uint8_t **const write_addr,
                             size_t &left_size);

  uint32_t total_size_ = 0;
  std::shared_ptr<uint8_t> buff_;
  // input list use map to keep input_desc in order by index
  std::map<uint32_t, OpDescPtr> input_op_list_;
  std::vector<OpDescPtr> output_op_list_;
  OpDescPtr case_desc_;
  std::vector<std::string> out_node_name_;
  ModelDesc modelIntroduction_;
};
}  // namespace ge
#endif