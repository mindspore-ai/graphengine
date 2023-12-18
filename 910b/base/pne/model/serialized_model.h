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
#ifndef BASE_PNE_MODEL_SERIALIZED_MODEL_H_
#define BASE_PNE_MODEL_SERIALIZED_MODEL_H_

#include "framework/pne/pne_model.h"

namespace ge {
class SerializedModel : public PneModel {
 public:
  using PneModel::PneModel;
  ~SerializedModel() override = default;

  Status SerializeModel(ModelBufferData &model_buff) override {
    model_buff.data = serialized_data_;
    model_buff.length = serialized_data_length_;
    return SUCCESS;
  }

  Status UnSerializeModel(const ModelBufferData &model_buff) override {
    serialized_data_ = model_buff.data;
    serialized_data_length_ = model_buff.length;
    return SUCCESS;
  }

  Status SetLogicDeviceId(const std::string &logic_device_id) override {
    logic_device_id_ = logic_device_id;
    return SUCCESS;
  }

  std::string GetLogicDeviceId() const override {
    return logic_device_id_;
  }

 private:
  std::shared_ptr<uint8_t> serialized_data_ = nullptr;
  uint64_t serialized_data_length_ = 0;
  std::string logic_device_id_;
};
}  // namespace ge
#endif  // BASE_PNE_MODEL_SERIALIZED_MODEL_H_
