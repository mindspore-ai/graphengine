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

#ifndef INC_PNE_DATA_FLOW_MODEL_PP_H_
#define INC_PNE_DATA_FLOW_MODEL_PP_H_

#include "pne/data_flow/inner_pp.h"
#include "common/ge_visibility.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
namespace dflow {
class ModelPpImpl;
class GE_FUNC_VISIBILITY ModelPp : public InnerPp {
 public:
  ModelPp(const char_t *pp_name, const char_t *model_path);
  ~ModelPp() override = default;

 protected:
  void InnerSerialize(std::map<ge::AscendString, ge::AscendString> &serialize_map) const override;

 private:
  std::shared_ptr<ModelPpImpl> impl_;
};

}  // namespace dflow
}  // namespace ge
#endif  // INC_PNE_DATA_FLOW_MODEL_PP_H_