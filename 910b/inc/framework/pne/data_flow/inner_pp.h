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

#ifndef INC_PNE_DATA_FLOW_INNER_PP_H_
#define INC_PNE_DATA_FLOW_INNER_PP_H_

#include <map>
#include "ge/ge_api_error_codes.h"
#include "flow_graph/process_point.h"

namespace ge {
namespace dflow {
class InnerPpImpl;
class GE_FUNC_VISIBILITY InnerPp : public ProcessPoint {
 public:
  InnerPp(const char_t *pp_name, const char_t *inner_type);
  ~InnerPp() override = default;
  void Serialize(ge::AscendString &str) const override;

 protected:
  virtual void InnerSerialize(std::map<ge::AscendString, ge::AscendString> &serialize_map) const = 0;

 private:
  std::shared_ptr<InnerPpImpl> impl_;
};
}  // namespace dflow
}  // namespace ge
#endif  // INC_PNE_DATA_FLOW_INNER_PP_H_