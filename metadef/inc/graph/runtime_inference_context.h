/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_
#define INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_

#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "external/graph/ge_error_codes.h"
#include "external/graph/tensor.h"
#include "ge_attr_value.h"

namespace ge {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY RuntimeInferenceContext {
 public:
  static graphStatus GetContext(const std::string &context_id, RuntimeInferenceContext **ctx);
  static graphStatus CreateContext(const std::string &context_id);
  static void DestroyContext(const std::string &context_id);

  graphStatus SetTensor(int64_t node_id, int output_id, Tensor &&tensor);
  graphStatus GetTensor(int64_t node_id, int output_id, GeTensorPtr &tensor);
  graphStatus GetTensor(int64_t node_id, int output_id, Tensor &tensor);

 private:
  std::map<int64_t, std::vector<Tensor>> tensors_;
  std::map<int64_t, std::vector<GeTensorPtr>> ge_tensors_;
  std::mutex mu_;

  static std::map<std::string, std::unique_ptr<RuntimeInferenceContext>> contexts_;
  static std::mutex ctx_mu_;
};
} // namespace ge

#endif // INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_
