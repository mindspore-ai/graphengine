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

#ifndef INC_GRAPH_UTILS_TENSOR_ADAPTER_H_
#define INC_GRAPH_UTILS_TENSOR_ADAPTER_H_

#include <memory>
#include "graph/ge_tensor.h"
#include "graph/tensor.h"

namespace ge {
using GeTensorPtr = std::shared_ptr<GeTensor>;
using ConstGeTensorPtr = std::shared_ptr<const GeTensor>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TensorAdapter {
 public:
  static GeTensorDesc TensorDesc2GeTensorDesc(const TensorDesc &tensorDesc);
  static TensorDesc GeTensorDesc2TensorDesc(const GeTensorDesc &geTensorDesc);
  static GeTensorPtr Tensor2GeTensor(const Tensor &tensor);
  static Tensor GeTensor2Tensor(const ConstGeTensorPtr &geTensor);

  static ConstGeTensorPtr AsGeTensorPtr(const Tensor &tensor);  // Share value
  static GeTensorPtr AsGeTensorPtr(Tensor &tensor);             // Share value
  static const GeTensor AsGeTensor(const Tensor &tensor);       // Share value
  static GeTensor AsGeTensor(Tensor &tensor);                   // Share value
  static const Tensor AsTensor(const GeTensor &tensor);         // Share value
  static Tensor AsTensor(GeTensor &tensor);                     // Share value
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_TENSOR_ADAPTER_H_
