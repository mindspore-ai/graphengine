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

#ifndef INC_EXTERNAL_GE_GE_CONTINUOUS_TENSOR_LIST_API_H
#define INC_EXTERNAL_GE_GE_CONTINUOUS_TENSOR_LIST_API_H
#include <memory>
#include <vector>
#include "graph/tensor.h"
#include "ge_error_codes.h"

namespace ge {
class ContinuousTensorListImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ContinuousTensorList {
 public:
  ContinuousTensorList() = default;
  ~ContinuousTensorList();
  explicit ContinuousTensorList(const std::vector<TensorDesc> &tensor_desc_list);

  graphStatus Initialize();
  graphStatus Finalize();

  const std::vector<TensorDesc> &GetTensorDescList() const;
  const std::vector<Tensor> &GetTensorList() const;

  std::vector<uint8_t *> GetDataList() const;
  std::vector<size_t> GetSizeList() const;
  size_t GetBufferSize() const;
  const void *GetBuffer() const;

 private:
  std::shared_ptr<ContinuousTensorListImpl> impl_;
};
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GE_CONTINUOUS_TENSOR_LIST_API_H
