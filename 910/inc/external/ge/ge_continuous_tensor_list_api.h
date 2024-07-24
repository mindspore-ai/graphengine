/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
