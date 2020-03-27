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

#ifndef GE_GE_RUNTIME_OP_INFO_UTILS_H_
#define GE_GE_RUNTIME_OP_INFO_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "cce/dnn.h"
#include "ge_runtime/op_info.h"
#include "graph/op_desc.h"
#include "common/ge_types.h"
#include "runtime/rt_model.h"

namespace ge {
namespace model_runner {

const uint32_t kNchwDimN = 0;
const uint32_t kNchwDimC = 1;
const uint32_t kNchwDimH = 2;
const uint32_t kNchwDimW = 3;

const uint32_t kNhwcDimN = 0;
const uint32_t kNhwcDimH = 1;
const uint32_t kNhwcDimW = 2;
const uint32_t kNhwcDimC = 3;

const uint32_t kHwckDimH = 0;
const uint32_t kHwckDimW = 1;
const uint32_t kHwckDimC = 2;
const uint32_t kHwckDimK = 3;

const string kNetOutPut = "NetOutput";

class OpInfoUtils {
 public:
  static bool InitTensorDescriptor(uint32_t format, uint32_t data_type, const std::vector<int64_t> &dim,
                                   cce::ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt = 0);
  static void DestroyTensorDescriptor(cce::ccTensorDescriptor_t &cc_tensor);
  static bool NeedTransFilter(const std::shared_ptr<OpInfo> &data_info);
  static bool TransFilterData(const std::shared_ptr<OpInfo> &data_info, const void *in_data, uint32_t length);
  static bool IsInputTensorNeedTrans(const std::shared_ptr<OpInfo> &data_info);
  static bool GetOutputSize(const std::shared_ptr<OpInfo> &op_info, std::vector<uint32_t> &output_size_list,
                            std::vector<uint32_t> &output_memory_size_list);

 private:
  static bool InitFilterTensorDescriptor(const std::vector<int64_t> &dims, uint32_t format, uint32_t dtype,
                                         cce::ccFilterDescriptor_t &cc_tensor);
  static void TransDataHWCK2KCHW(const void *input, int64_t H, int64_t W, int64_t C, int64_t K, void **output);
  static void DestroyFilterDescriptor(cce::ccFilterDescriptor_t &cc_filter);
  static bool IsComputDimsSize(const uint32_t format, const uint32_t real_dim_cnt);
  static void TransferDim(const std::vector<int64_t> &dim, std::vector<int64_t> &dim_vector);
  static bool InitTensorNdDescriptor(uint32_t data_type, const std::vector<int64_t> &dim,
                                     cce::ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt);
  static bool InitTensorPoolingMaskDescriptor(uint32_t format, uint32_t data_type, const std::vector<int64_t> &dim,
                                              cce::ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt);
  static bool InitTensor6dDescriptor(uint32_t format, uint32_t data_type, const std::vector<int64_t> &dim,
                                     cce::ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt);
  static bool InitTensor4dDescriptor(uint32_t format, uint32_t data_type, cce::ccTensorDescriptor_t &cc_tensor,
                                     int32_t n, int32_t c, int32_t h, int32_t w, uint32_t real_dim_cnt);
  static bool CheckParam(uint32_t format, uint32_t data_type, const std::vector<int64_t> &dim_vector);
};
}  // namespace model_runner
}  // namespace ge

#endif  // GE_GE_RUNTIME_OP_INFO_UTILS_H_
