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

#include "ge_runtime/op_info_utils.h"

#include <list>
#include <memory>

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "runtime/rt_model.h"

namespace ge {
namespace model_runner {
static const uint32_t kDimMaxSize = 8;
static const uint32_t kPoolMaskDescDimSize = 6;
static const uint32_t kPoolMaskDescWinH = 4;
static const uint32_t kPoolMaskDescWinW = 5;

bool OpInfoUtils::NeedTransFilter(const std::shared_ptr<OpInfo> &data_info) {
  if (data_info == nullptr) {
    GELOGE(PARAM_INVALID, "data info is null.");
    return false;
  }

  if (data_info->input_tensors.empty()) {
    GELOGE(PARAM_INVALID, "data info input tensors is empty.");
    return false;
  }

  return static_cast<Format>(data_info->input_tensors[0].format) == FORMAT_FILTER_HWCK ||
      static_cast<Format>(data_info->input_tensors[0].format) == FORMAT_HWCN;
}

bool OpInfoUtils::TransFilterData(const std::shared_ptr<OpInfo> &data_info, const void *in_data, uint32_t length) {
  GELOGI("Start trans filter data.");
  if (data_info == nullptr) {
    GELOGE(PARAM_INVALID, "data info ptr is null.");
    return false;
  }

  if (data_info->input_tensors.empty() || data_info->output_tensors.empty()) {
    GELOGE(PARAM_INVALID, "data info input tensors size %zu, output tensor size %zu.", data_info->input_tensors.size(),
           data_info->output_tensors.size());
    return false;
  }

  if (in_data == nullptr) {
    GELOGE(PARAM_INVALID, "In data ptr is null.");
    return false;
  }

  // Transform to KCHW
  GELOGI("copy filter data op: %s, need transfer.", data_info->name.c_str());
  data_info->input_tensors[0].format = static_cast<uint32_t>(FORMAT_NCHW);
  data_info->input_tensors[0].datatype = static_cast<uint32_t>(DT_FLOAT);
  data_info->input_tensors[0].dims = std::vector<int64_t>(
      {data_info->input_tensors[0].GetDim(kHwckDimK), data_info->input_tensors[0].GetDim(kHwckDimC),
       data_info->input_tensors[0].GetDim(kHwckDimH), data_info->input_tensors[0].GetDim(kHwckDimW)});

  void *out_data = nullptr;
  auto total_size = static_cast<uint32_t>(data_info->input_tensors[0].GetShapeSize() * sizeof(float));
  if (total_size != length) {
    GELOGE(FAILED, "Input filter data length(%u) not correct,need:%u!", length, total_size);
    return false;
  }
  TransDataHWCK2KCHW(in_data, data_info->input_tensors[0].GetDim(kHwckDimH),
                     data_info->input_tensors[0].GetDim(kHwckDimW), data_info->input_tensors[0].GetDim(kHwckDimC),
                     data_info->input_tensors[0].GetDim(kHwckDimK), &out_data);

  // Transform to FracZ
  // using namespace cce;
  cce::ccFilterDescriptor_t input_desc = nullptr;
  GE_MAKE_GUARD(input_desc, [&] {
    if (input_desc) GE_CHK_CCE(cce::ccDestroyFilterDescriptor(&input_desc));
  });
  cce::ccFilterDescriptor_t output_desc = nullptr;
  GE_MAKE_GUARD_FILTER_DESC(output_desc);
  bool ret = InitFilterTensorDescriptor(data_info->input_tensors[0].dims, data_info->input_tensors[0].format,
                                        data_info->input_tensors[0].datatype, input_desc);
  if (!ret) {
    delete[] reinterpret_cast<float *>(out_data);
    out_data = nullptr;
    DestroyFilterDescriptor(input_desc);
    GELOGE(INTERNAL_ERROR, "InitTensorDescriptor input_desc failed.");
    return false;
  }

  ret = InitFilterTensorDescriptor(data_info->output_tensors[0].dims, data_info->input_tensors[0].format,
                                   data_info->input_tensors[0].datatype, output_desc);
  if (!ret) {
    delete[] reinterpret_cast<float *>(out_data);
    out_data = nullptr;
    DestroyFilterDescriptor(output_desc);
    DestroyFilterDescriptor(input_desc);
    GELOGE(INTERNAL_ERROR, "InitTensorDescriptor output_desc failed.");
    return false;
  }

  void *fp16_data_addr = nullptr;
  uint32_t output_size = data_info->output_tensors[0].size;

  rtError_t rt_ret = rtMallocHost(&fp16_data_addr, output_size);
  if (rt_ret != RT_ERROR_NONE) {
    delete[] reinterpret_cast<float *>(out_data);
    out_data = nullptr;
    DestroyFilterDescriptor(output_desc);
    DestroyFilterDescriptor(input_desc);
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }
  GE_MAKE_GUARD_RTMEM(fp16_data_addr);

  cce::ccStatus_t cc_ret = cce::ccTransFilter(input_desc, out_data, output_desc, fp16_data_addr, output_size);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    delete[] reinterpret_cast<float *>(out_data);
    out_data = nullptr;
    DestroyFilterDescriptor(output_desc);
    DestroyFilterDescriptor(input_desc);
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return false;
  }

  delete[] reinterpret_cast<float *>(out_data);
  out_data = nullptr;

  // Copy input data to data node
  const std::vector<uintptr_t> &outputs = data_info->output_addrs;
  if (outputs.empty()) {
    GELOGE(PARAM_INVALID, "data_info %s output_addrs is empty.", data_info->name.c_str());
    return false;
  }

  rt_ret = rtMemcpy(reinterpret_cast<void *>(outputs[0]), output_size, fp16_data_addr, output_size,
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  GELOGI("Filter data op transdata end.");
  return true;
}

bool OpInfoUtils::InitFilterTensorDescriptor(const std::vector<int64_t> &dims, uint32_t format, uint32_t dtype,
                                             cce::ccFilterDescriptor_t &cc_tensor) {
  if (dims.empty()) {
    GELOGE(FAILED, "Invalid dim size");
    return false;
  }
  cce::ccTensorFormat_t cc_format = cce::tagCcTensorFormat(format);
  cce::ccDataType_t data_type = cce::tagCcDataType(dtype);
  if (cc_format != cce::CC_TENSOR_NCHW && cc_format != cce::CC_TENSOR_FRACTAL_Z && cc_format != cce::CC_TENSOR_HWCN) {
    GELOGE(PARAM_INVALID, "Filter tensor cc_format:%u not correct.", format);
    return false;
  }
  if (dims.size() <= static_cast<size_t>(kNchwDimW)) {
    GELOGE(PARAM_INVALID, "Array index is invalid!");
    return false;
  }

  // Create tensor descriptor
  cce::ccStatus_t cc_ret = cce::ccCreateFilterDescriptor(&cc_tensor);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return false;
  }

  if (cc_format == cce::CC_TENSOR_FRACTAL_Z) {
    cc_ret = cce::ccSetFilterFractalDescriptor(
        cc_tensor, cc_format, data_type, static_cast<int32_t>(dims[kNchwDimN]), static_cast<int32_t>(dims[kNchwDimC]),
        static_cast<int32_t>(dims[kNchwDimH]), static_cast<int32_t>(dims[kNchwDimW]));
    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
      return false;
    }
  } else if (cc_format == cce::CC_TENSOR_HWCN) {
    cc_ret = cce::ccSetFilterFractalDescriptor(
        cc_tensor, cc_format, data_type, static_cast<int32_t>(dims[kNchwDimW]), static_cast<int32_t>(dims[kNchwDimH]),
        static_cast<int32_t>(dims[kNchwDimN]), static_cast<int32_t>(dims[kNchwDimC]));
    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
      return false;
    }
  } else {
    cc_ret = cce::ccSetFilterFractalDescriptor(
        cc_tensor, cc_format, data_type, static_cast<int32_t>(dims[kNchwDimN]), static_cast<int32_t>(dims[kNchwDimC]),
        static_cast<int32_t>(dims[kNchwDimH]), static_cast<int32_t>(dims[kNchwDimW]));
    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
      return false;
    }
  }
  return true;
}

void OpInfoUtils::DestroyFilterDescriptor(cce::ccFilterDescriptor_t &cc_filter) {
  if (cc_filter != nullptr) {
    cce::ccStatus_t cc_ret = ccDestroyFilterDescriptor(&cc_filter);
    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(CCE_FAILED, "ccDestroyFilterDescriptor failed. ret = %d", static_cast<int32_t>(cc_ret));
    }

    cc_filter = nullptr;
  }
}

void OpInfoUtils::DestroyTensorDescriptor(cce::ccTensorDescriptor_t &cc_tensor) {
  if (cc_tensor != nullptr) {
    cce::ccStatus_t cc_ret = cce::ccDestroyTensorDescriptor(&cc_tensor);
    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
      return;
    }
    cc_tensor = nullptr;
  }
}

bool OpInfoUtils::IsInputTensorNeedTrans(const std::shared_ptr<OpInfo> &data_info) {
  if (data_info == nullptr) {
    GELOGE(PARAM_INVALID, "data info is null.");
    return false;
  }

  if (data_info->input_tensors.empty() || data_info->output_tensors.empty()) {
    GELOGE(PARAM_INVALID, "data info input tensors size %zu, output tensor size %zu.", data_info->input_tensors.size(),
           data_info->output_tensors.size());
    return false;
  }

  if (static_cast<Format>(data_info->output_tensors[0].format) == FORMAT_NC1HWC0 &&
      static_cast<DataType>(data_info->output_tensors[0].datatype) == DT_INT8) {
    // AIPP input，Consider compatibility and judge according to this condition.
    // Add attribute in data node to mark whether it is AIPP
    return false;
  }

  return data_info->input_tensors[0].format != data_info->output_tensors[0].format ||
         data_info->input_tensors[0].datatype != data_info->output_tensors[0].datatype;
}

void OpInfoUtils::TransDataHWCK2KCHW(const void *input, int64_t H, int64_t W, int64_t C, int64_t K, void **output) {
  if (input == nullptr) {
    return;
  }
  if (output == nullptr) {
    return;
  }
  const char *w_data = reinterpret_cast<const char *>(input);

  int64_t count = H * W * C * K;
  if (count <= 0) {
    GELOGE(PARAM_INVALID, "Count value must be greater than 0, but count = %ld", count);
    return;
  }

  float *buf = new (std::nothrow) float[count]();
  if (buf == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Buf must not be null.");
    return;
  }

  const float *src_buff = nullptr;
  float *dst_buff = nullptr;
  for (int64_t h = 0; h < H; ++h) {
    for (int64_t w = 0; w < W; ++w) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t k = 0; k < K; ++k) {
          src_buff = reinterpret_cast<const float *>(w_data) + ((h * W * C * K) + (w * C * K) + (c * K) + (k));
          dst_buff = buf + ((k * C * H * W) + (c * H * W) + (h * W) + (w));
          *dst_buff = *src_buff;
        }
      }
    }
  }
  *output = buf;
}

bool OpInfoUtils::IsComputDimsSize(const uint32_t format, const uint32_t real_dim_cnt) {
  return ((format == static_cast<uint32_t>(cce::CC_TENSOR_ND)) ||
      ((format != static_cast<uint32_t>(cce::CC_TENSOR_NC1KHKWHWC0)) &&
          (format != static_cast<uint32_t>(cce::CC_TENSOR_C1HWNCoC0)) &&
           (real_dim_cnt > static_cast<uint32_t>(DIM_DEFAULT_SIZE))));
}

static const auto set_real_dim_cnt = [](uint32_t real_dim_cnt, const std::vector<int64_t> &dim) {
  return static_cast<uint32_t>(((real_dim_cnt == 0) && (dim.size() > DIM_DEFAULT_SIZE)) ? dim.size()
                                                                                              : real_dim_cnt);
};

bool OpInfoUtils::InitTensorDescriptor(uint32_t format, uint32_t data_type, const std::vector<int64_t> &dim,
                                       cce::ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt) {
  cce::ccDataType_t data_type_ = cce::tagCcDataType(data_type);

  real_dim_cnt = set_real_dim_cnt(real_dim_cnt, dim);

  if (IsComputDimsSize(format, real_dim_cnt)) {
    // (Format is ND) or (Dimension is greater than 4 and format is not NC1KHKWHWC0 or C1HWNCoC0)
    return InitTensorNdDescriptor(data_type, dim, cc_tensor, real_dim_cnt);
  } else if (format == static_cast<uint32_t>(cce::CC_TENSOR_NC1KHKWHWC0)) {
    return InitTensorPoolingMaskDescriptor(format, data_type, dim, cc_tensor, real_dim_cnt);
  } else if (format == static_cast<uint32_t>(cce::CC_TENSOR_C1HWNCoC0)) {
    return InitTensor6dDescriptor(format, data_type, dim, cc_tensor, real_dim_cnt);
  }
  std::vector<int64_t> dim_vector;
  TransferDim(dim, dim_vector);

  if (!CheckParam(format, data_type, dim_vector)) {
    GELOGE(PARAM_INVALID, "Check param fail.");
    return false;
  }

  // Create tensor descriptor
  cce::ccStatus_t cc_ret = cce::ccCreateTensorDescriptor(&cc_tensor);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return false;
  }

  // The last two outputs of fusedbatchnormgrad are 0. The special processing of fusedbatchnormgrad
  if (dim.size() == 1 && dim[0] == 0) {
    (void)cce::ccSetTensorRealDimCnt(cc_tensor, real_dim_cnt);
    (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
    cc_tensor = nullptr;
    return false;
  }

  if (format >= static_cast<uint32_t>(cce::CC_TENSOR_HASHTABLE_LOOKUP_LOOKUPS) &&
      format <= static_cast<uint32_t>(cce::CC_TENSOR_HASHTABLE_LOOKUP_HITS)) {
    int32_t dims[dim.size()];
    for (size_t i = 0; i < dim.size(); ++i) {
      dims[i] = static_cast<int32_t>(dim[i]);
    }

    cc_ret = cce::ccSetTensorNdDescriptor(cc_tensor, data_type_, dim.size(), dims);
    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(CCE_FAILED, "Call cce api failed, ret: %d", static_cast<int32_t>(cc_ret));
      (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
      cc_tensor = nullptr;
      return false;
    }

    cce::ccTensorFormat_t tensor_format = cce::tagCcTensorFormat(format);
    cc_ret = cce::ccSetTensorFormat(cc_tensor, tensor_format);
    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(CCE_FAILED, "Call cce api failed, ret: %d", static_cast<int32_t>(cc_ret));
      (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
      cc_tensor = nullptr;
      return false;
    }

    cc_ret = cce::ccSetTensorRealDimCnt(cc_tensor, real_dim_cnt);
    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
      (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
      cc_tensor = nullptr;
      return false;
    }

    return true;
  } else if (format == static_cast<uint32_t>(cce::CC_TENSOR_NHWC)) {
    return InitTensor4dDescriptor(format, data_type, cc_tensor, static_cast<int32_t>(dim_vector.at(0)),
                                  static_cast<int32_t>(dim_vector.at(3)), static_cast<int32_t>(dim_vector.at(1)),
                                  static_cast<int32_t>(dim_vector.at(2)), real_dim_cnt);
  } else if (format == static_cast<uint32_t>(cce::CC_TENSOR_HWCN)) {
    return InitTensor4dDescriptor(format, data_type, cc_tensor, static_cast<int32_t>(dim_vector.at(3)),
                                  static_cast<int32_t>(dim_vector.at(2)), static_cast<int32_t>(dim_vector.at(0)),
                                  static_cast<int32_t>(dim_vector.at(1)), real_dim_cnt);
  }

  // else default
  return InitTensor4dDescriptor(format, data_type, cc_tensor, static_cast<int32_t>(dim_vector.at(0)),
                                static_cast<int32_t>(dim_vector.at(1)), static_cast<int32_t>(dim_vector.at(2)),
                                static_cast<int32_t>(dim_vector.at(3)), real_dim_cnt);
}

void OpInfoUtils::TransferDim(const std::vector<int64_t> &dim, std::vector<int64_t> &dim_vector) {
  uint32_t input_shape_size = static_cast<uint32_t>(dim.size());
  std::list<uint32_t> new_dim_list;

  for (auto dim_temp : dim) {
    new_dim_list.push_back(dim_temp);
  }
  if (input_shape_size > static_cast<uint32_t>(DIM_DEFAULT_SIZE)) {
    dim_vector = dim;
    GELOGI("The size of dim_vector is %u, do not to transfer dim", input_shape_size);
    return;
  }
  switch (input_shape_size) {
    case 0: {
      new_dim_list.push_back(1);
      new_dim_list.push_back(1);
      new_dim_list.push_back(1);
      new_dim_list.push_back(1);
      break;
    }
    case 1: {
      new_dim_list.push_front(1);
      new_dim_list.push_back(1);
      new_dim_list.push_back(1);
      break;
    }
    case 2: {
      new_dim_list.push_front(1);
      new_dim_list.push_back(1);
      break;
    }
    case 3: {
      new_dim_list.push_front(1);
      break;
    }
    default: {}
  }

  dim_vector.clear();
  for (auto new_dim : new_dim_list) {
    dim_vector.push_back(new_dim);
  }
}

bool OpInfoUtils::InitTensorNdDescriptor(uint32_t data_type, const std::vector<int64_t> &dim,
                                         cce::ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt) {
  cce::ccDataType_t data_type_ = cce::tagCcDataType(data_type);
  cce::ccStatus_t cc_ret = cce::ccCreateTensorDescriptor(&cc_tensor);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return false;
  }

  int32_t *real_dim = nullptr;
  if (real_dim_cnt > 0) {
    real_dim = new (std::nothrow) int32_t[real_dim_cnt];
    if (real_dim == nullptr) {
      GELOGE(FAILED, "Failed to malloc memory");
      return false;
    }
  }

  for (size_t i = 0; i < dim.size(); ++i) {
    if (i >= real_dim_cnt || i >= kDimMaxSize) {
      break;
    }
    real_dim[i] = static_cast<int32_t>(dim[i]);
  }

  cc_ret = cce::ccSetTensorNdDescriptor(cc_tensor, data_type_, real_dim_cnt, real_dim);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
    cc_tensor = nullptr;
    delete[] real_dim;
    return false;
  }

  delete[] real_dim;
  return true;
}

bool OpInfoUtils::InitTensorPoolingMaskDescriptor(uint32_t format, uint32_t data_type, const std::vector<int64_t> &dim,
                                                  cce::ccTensorDescriptor_t &cc_tensor, uint32_t) {
  cce::ccStatus_t cc_ret = cce::ccCreatePoolingMaskDescriptor(&cc_tensor);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return false;
  }
  cce::ccTensorFormat_t format_ = cce::tagCcTensorFormat(format);
  cce::ccDataType_t data_type_ = cce::tagCcDataType(data_type);

  if (dim.size() != kPoolMaskDescDimSize) {
    GELOGE(PARAM_INVALID, "The dim size of format CC_TENSOR_NC1KHKWHWC0 must be 6,dim size id %zu.", dim.size());
    (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
    cc_tensor = nullptr;
    return false;
  }

  cc_ret = cce::ccSetPoolingMaskTensorDescriptor(
      cc_tensor, format_, data_type_, static_cast<int32_t>(dim[kNchwDimN]), static_cast<int32_t>(dim[kNchwDimC]),
      static_cast<int32_t>(dim[kNchwDimH]), static_cast<int32_t>(dim[kNchwDimW]),
      static_cast<int32_t>(dim[kPoolMaskDescWinH]), static_cast<int32_t>(dim[kPoolMaskDescWinW]));
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
    cc_tensor = nullptr;
    return false;
  }

  return true;
}

bool OpInfoUtils::InitTensor6dDescriptor(uint32_t format, uint32_t data_type, const std::vector<int64_t> &dim,
                                         cce::ccTensorDescriptor_t &cc_tensor, uint32_t) {
  cce::ccDataType_t data_type_ = cce::tagCcDataType(data_type);
  cce::ccStatus_t cc_ret = cce::ccCreateTensorDescriptor(&cc_tensor);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    return false;
  }

  cce::ccTensorFormat_t format_ = cce::tagCcTensorFormat(format);
  if (dim.size() != static_cast<uint32_t>(DIM_C1HWNCoC0_SIZE)) {
    GELOGE(PARAM_INVALID, "The dim size of format C1HWNCoC0_DIM_SIZE must be 5,dim size id %zu.", dim.size());
    (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
    cc_tensor = nullptr;
    return false;
  }

  cc_ret = cce::ccSetFilter6dDescriptor(
      cc_tensor, format_, data_type_, static_cast<int32_t>(dim[C1HWNCoC0_DIM_C1]),
      static_cast<int32_t>(dim[C1HWNCoC0_DIM_H]), static_cast<int32_t>(dim[C1HWNCoC0_DIM_W]),
      static_cast<int32_t>(dim[C1HWNCoC0_DIM_N]), static_cast<int32_t>(dim[C1HWNCoC0_DIM_Co]),
      static_cast<int32_t>(dim[C1HWNCoC0_DIM_C0]));
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
    cc_tensor = nullptr;
    return false;
  }

  return true;
}

bool OpInfoUtils::InitTensor4dDescriptor(uint32_t format, uint32_t data_type, cce::ccTensorDescriptor_t &cc_tensor,
                                         int32_t n, int32_t c, int32_t h, int32_t w, uint32_t real_dim_cnt) {
  cce::ccDataType_t data_type_ = cce::tagCcDataType(data_type);
  cce::ccTensorFormat_t format_ = cce::tagCcTensorFormat(format);
  auto cc_ret = cce::ccSetTensor4dDescriptor(cc_tensor, format_, data_type_, n, c, h, w);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
    cc_tensor = nullptr;
    return false;
  }

  cc_ret = cce::ccSetTensorRealDimCnt(cc_tensor, real_dim_cnt);
  if (cc_ret != cce::CC_STATUS_SUCCESS) {
    GELOGE(CCE_FAILED, "Call cce api failed, ret: 0x%X", cc_ret);
    (void)cce::ccDestroyTensorDescriptor(&cc_tensor);
    cc_tensor = nullptr;
    return false;
  }

  return true;
}

bool OpInfoUtils::CheckParam(uint32_t format, uint32_t data_type, const std::vector<int64_t> &dim_vector) {
  // format
  if (format >= static_cast<uint32_t>(cce::CC_TENSOR_RESERVED)) {
    GELOGE(PARAM_INVALID, "Not supported format, format = %u", format);
    return false;
  }

  // data type
  if (data_type >= static_cast<uint32_t>(cce::CC_DATA_RESERVED)) {
    GELOGE(PARAM_INVALID, "Not supported data type, type = %u", data_type);
    return false;
  }

  // input shape
  auto input_shape_size = dim_vector.size();
  if (input_shape_size != static_cast<size_t>(DIM_DEFAULT_SIZE)) {
    GELOGW("input_shape_size is %u", input_shape_size);
  }

  return true;
}

bool OpInfoUtils::GetOutputSize(const std::shared_ptr<OpInfo> &op_info, std::vector<uint32_t> &output_size_list,
                                std::vector<uint32_t> &output_memory_size_list) {
  if (op_info == nullptr) {
    GELOGE(PARAM_INVALID, "op info is null.");
    return false;
  }

  for (size_t i = 0; i < op_info->output_tensors.size(); ++i) {
    auto output_desc = op_info->output_tensors[i];
    bool output_tensor = op_info->output_tensors[i].is_output;

    if (output_tensor) {
      // Recalculate the size directly using desc of net output op.
      cce::ccTensorDescriptor_t cctensor = nullptr;
      bool status = InitTensorDescriptor(output_desc.format, output_desc.datatype, output_desc.dims, cctensor,
                                         output_desc.real_dim_cnt);
      if (!status) {
        GELOGE(FAILED, "InitTensorDescriptor fail.");
        return false;
      }
      // Call the API of CCE to obtain the converted size and other parameters.
      uint32_t size = 0;
      uint32_t memory_size = 0;
      auto cc_ret0 = cce::ccGetTensorSizeInBytes(cctensor, &size);
      auto cc_ret1 = cce::ccGetTensorMemorySizeInBytes(cctensor, &memory_size);
      DestroyTensorDescriptor(cctensor);
      if (cc_ret0 != cce::CC_STATUS_SUCCESS) {
        GELOGE(CCE_FAILED, "ccGetTensorSizeInBytes fail, ret = 0x%X.", cc_ret0);
        return false;
      }
      if (cc_ret1 != cce::CC_STATUS_SUCCESS) {
        GELOGE(CCE_FAILED, "ccGetTensorMemorySizeInBytes fail, ret = 0x%X.", cc_ret1);
        return false;
      }

      output_size_list.push_back(size);
      output_memory_size_list.push_back(memory_size);
    }
  }

  if (output_size_list.size() != output_memory_size_list.size()) {
    GELOGE(INTERNAL_ERROR, "Output size list length %zu not equal output memory size list length %zu.",
           output_size_list.size(), output_memory_size_list.size());
    return false;
  }

  return true;
}

}  // namespace model_runner
}  // namespace ge
