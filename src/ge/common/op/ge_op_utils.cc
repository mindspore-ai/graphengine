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

#include "framework/common/op/ge_op_utils.h"

#include <list>

#include "cce/dnn.h"
#include "cce/dnn_struct.hpp"
#include "common/ge/ge_util.h"
#include "external/graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/fmk_error_codes.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/attr_value_util.h"
#include "framework/common/util.h"
#include "graph/anchor.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "mmpa/mmpa_api.h"

#define RETURN_IF_TRUE(cond, errcode, ...) \
  do {                                     \
    if (cond) {                            \
      GELOGE(errcode, __VA_ARGS__);        \
      return errcode;                      \
    }                                      \
  } while (0);

using domi::DOMI_TENSOR_NCHW;
using std::vector;

namespace ge {
// General constant
const int32_t kDimMaxSize = 8;
const float DEFAULT_ALPHA_VALUE = 1.0;
const float DEFAULT_BETA_VALUE = 0.0;
const int NORMAL_TENSOR_SIZE = 4;
const int32_t kDimSizeZero = 0;
const int32_t kDimSizeOne = 1;
const int32_t kDimSizeTwo = 2;
const int32_t kDimSizeThree = 3;
const uint32_t kSliceDataNum = 2;

// Add Sub Mul
const uint32_t ADD_INPUT_NUM = 2;
const uint32_t SUB_INPUT_NUM = 2;
const uint32_t MUL_INPUT_NUM = 2;

// Permute
const int32_t PERMUTE_ORDER_NUM = 4;
// Ssd PriroBox
const double SSD_PRIORBOX_ASPECT_RATIO_VALUE = 1.0;
// Switch
const uint32_t SWITCH_INPUT_NUM = 2;
const uint32_t SWITCH_OUTPUT_NUM = 2;
const uint32_t SWITCH_FALSE_OUTPUT = 0;
const uint32_t SWITCH_TRUE_OUTPUT = 1;
const uint32_t SWITCH_DATA_INPUT = 0;
const uint32_t SWITCH_PRED_INPUT = 1;
// Internal constant
const uint32_t kPoolMaskDescWinH = 4;
const uint32_t kPoolMaskDescWinW = 5;
const uint32_t kPoolMaskDescDimSize = 6;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool OpUtils::IsComputDimsSize(const int32_t format,
                                                                                const uint32_t real_dim_cnt) {
  return ((format == cce::CC_TENSOR_ND) ||
          ((format != cce::CC_TENSOR_NC1KHKWHWC0) && (format != cce::CC_TENSOR_C1HWNCoC0) &&
           (real_dim_cnt > DIM_DEFAULT_SIZE)));
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
OpUtils::InitTensorDescriptor(const GeTensorDesc &tensor, cce::ccTensorDescriptor_t &cc_tensor) {
  return InitTensorDescriptor(tensor, static_cast<int32_t>(tensor.GetDataType()), cc_tensor);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OpUtils::InitTensorDescriptor(
    const GeTensorDesc &model_tensor, int32_t dst_data_type, cce::ccTensorDescriptor_t &cc_tensor) {
  uint32_t real_dim_cnt = OpUtils::GetRealDimCnt(model_tensor);
  return InitTensorDescriptor(static_cast<int32_t>(model_tensor.GetFormat()), dst_data_type,
                              model_tensor.GetShape().GetDims(), cc_tensor, real_dim_cnt);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
OpUtils::InitTensorDescriptor(const GeTensor &model_tensor, cce::ccTensorDescriptor_t &cc_tensor) {
  return InitTensorDescriptor(model_tensor, static_cast<int32_t>(model_tensor.GetTensorDesc().GetDataType()),
                              cc_tensor);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OpUtils::InitTensorDescriptor(
    const GeTensor &model_tensor, int32_t dst_data_type, cce::ccTensorDescriptor_t &cc_tensor) {
  const GeTensorDesc &tensor_desc = model_tensor.GetTensorDesc();
  const GeShape &shape = tensor_desc.GetShape();
  return InitTensorDescriptor(static_cast<int32_t>(tensor_desc.GetFormat()), dst_data_type, shape.GetDims(), cc_tensor,
                              static_cast<uint32_t>(shape.GetDimNum()));
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
OpUtils::InitTensorDescriptor(int32_t format, int32_t data_type, const std::vector<int64_t> &dim,
                              cce::ccTensorDescriptor_t &cc_tensor, uint32_t real_dim_cnt) {
  Status ret = SUCCESS;
  ccDataType_t data_type_ = cce::tagCcDataType(data_type);
  real_dim_cnt =
      static_cast<uint32_t>(((real_dim_cnt == 0) && (dim.size() > DIM_DEFAULT_SIZE)) ? dim.size() : real_dim_cnt);
  if (IsComputDimsSize(format, real_dim_cnt)) {
    GE_CHK_CCE_RET(cce::ccCreateTensorDescriptor(&cc_tensor));
#if (defined(__GNUC__) && !(defined(__ICC) || defined(__INTEL_COMPILER))) && \
    (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 50000
    // Variable length array initialization is not supported in gcc4. X compilation environment
    GE_CHK_BOOL_RET_STATUS(real_dim_cnt <= CC_DIM_MAX, domi::CCE_FAILED, "real_dim_cnt support <= 8.");
    int32_t real_dim[CC_DIM_MAX] = {0};
#else
    int32_t real_dim[real_dim_cnt] = {};
#endif
    uint32_t i = 0;
    for (auto dim_temp : dim) {
      GE_CHK_BOOL_EXEC_NOLOG(i < real_dim_cnt && i < kDimMaxSize, break);
      real_dim[i] = static_cast<int32_t>(dim_temp);
      i++;
    }

    auto cc_ret = cce::ccSetTensorNdDescriptor(cc_tensor, data_type_, real_dim_cnt, real_dim);
    GE_IF_BOOL_EXEC(cc_ret != cce::CC_STATUS_SUCCESS,
                      GELOGE(domi::CCE_FAILED, "Call cce failed. cc_ret = %d", cc_ret);
                      GE_CHK_CCE(cce::ccDestroyTensorDescriptor(&cc_tensor)); return domi::CCE_FAILED);

    return ret;
  } else if (format == cce::CC_TENSOR_NC1KHKWHWC0) {
    GE_CHK_CCE_RET(cce::ccCreatePoolingMaskDescriptor(&cc_tensor));
    cce::ccTensorFormat_t format_new = cce::tagCcTensorFormat(format);
    GE_IF_BOOL_EXEC(
        dim.size() != kPoolMaskDescDimSize,
        GELOGE(PARAM_INVALID, "format CC_TENSOR_NC1KHKWHWC0 dim size must be 6,dim size id %lu.", dim.size());
        GE_CHK_CCE(cce::ccDestroyTensorDescriptor(&cc_tensor)); return PARAM_INVALID);
    auto cc_ret = ccSetPoolingMaskTensorDescriptor(
        cc_tensor, format_new, data_type_, static_cast<int32_t>(dim[NCHW_DIM_N]),
        static_cast<int32_t>(dim[NCHW_DIM_C]), static_cast<int32_t>(dim[NCHW_DIM_H]),
        static_cast<int32_t>(dim[NCHW_DIM_W]), static_cast<int32_t>(dim[kPoolMaskDescWinH]),
        static_cast<int32_t>(dim[kPoolMaskDescWinW]));

    GE_IF_BOOL_EXEC(cc_ret != cce::CC_STATUS_SUCCESS,
                      GELOGE(domi::CCE_FAILED, "Call cce failed. cc_ret = %d", cc_ret);
                      GE_CHK_CCE(cce::ccDestroyTensorDescriptor(&cc_tensor)); return domi::CCE_FAILED);
    return ret;
  } else if (format == cce::CC_TENSOR_C1HWNCoC0) {
    GE_CHK_CCE_RET(cce::ccCreateTensorDescriptor(&cc_tensor));
    cce::ccTensorFormat_t format_new = cce::tagCcTensorFormat(format);
    GE_IF_BOOL_EXEC(
        dim.size() != DIM_C1HWNCoC0_SIZE,
        GELOGE(PARAM_INVALID, "format C1HWNCoC0_DIM_SIZE dim size must be 5,dim size id %lu.", dim.size());
        GE_CHK_CCE(cce::ccDestroyTensorDescriptor(&cc_tensor)); return PARAM_INVALID);

    auto cc_ret = cce::ccSetFilter6dDescriptor(
        cc_tensor, format_new, data_type_, static_cast<int32_t>(dim[C1HWNCoC0_DIM_C1]),
        static_cast<int32_t>(dim[C1HWNCoC0_DIM_H]), static_cast<int32_t>(dim[C1HWNCoC0_DIM_W]),
        static_cast<int32_t>(dim[C1HWNCoC0_DIM_N]), static_cast<int32_t>(dim[C1HWNCoC0_DIM_Co]),
        static_cast<int32_t>(dim[C1HWNCoC0_DIM_C0]));

    GE_IF_BOOL_EXEC(cc_ret != cce::CC_STATUS_SUCCESS, GELOGE(CCE_FAILED, "Call cce failed. cc_ret = %d", cc_ret);
                      GE_CHK_CCE(cce::ccDestroyTensorDescriptor(&cc_tensor)); return CCE_FAILED);

    return ret;
  }
  std::vector<int64_t> dim_vector;
  (void)TransferDim(dim, dim_vector);  // TransferDim always return success, no need to check value
  // format
  if (!CheckEnumValid(format, cce::CC_TENSOR_NCHW, cce::CC_TENSOR_RESERVED)) {
    GELOGE(PARAM_INVALID, "not supported format, format = %d", format);
    return PARAM_INVALID;
  }
  cce::ccTensorFormat_t format_new = cce::tagCcTensorFormat(format);

  // data type
  if (!CheckEnumValid(data_type, cce::CC_DATA_FLOAT, cce::CC_DATA_RESERVED)) {
    GELOGE(PARAM_INVALID, "not supported data type, type = %d", data_type);
    return PARAM_INVALID;
  }

  // create tensor descriptor
  GE_CHK_CCE_RET(cce::ccCreateTensorDescriptor(&cc_tensor));

  // input shape
  size_t input_shape_size = dim_vector.size();
  GE_IF_BOOL_EXEC(input_shape_size != DIM_DEFAULT_SIZE, GELOGI("input_shape_size is %zu", input_shape_size));

  // The last two outputs of fusedbatchnormgrad are 0. Need special processing for fusedbatchnormgrad.
  GE_IF_BOOL_EXEC(dim.size() == 1 && dim[0] == 0,
                    GE_IF_BOOL_EXEC(cce::ccSetTensorRealDimCnt(cc_tensor, real_dim_cnt) != cce::CC_STATUS_SUCCESS,
                                      GELOGE(domi::CCE_FAILED, "Call cce failed.");
                                      GE_CHK_CCE(cce::ccDestroyTensorDescriptor(&cc_tensor)); return domi::CCE_FAILED);
                    return ret);

  if (format == cce::CC_TENSOR_NHWC) {
    auto cc_ret = cce::ccSetTensor4dDescriptor(
        cc_tensor, format_new, data_type_, static_cast<int32_t>(dim_vector.at(NHWC_DIM_N)),
        static_cast<int32_t>(dim_vector.at(NHWC_DIM_C)), static_cast<int32_t>(dim_vector.at(NHWC_DIM_H)),
        static_cast<int32_t>(dim_vector.at(NHWC_DIM_W)));

    GE_IF_BOOL_EXEC(cc_ret != cce::CC_STATUS_SUCCESS,
                      GELOGE(domi::CCE_FAILED, "Call cce failed. cc_ret = %d", cc_ret);
                      ret = domi::CCE_FAILED);
  } else if (format == cce::CC_TENSOR_HWCN) {
    auto cc_ret = cce::ccSetTensor4dDescriptor(
        cc_tensor, format_new, data_type_, static_cast<int32_t>(dim_vector.at(NHWC_DIM_C)),
        static_cast<int32_t>(dim_vector.at(NHWC_DIM_W)), static_cast<int32_t>(dim_vector.at(NHWC_DIM_N)),
        static_cast<int32_t>(dim_vector.at(NHWC_DIM_H)));

    GE_IF_BOOL_EXEC(cc_ret != cce::CC_STATUS_SUCCESS,
                      GELOGE(domi::CCE_FAILED, "Call cce failed. cc_ret = %d", cc_ret);
                      ret = domi::CCE_FAILED);
  } else if (format >= cce::CC_TENSOR_HASHTABLE_LOOKUP_LOOKUPS && format <= cce::CC_TENSOR_HASHTABLE_LOOKUP_HITS) {
    int32_t dims[dim.size()];
    for (size_t i = 0; i < dim.size(); i++) {
      dims[i] = static_cast<int32_t>(dim[i]);
    }

    auto cc_ret = cce::ccSetTensorNdDescriptor(cc_tensor, data_type_, static_cast<int32_t>(dim.size()), dims);
    cce::ccSetTensorFormat(cc_tensor, format_new);
    GE_IF_BOOL_EXEC(cc_ret != cce::CC_STATUS_SUCCESS, GELOGE(CCE_FAILED, "Call cce failed. cc_ret = %d", cc_ret);
                      ret = CCE_FAILED);
  } else {
    auto cc_ret = cce::ccSetTensor4dDescriptor(
        cc_tensor, format_new, data_type_, static_cast<int32_t>(dim_vector.at(NHWC_DIM_N)),
        static_cast<int32_t>(dim_vector.at(NHWC_DIM_H)), static_cast<int32_t>(dim_vector.at(NHWC_DIM_W)),
        static_cast<int32_t>(dim_vector.at(NHWC_DIM_C)));

    GE_IF_BOOL_EXEC(cc_ret != cce::CC_STATUS_SUCCESS,
                      GELOGE(domi::CCE_FAILED, "Call cce failed. cc_ret = %d", cc_ret);
                      ret = domi::CCE_FAILED);
  }
  auto cc_ret = cce::ccSetTensorRealDimCnt(cc_tensor, real_dim_cnt);
  GE_IF_BOOL_EXEC(cc_ret != cce::CC_STATUS_SUCCESS, GELOGE(domi::CCE_FAILED, "Call cce failed. cc_ret = %d", cc_ret);
                    ret = domi::CCE_FAILED);

  if (ret != SUCCESS) {
    GE_CHK_CCE(cce::ccDestroyTensorDescriptor(&cc_tensor));
    cc_tensor = nullptr;
  }

  return ret;
}

// Initialize filter description
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
OpUtils::InitFilterDescriptor(const GeTensor &model_filter, cce::ccFilterDescriptor_t &cc_filter) {
  const GeTensorDesc &tensor_desc = model_filter.GetTensorDesc();
  const GeShape &shape = tensor_desc.GetShape();
  const std::vector<int64_t> dims = shape.GetDims();

  // format
  RETURN_IF_TRUE(!CheckEnumValid(tensor_desc.GetFormat(), cce::CC_TENSOR_NCHW, cce::CC_TENSOR_RESERVED), PARAM_INVALID,
                 "not supported format, format = %d", tensor_desc.GetFormat());

  uint32_t tmp_int = static_cast<uint32_t>(tensor_desc.GetFormat());
  cce::ccTensorFormat_t format = cce::tagCcTensorFormat(tmp_int);

  // data type
  RETURN_IF_TRUE(!CheckEnumValid(tensor_desc.GetDataType(), cce::CC_DATA_FLOAT, cce::CC_DATA_RESERVED), PARAM_INVALID,
                 "not supported data type, type = %s",
                 TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());

  uint32_t dt_tmp = static_cast<uint32_t>(tensor_desc.GetDataType());

  ccDataType_t dataType = cce::tagCcDataType(dt_tmp);

  // create filter descriptor
  GE_CHK_CCE_RET(cce::ccCreateFilterDescriptor(&cc_filter));

  Status ret = SUCCESS;
  // input filter
  size_t filter_shape_size = shape.GetDimNum();
  if (filter_shape_size == DIM_DEFAULT_SIZE) {
    cce::ccStatus_t cc_ret = cce::CC_STATUS_SUCCESS;

    GE_IF_BOOL_EXEC(dims.size() < 4, GELOGE(domi::CCE_FAILED, "dims is invalid!"); return domi::CCE_FAILED);

    if (dataType == CC_DATA_INT8) {
      cc_ret = ccSetInt8Filter4dDescriptor(cc_filter, format, dataType, static_cast<int32_t>(dims[KCHW_DIM_K]),
                                           static_cast<int32_t>(dims[KCHW_DIM_C]),
                                           static_cast<int32_t>(dims[KCHW_DIM_H]),
                                           static_cast<int32_t>(dims[KCHW_DIM_W]), cce::CC_DATA_HALF);
    } else if (format == cce::CC_TENSOR_FRACTAL_Z_C04 || format == cce::CC_TENSOR_FRACTAL_DECONV_SP_STRIDE_TRANS ||
               format == cce::CC_TENSOR_FRACTAL_Z || format == cce::CC_TENSOR_FRACTAL_DECONV) {
      cc_ret = cce::ccSetFilterFractalDescriptor(
          cc_filter, format, dataType, static_cast<int32_t>(dims[KCHW_DIM_K]),
          static_cast<int32_t>(dims[KCHW_DIM_C]), static_cast<int32_t>(dims[KCHW_DIM_H]),
          static_cast<int32_t>(dims[KCHW_DIM_W]));
    } else if (format == cce::CC_TENSOR_NHWC) {
      cc_ret = cce::ccSetFilter4dDescriptor(cc_filter, format, dataType, static_cast<int32_t>(dims[NHWC_DIM_N]),
                                            static_cast<int32_t>(dims[NHWC_DIM_C]),
                                            static_cast<int32_t>(dims[NHWC_DIM_H]),
                                            static_cast<int32_t>(dims[NHWC_DIM_W]));
    } else if (format == cce::CC_TENSOR_CHWN) {
      cc_ret = cce::ccSetFilter4dDescriptor(cc_filter, format, dataType, static_cast<int32_t>(dims[CHWN_DIM_N]),
                                            static_cast<int32_t>(dims[CHWN_DIM_C]),
                                            static_cast<int32_t>(dims[CHWN_DIM_H]),
                                            static_cast<int32_t>(dims[CHWN_DIM_W]));
    } else if (format == cce::CC_TENSOR_HWCN) {
      cc_ret = cce::ccSetFilter4dDescriptor(cc_filter, format, dataType, static_cast<int32_t>(dims[NHWC_DIM_C]),
                                            static_cast<int32_t>(dims[NHWC_DIM_W]),
                                            static_cast<int32_t>(dims[NHWC_DIM_N]),
                                            static_cast<int32_t>(dims[NHWC_DIM_H]));
    } else {
      cc_ret = cce::ccSetFilter4dDescriptor(cc_filter, format, dataType, static_cast<int32_t>(dims[KCHW_DIM_K]),
                                            static_cast<int32_t>(dims[KCHW_DIM_C]),
                                            static_cast<int32_t>(dims[KCHW_DIM_H]),
                                            static_cast<int32_t>(dims[KCHW_DIM_W]));
    }

    if (cc_ret != cce::CC_STATUS_SUCCESS) {
      GELOGE(domi::CCE_FAILED, "ccSetFilterDescriptor failed. cc_ret = %d, format1 = %d", cc_ret, format);
      ret = domi::CCE_FAILED;
    }
  } else {
    GELOGE(UNSUPPORTED, "not supported shape size. size = %d", filter_shape_size);
    ret = UNSUPPORTED;
  }

  if (ret != SUCCESS) {
    GE_CHK_CCE(cce::ccDestroyFilterDescriptor(&cc_filter));
    cc_filter = nullptr;
  }

  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool OpUtils::ConvertDim(cce::ccTensorFormat_t src_format,
                                                                          const std::vector<int64_t> &src,
                                                                          cce::ccTensorFormat_t dst_format,
                                                                          std::vector<int64_t> &dst) {
  // The input of 3-dimension and 4-dimension is considered as picture dimension,
  // which needs to be converted according to specific format
  if ((src.size() != DIM_DEFAULT_SIZE && src.size() != 3) || src_format == dst_format) {
    GELOGI("Convert format , src size %zu <3 ,not need convert", src.size());
    dst = src;
    return true;
  }

  std::vector<int64_t> nchw_dim;

  switch (src_format) {
    case cce::CC_TENSOR_NCHW:
      nchw_dim = src;
      break;
    case cce::CC_TENSOR_NHWC:
      if (src.size() == DIM_DEFAULT_SIZE) {
        nchw_dim.push_back(src[NHWC_DIM_N]);
        nchw_dim.push_back(src[NHWC_DIM_C]);
        nchw_dim.push_back(src[NHWC_DIM_H]);
        nchw_dim.push_back(src[NHWC_DIM_W]);
      } else {
        nchw_dim.push_back(src[HWC_DIM_C]);
        nchw_dim.push_back(src[HWC_DIM_H]);
        nchw_dim.push_back(src[HWC_DIM_W]);
      }
      break;
    default:
      GELOGW("Not support src format is %d", src_format);
      return false;
  }

  if (nchw_dim.empty()) {
    GELOGW("Vector is empty!");
    return false;
  }

  switch (dst_format) {
    case cce::CC_TENSOR_NCHW:
      dst = nchw_dim;
      break;
    case cce::CC_TENSOR_NHWC:
      if (src.size() == DIM_DEFAULT_SIZE) {
        dst.push_back(nchw_dim[NCHW_DIM_N]);
        dst.push_back(nchw_dim[NCHW_DIM_H]);
        dst.push_back(nchw_dim[NCHW_DIM_W]);
        dst.push_back(nchw_dim[NCHW_DIM_C]);
      } else {
        dst.push_back(nchw_dim[CHW_DIM_H]);
        dst.push_back(nchw_dim[CHW_DIM_W]);
        dst.push_back(nchw_dim[CHW_DIM_C]);
      }
      break;
    default:
      GELOGW("Not support dst format of %d", dst_format);
      return false;
  }

  return true;
}
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void OpUtils::DestroyTensorDescriptor(
    cce::ccTensorDescriptor_t &cc_tensor) noexcept {
  if (cc_tensor != nullptr) {
    cce::ccStatus_t ret = cce::ccDestroyTensorDescriptor(&cc_tensor);
    GE_LOGE_IF(ret != cce::CC_STATUS_SUCCESS, "cce::ccDestroyTensorDescriptor failed. ret = %d", ret);
    cc_tensor = nullptr;
  }
}
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void OpUtils::DestroyFilterDescriptor(
    cce::ccFilterDescriptor_t &cc_filter) {
  if (cc_filter != nullptr) {
    cce::ccStatus_t ret = ccDestroyFilterDescriptor(&cc_filter);
    GE_LOGE_IF(ret != cce::CC_STATUS_SUCCESS, "ccDestroyFilterDescriptor failed. ret = %d", ret);
    cc_filter = nullptr;
  }
}

// Get the value of key from attr
#define AIPP_GET_ATTR_VALUE(KEY, ATTR_TYPE)                          \
  if (aipp_attr.GetItem(#KEY).GetValue<ATTR_TYPE>(KEY) != SUCCESS) { \
    GELOGI("Attr %s will take default value.", #KEY);                \
    break;                                                           \
  }

// Converting aippparams and attrdefmap
#define AIPP_CONVERT_FORMAT_EX(KEY, ORG_TYPE, SAVE_TYPE, ATTR_TYPE) \
  do {                                                              \
    SAVE_TYPE KEY = static_cast<SAVE_TYPE>(0);                      \
    AIPP_GET_ATTR_VALUE(KEY, ATTR_TYPE)                             \
    aipp_params->set_##KEY(ORG_TYPE(KEY));                          \
  } while (0)

// Converting aippparams and attrdefmap
#define AIPP_CONVERT_FORMAT(KEY, KEY_TYPE, ATTR_TYPE) AIPP_CONVERT_FORMAT_EX(KEY, KEY_TYPE, KEY_TYPE, ATTR_TYPE)

#define AIPP_CONVERT_INT(KEY) AIPP_CONVERT_FORMAT(KEY, int64_t, GeAttrValue::INT)

#define AIPP_CONVERT_BOOL(KEY) AIPP_CONVERT_FORMAT(KEY, bool, GeAttrValue::BOOL)

#define AIPP_CONVERT_FLOAT(KEY) AIPP_CONVERT_FORMAT(KEY, float, GeAttrValue::FLOAT)

// Transform aippparams (with repeated decoration) and attrdefmap
#define AIPP_CONVERT_LIST_FORMAT(KEY, KEY_TYPE, REQUIRED, ATTR_TYPE) \
  do {                                                               \
    if (REQUIRED) {                                                  \
      KEY_TYPE KEY;                                                  \
      AIPP_GET_ATTR_VALUE(KEY, ATTR_TYPE)                            \
      aipp_params->add_##KEY(KEY);                                   \
    }                                                                \
  } while (0)

#define AIPP_CONVERT_LIST_INT(KEY, REQUIRED) AIPP_CONVERT_LIST_FORMAT(KEY, int64_t, REQUIRED, GeAttrValue::INT)

#define AIPP_CONVERT_LIST_BOOL(KEY, REQUIRED) AIPP_CONVERT_LIST_FORMAT(KEY, bool, REQUIRED, GeAttrValue::BOOL)

#define AIPP_CONVERT_LIST_FLOAT(KEY, REQUIRED) AIPP_CONVERT_LIST_FORMAT(KEY, float, REQUIRED, GeAttrValue::FLOAT)

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
OpUtils::ConvertAippParams(const GeAttrValue::NamedAttrs &aipp_attr, domi::AippOpParams *aipp_params) {
  GE_CHECK_NOTNULL(aipp_params);
  AIPP_CONVERT_FORMAT_EX(aipp_mode, domi::AippOpParams::AippMode, int32_t, GeAttrValue::INT);

  if (aipp_params->aipp_mode() == domi::AippOpParams::dynamic) {
    AIPP_CONVERT_INT(max_src_image_size);
    AIPP_CONVERT_BOOL(support_rotation);
  } else {
    AIPP_CONVERT_FORMAT_EX(input_format, domi::AippOpParams::InputFormat, int32_t, GeAttrValue::INT);
    AIPP_CONVERT_BOOL(csc_switch);
    AIPP_CONVERT_BOOL(crop);
    AIPP_CONVERT_INT(load_start_pos_w);
    AIPP_CONVERT_INT(load_start_pos_h);
    AIPP_CONVERT_INT(crop_size_w);
    AIPP_CONVERT_INT(crop_size_h);
    AIPP_CONVERT_BOOL(resize);
    AIPP_CONVERT_INT(resize_output_w);
    AIPP_CONVERT_INT(resize_output_h);
    AIPP_CONVERT_BOOL(padding);
    AIPP_CONVERT_INT(left_padding_size);
    AIPP_CONVERT_INT(right_padding_size);
    AIPP_CONVERT_INT(top_padding_size);
    AIPP_CONVERT_INT(bottom_padding_size);
    AIPP_CONVERT_INT(src_image_size_w);
    AIPP_CONVERT_INT(src_image_size_h);
    AIPP_CONVERT_FLOAT(cpadding_value);
    AIPP_CONVERT_BOOL(rbuv_swap_switch);
    AIPP_CONVERT_BOOL(ax_swap_switch);
    AIPP_CONVERT_BOOL(single_line_mode);
    AIPP_CONVERT_INT(mean_chn_0);
    AIPP_CONVERT_INT(mean_chn_1);
    AIPP_CONVERT_INT(mean_chn_2);
    AIPP_CONVERT_FLOAT(min_chn_0);
    AIPP_CONVERT_FLOAT(min_chn_1);
    AIPP_CONVERT_FLOAT(min_chn_2);
    AIPP_CONVERT_LIST_FLOAT(var_reci_chn_0, true);
    AIPP_CONVERT_LIST_FLOAT(var_reci_chn_1, true);
    AIPP_CONVERT_LIST_FLOAT(var_reci_chn_2, true);

    const bool csc_switch = aipp_params->csc_switch();
    AIPP_CONVERT_LIST_INT(matrix_r0c0, csc_switch);
    AIPP_CONVERT_LIST_INT(matrix_r0c1, csc_switch);
    AIPP_CONVERT_LIST_INT(matrix_r0c2, csc_switch);
    AIPP_CONVERT_LIST_INT(matrix_r1c0, csc_switch);
    AIPP_CONVERT_LIST_INT(matrix_r1c1, csc_switch);
    AIPP_CONVERT_LIST_INT(matrix_r1c2, csc_switch);
    AIPP_CONVERT_LIST_INT(matrix_r2c0, csc_switch);
    AIPP_CONVERT_LIST_INT(matrix_r2c1, csc_switch);
    AIPP_CONVERT_LIST_INT(matrix_r2c2, csc_switch);
    AIPP_CONVERT_LIST_INT(output_bias_0, csc_switch);
    AIPP_CONVERT_LIST_INT(output_bias_1, csc_switch);
    AIPP_CONVERT_LIST_INT(output_bias_2, csc_switch);
    AIPP_CONVERT_LIST_INT(input_bias_0, csc_switch);
    AIPP_CONVERT_LIST_INT(input_bias_1, csc_switch);
    AIPP_CONVERT_LIST_INT(input_bias_2, csc_switch);
  }

  return SUCCESS;
}

CceTensorDescriptor::CceTensorDescriptor(cce::ccTensorDescriptor_t cc_tensor) : cc_tensor_(cc_tensor) {}

CceTensorDescriptor::~CceTensorDescriptor() {
  if (cc_tensor_ != nullptr) {
    OpUtils::DestroyTensorDescriptor(cc_tensor_);
    cc_tensor_ = nullptr;
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status CceTensorDescriptor::InitTensor(int32_t format,
                                                                                        int32_t data_type,
                                                                                        const vector<int64_t> &dims) {
  if (cc_tensor_ != nullptr) {
    GELOGE(PARAM_INVALID, "Cannot init cce tensor descriptor twice!");
    return PARAM_INVALID;
  }
  cce::ccTensorDescriptor_t cc_tensor = nullptr;

  Status ret = OpUtils::InitTensorDescriptor(format, data_type, dims, cc_tensor);

  GE_CHK_STATUS_EXEC(ret, OpUtils::DestroyTensorDescriptor(cc_tensor); return FAILED, "init cc_tensor failed.");

  cc_tensor_ = cc_tensor;
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status CceTensorDescriptor::InitTensor(int32_t format,
                                                                                        int32_t data_type,
                                                                                        const ge::GeShape &shape) {
  return InitTensor(format, data_type, shape.GetDims());
}

Status CceTensorDescriptor::GetFormat(cce::ccTensorFormat_t *format) {
  GE_CHECK_NOTNULL(format);
  GE_CHK_CCE_RET(cce::ccGetTensorFormat(cc_tensor_, format));
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status CceTensorDescriptor::GetTensorSizeInBytes(uint32_t *size) {
  GE_CHECK_NOTNULL(size);
  GE_CHK_CCE_RET(cce::ccGetTensorSizeInBytes(cc_tensor_, size));
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
CceTensorDescriptor::TransTensor(const cce::ccTensorDescriptor_t x_desc, const void *x,
                                 const CceTensorDescriptorPtr &y_desc, void *y, uint32_t y_size_in_bytes) {
  GE_CHECK_NOTNULL(y_desc);
  GE_CHK_CCE_RET(cce::ccTransTensor(x_desc, x, y_desc->cc_tensor_, y, y_size_in_bytes));
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY CceTensorDescriptorPtr CceTensorDescriptor::Create() {
  shared_ptr<CceTensorDescriptor> desc = nullptr;
  desc = ge::MakeShared<CceTensorDescriptor>(nullptr);
  if (desc == nullptr) {
    GELOGE(FAILED, "Make CceTensorDescriptor failed.");
    return nullptr;
  }
  return desc;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OpUtils::TransferDim(const std::vector<int64_t> &dim,
                                                                             std::vector<int64_t> &dim_vector) {
  size_t input_shape_size = dim.size();
  std::list<uint32_t> new_dim_list;
  for (auto dim_temp : dim) {
    new_dim_list.push_back(dim_temp);
  }
  if (input_shape_size > DIM_DEFAULT_SIZE) {
    dim_vector = dim;
    GELOGI("Dim_vector size is %zu, do not to transfer dim", input_shape_size);
    return SUCCESS;
  }
  switch (input_shape_size) {
    case kDimSizeZero: {
      new_dim_list.push_back(1);
      new_dim_list.push_back(1);
      new_dim_list.push_back(1);
      new_dim_list.push_back(1);
      break;
    }
    case kDimSizeOne: {
      new_dim_list.push_front(1);
      new_dim_list.push_back(1);
      new_dim_list.push_back(1);
      break;
    }
    case kDimSizeTwo: {
      new_dim_list.push_front(1);
      new_dim_list.push_back(1);
      break;
    }
    case kDimSizeThree: {
      new_dim_list.push_front(1);
      break;
    }
    default:
      GELOGI("Invalid input_shape_size.");
      break;
  }

  dim_vector.clear();
  for (auto dims : new_dim_list) {
    dim_vector.push_back(dims);
  }
  return SUCCESS;
}

void OpUtils::SliceData(std::vector<char *> &input, int64_t chunk_size, std::vector<char *> &output, int64_t begin,
                        int64_t out_dim, int64_t stride) {
  char *slice = nullptr;
  for (size_t j = 0; j < input.size(); j++) {
    slice = input[j] + sizeof(int32_t) * begin * chunk_size;
    for (int64_t i = 0; i < out_dim; i++) {
      output.push_back(slice + sizeof(int32_t) * i * chunk_size * stride);
    }
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OpUtils::SetOutputSliceData(
    void *data, int64_t data_size, int32_t data_type, std::vector<int64_t> &input_dims, std::vector<int64_t> &begin,
    std::vector<int64_t> &output_dims, GeTensor *output, std::vector<int64_t> &stride) {
  GE_CHECK_NOTNULL(data);
  GE_CHECK_NOTNULL(output);
  std::vector<char *> chunk_input;
  std::vector<char *> chunk_output;
  chunk_input.push_back(reinterpret_cast<char *>(data));
  int64_t chunk_size = data_size;
  int dim_size = static_cast<int>(input_dims.size());
  for (int i = 0; i < dim_size; i++) {
    int64_t begin_i = begin[i];
    int64_t size_i = output_dims[i];
    int64_t dim_i = input_dims[i];
    int64_t stride_i = stride[i];
    GE_CHK_BOOL_EXEC((dim_i != 0), return PARAM_INVALID, "Dim_i can't be 0.");
    chunk_size = chunk_size / dim_i;

    if (i % kSliceDataNum == 0) {
      SliceData(chunk_input, chunk_size, chunk_output, begin_i, size_i, stride_i);
      chunk_input.clear();
    } else {
      SliceData(chunk_output, chunk_size, chunk_input, begin_i, size_i, stride_i);
      chunk_output.clear();
    }
  }

  size_t out_size = chunk_input.size() + chunk_output.size();
  GE_CHK_BOOL_RET_STATUS(out_size > 0, FAILED, "Out_size <= 0");

  if (data_type == DT_FLOAT) {
    float *output_data = new (std::nothrow) float[out_size]();
    GE_CHECK_NOTNULL(output_data);
    if (!chunk_input.empty()) {
      for (size_t j = 0; j < out_size; j++) {
        float *value = reinterpret_cast<float *>(chunk_input[j]);
        output_data[j] = *value;
      }
    } else {
      for (size_t j = 0; j < out_size; j++) {
        float *value = reinterpret_cast<float *>(chunk_output[j]);
        output_data[j] = *value;
      }
    }
    (void)output->SetData(reinterpret_cast<uint8_t *>(output_data), out_size * sizeof(float));
    // output_data != nullptr and out_size > 0, SetData always return success, no need to check value
    GE_DELETE_NEW_ARRAY(output_data);
  } else if (data_type == DT_INT32) {
    int *output_data = new (std::nothrow) int[out_size]();
    GE_CHECK_NOTNULL(output_data);

    if (!chunk_input.empty()) {
      for (size_t j = 0; j < out_size; j++) {
        int *value = reinterpret_cast<int *>(chunk_input[j]);
        output_data[j] = *value;
      }
    } else {
      for (size_t j = 0; j < out_size; j++) {
        int *value = reinterpret_cast<int *>(chunk_output[j]);
        output_data[j] = *value;
      }
    }
    (void)output->SetData(reinterpret_cast<uint8_t *>(output_data), out_size * sizeof(int));
    // output_data != nullptr and out_size > 0, SetData always return success, no need to check value
    GE_DELETE_NEW_ARRAY(output_data);
  } else {
    GELOGE(FAILED, "Data type of Slice OP must be float or int32.");
    return FAILED;
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void OpUtils::TransDataHWCK2KCHW(const void *input, int64_t h,
                                                                                  int64_t w, int64_t c, int64_t k,
                                                                                  void **output) {
  if (input == nullptr) {
    return;
  }
  if (output == nullptr) {
    return;
  }
  const char *w_data = (const char *)input;

  int64_t count = h * w * c * k;
  GE_IF_BOOL_EXEC(count <= 0, GELOGW("Count value must be greater than 0, but count = %ld", count); return);
  float *buf = new (std::nothrow) float[count]();
  GE_RT_VOID_CHECK_NOTNULL(buf);
  float *src_buff = nullptr;
  float *dst_buff = nullptr;
  for (int h_i = 0; h_i < h; ++h_i) {
    for (int w_i = 0; w_i < w; ++w_i) {
      for (int c_i = 0; c_i < c; ++c_i) {
        for (int k_i = 0; k_i < k; ++k_i) {
          src_buff = reinterpret_cast<float *>(const_cast<char *>(w_data)) +
                     ((h_i * w * c * k) + (w_i * c * k) + (c_i * k) + (k_i));

          dst_buff = buf + ((k_i * c * h * w) + (c_i * h * w) + (h_i * w) + (w_i));

          *dst_buff = *src_buff;
        }
      }
    }
  }
  *output = buf;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void OpUtils::TransDataKCHW2HWCK(const void *input, int64_t k,
                                                                                  int64_t c, int64_t h, int64_t w,
                                                                                  void *output) {
  if ((input == nullptr) || (output == nullptr)) {
    GELOGD("%s[%d]: input param is nullptr.", __FILE__, __LINE__);
    return;
  }

  const char *w_data = (const char *)input;

  float *buf = reinterpret_cast<float *>(output);
  float *src_buff = nullptr;
  float *dst_buff = nullptr;
  for (int k_i = 0; k_i < k; ++k_i) {
    for (int c_i = 0; c_i < c; ++c_i) {
      for (int h_i = 0; h_i < h; ++h_i) {
        for (int w_i = 0; w_i < w; ++w_i) {
          src_buff = reinterpret_cast<float *>(const_cast<char *>(w_data)) +
                     ((k_i * c * h * w) + (c_i * h * w) + (h_i * w) + (w_i));

          dst_buff = buf + ((h_i * w * c * k) + (w_i * c * k) + (c_i * k) + (k_i));

          *dst_buff = *src_buff;
        }
      }
    }
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
OpUtils::InitFilterTensorDescriptor(const GeTensorDesc &model_tensor, cce::ccFilterDescriptor_t &cc_tensor) {
  auto dims = model_tensor.GetShape().GetDims();
  auto dim_size = dims.size();
  if (dim_size == 0) {
    GELOGE(FAILED, "Invalid dim size");
    return FAILED;
  }
  uint32_t cc_format_tmp = static_cast<uint32_t>(model_tensor.GetFormat());
  cce::ccTensorFormat_t format = cce::tagCcTensorFormat(cc_format_tmp);
  uint32_t model_tensor_dt = static_cast<uint32_t>(model_tensor.GetDataType());
  ccDataType_t data_type = cce::tagCcDataType(model_tensor_dt);
  GE_CHK_BOOL_EXEC(
      ((format == cce::CC_TENSOR_NCHW) || (format == cce::CC_TENSOR_FRACTAL_Z) || (format == cce::CC_TENSOR_HWCN)),
      return PARAM_INVALID, "Filter tensor format:%d not correct.", format);
  GE_IF_BOOL_EXEC(static_cast<uint32_t>(dims.size()) <= NCHW_DIM_W,
                    GELOGE(PARAM_INVALID, "Array index is invalid!");
                    return PARAM_INVALID);
  // create tensor descriptor
  GE_CHK_CCE_RET(cce::ccCreateFilterDescriptor(&cc_tensor));
  if (format == cce::CC_TENSOR_FRACTAL_Z) {
    GE_CHK_CCE_RET(cce::ccSetFilterFractalDescriptor(
        cc_tensor, format, data_type, static_cast<int32_t>(dims[NCHW_DIM_N]),
        static_cast<int32_t>(dims[NCHW_DIM_C]), static_cast<int32_t>(dims[NCHW_DIM_H]),
        static_cast<int32_t>(dims[NCHW_DIM_W])));
  } else if (format == cce::CC_TENSOR_HWCN) {
    GE_CHK_CCE_RET(cce::ccSetFilter4dDescriptor(
        cc_tensor, format, data_type, static_cast<int32_t>(dims[NCHW_DIM_W]),
        static_cast<int32_t>(dims[NCHW_DIM_H]), static_cast<int32_t>(dims[NCHW_DIM_N]),
        static_cast<int32_t>(dims[NCHW_DIM_C])));
  } else {
    GE_CHK_CCE_RET(cce::ccSetFilter4dDescriptor(
        cc_tensor, format, data_type, static_cast<int32_t>(dims[NCHW_DIM_N]),
        static_cast<int32_t>(dims[NCHW_DIM_C]), static_cast<int32_t>(dims[NCHW_DIM_H]),
        static_cast<int32_t>(dims[NCHW_DIM_W])));
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void OpUtils::SetTensorDescriptorAllOffsetQuantizeInfo(
    const GeTensorDesc &tensor, cce::ccTensorDescriptor_t cc_tensor) {
  GE_IF_BOOL_EXEC(!TensorUtils::HasAlloffsetQuantizeInfo(tensor), return;);
  ccVecQuantizePara_t temp;
  AllOffsetQuantizeInfo temp_quantInfo;
  GE_CHK_BOOL_EXEC(TensorUtils::GetAlloffsetQuantizeInfo(tensor, temp_quantInfo) == GRAPH_SUCCESS, return,
                   "Execute GetAlloffsetQuantizeInfo failed.");
  temp.scale = temp_quantInfo.scale;
  temp.offset = static_cast<uint16_t>(temp_quantInfo.offset);
  temp.rrv = 0;
  cce::ccSetTensorDescriptorQuantizeParam(cc_tensor, &temp);
}

vector<ConstGeTensorPtr> OpUtils::GetWeights(const ge::Node &node) { return OpDescUtils::GetWeights(node); }

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY vector<ConstGeTensorPtr> OpUtils::GetWeights(ge::ConstNodePtr node) {
  return OpDescUtils::GetWeights(node);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY vector<GeTensorPtr> OpUtils::MutableWeights(const ge::Node &node) {
  return OpDescUtils::MutableWeights(node);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY vector<GeTensorPtr> OpUtils::MutableWeights(const ge::NodePtr node) {
  return OpDescUtils::MutableWeights(node);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OpUtils::SetWeights(ge::Node &node,
                                                                            const vector<ge::GeTensorPtr> &weights) {
  return OpDescUtils::SetWeights(node, weights);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OpUtils::SetWeights(ge::NodePtr node,
                                                                            const vector<ge::GeTensorPtr> &weights) {
  return OpDescUtils::SetWeights(node, weights);
}

// The caller guarantees that the input sensor is constant
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status
OpUtils::GetShapeDataFromConstTensor(const ConstGeTensorPtr &tensor, DataType type, std::vector<int64_t> &dims) {
  if (tensor == nullptr) {
    GELOGE(PARAM_INVALID, "Input tensor is nullptr");
    return PARAM_INVALID;
  }

  // If the tensor data is a vector, the shape dimension must be 1
  if (tensor->GetTensorDesc().GetShape().GetDims().size() > 1) {
    GELOGE(PARAM_INVALID, "The dimension of the input tensor shape cannot be more than 1, it is %zu",
           tensor->GetTensorDesc().GetShape().GetDims().size());
    return PARAM_INVALID;
  }

  if (type == DT_INT32) {
    int32_t *shape_data = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(tensor->GetData().GetData()));
    size_t dims_num = tensor->GetData().size() / sizeof(int32_t);
    for (size_t i = 0; i < dims_num; i++) {
      dims.push_back(static_cast<int64_t>(shape_data[i]));
    }
  } else if (type == DT_INT64) {
    int64_t *shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(tensor->GetData().GetData()));
    size_t dims_num = tensor->GetData().size() / sizeof(int64_t);
    for (size_t i = 0; i < dims_num; i++) {
      dims.push_back(shape_data[i]);
    }
  } else {
    GELOGE(PARAM_INVALID, "Data type only can be DT_INT32 or DT_INT64. type is %s",
           TypeUtils::DataTypeToSerialString(type).c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

uint32_t OpUtils::GetRealDimCnt(const GeTensorDesc &tensor_desc) {
  uint32_t real_dim_cnt = 0;
  domi::Status ret = TensorUtils::GetRealDimCnt(tensor_desc, real_dim_cnt);
  return (ret == domi::SUCCESS) ? real_dim_cnt : 0;
}
}  // namespace ge
