/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

/*!
 * \file axis_util.cpp
 * \brief get the axis value
 */
#include "transformer/inc/axis_util.h"
#include "graph/types.h"

namespace common {
namespace transformer {
using namespace ge;
using namespace std;

AxisUtil::AxisUtil() {
  getAxisValueFuncMap = {{FORMAT_NCHW, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNCHW)},
                         {FORMAT_NHWC, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNHWC)},
                         {FORMAT_NC1HWC0, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNC1HWC0)},
                         {FORMAT_HWCN, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByHWCN)},
                         {FORMAT_ND, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByND)},
                         {FORMAT_C1HWNCoC0, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByC1HWNCoC0)},
                         {FORMAT_NDHWC, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNDHWC)},
                         {FORMAT_NCDHW, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNCDHW)},
                         {FORMAT_DHWCN, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByDHWCN)},
                         {FORMAT_DHWNC, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByDHWNC)}};
}

int64_t DivisionCeiling(int64_t dividend, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  } else {
    return (dividend + divisor - 1) / divisor;
  }
}

bool AxisUtil::GetAxisValueByOriginFormat(const Format &format, const vector<int64_t> &dimVec, const uint32_t &c0,
                                          vector<int64_t> &axisValue, vector<int64_t> &ndValue) {
  auto iterGetAxisFunc = getAxisValueFuncMap.find(format);
  if (iterGetAxisFunc == getAxisValueFuncMap.end()) {
    GELOGI("Can not get axis value of old format %u!", format);
    return false;
  }
  GetAxisValueInfoByFormatPtr getAxisFunc = iterGetAxisFunc->second;
  CHECK_NOTNULL(getAxisFunc);
  return (*getAxisFunc)(dimVec, c0, axisValue, ndValue);
}

bool AxisUtil::HasAxisValueFunc(const Format &format) {
  auto iterGetAxisFunc = getAxisValueFuncMap.find(format);
  if (iterGetAxisFunc == getAxisValueFuncMap.end()) {
    GELOGI("Can not get axis value of format %u!", format);
    return false;
  }
  return true;
}

bool AxisUtil::CheckParams(const vector<int64_t> &originalDimVec, const uint32_t &c0, vector<int64_t> &axisValue,
                           vector<int64_t> &ndValue) {
  ndValue = originalDimVec;
  auto dimSize = originalDimVec.size();
  if (dimSize < DIM_DEFAULT_SIZE) {
    /* Before this funcion, we should call function PadDimensionTo4. */
    GELOGI("Dimension size %zu is invalid.", dimSize);
    return false;
  }
  if (c0 == 0) {
    GELOGE(GRAPH_FAILED, "[ERROR]c0 is zero!");
    return false;
  }

  return true;
}

bool AxisUtil::GetAxisValueByND(const vector<int64_t> &originalDimVec, const uint32_t &c0, vector<int64_t> &axisValue,
                                vector<int64_t> &ndValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), GELOGI("Original dim vector is empty!"), return true);
  ndValue = originalDimVec;
  /* To differentiate the input datatype of int8 and others */
  axisValue[AXIS_C0] = c0;
  if (originalDimVec.size() == NCHW_DIMENSION_NUM) {
    axisValue[AXIS_N] = originalDimVec[AXIS_NCHW_DIM_N];
    axisValue[AXIS_C] = originalDimVec[AXIS_NCHW_DIM_C];
    axisValue[AXIS_H] = originalDimVec[AXIS_NCHW_DIM_H];
    axisValue[AXIS_W] = originalDimVec[AXIS_NCHW_DIM_W];
    axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_NCHW_DIM_C], (int64_t)c0);
    axisValue[AXIS_Co] = c0;
  }
  return true;
}

bool AxisUtil::GetAxisValueByNCHW(const vector<int64_t> &originalDimVec, const uint32_t &c0, vector<int64_t> &axisValue,
                                  vector<int64_t> &ndValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), GELOGI("Original dim vector is empty!"), return true);
  /* C0 Must be set for case ND or 2D-NCHW to NZ */
  axisValue[AXIS_C0] = c0;
  // TODO: temporarily modified to warning level.If modified normally, it needs complementary dimension for origin shape
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true, GELOGW("[WARNING]Parameter is invalid!"),
        return false);

  axisValue[AXIS_N] = originalDimVec[AXIS_NCHW_DIM_N];
  axisValue[AXIS_C] = originalDimVec[AXIS_NCHW_DIM_C];
  axisValue[AXIS_H] = originalDimVec[AXIS_NCHW_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_NCHW_DIM_W];
  axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_NCHW_DIM_C], (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByNHWC(const vector<int64_t> &originalDimVec, const uint32_t &c0, vector<int64_t> &axisValue,
                                  vector<int64_t> &ndValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), GELOGI("Original dim vector is empty!"), return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  // TODO: temporarily modified to warning level.If modified normally, it needs complementary dimension for origin shape
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true, GELOGW("[WARNING]Parameter is invalid!"),
        return false);

  axisValue[AXIS_N] = originalDimVec[AXIS_NHWC_DIM_N];
  axisValue[AXIS_C] = originalDimVec[AXIS_NHWC_DIM_C];
  axisValue[AXIS_H] = originalDimVec[AXIS_NHWC_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_NHWC_DIM_W];
  axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_NHWC_DIM_C], (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByNC1HWC0(const vector<int64_t> &originalDimVec, const uint32_t &c0,
                                     vector<int64_t> &axisValue, vector<int64_t> &ndValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), GELOGI("Original dim vector is empty!"), return true);
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true, GELOGE(GRAPH_FAILED,"[ERROR]Parameter is invalid!"),
        return false);

  auto dimSize = originalDimVec.size();
  if (dimSize == DIM_DEFAULT_SIZE + 1) {
    axisValue[AXIS_C1] = originalDimVec[AXIS_NC1HWC0_DIM_C1];
    axisValue[AXIS_C0] = originalDimVec[AXIS_NC1HWC0_DIM_C0];
    axisValue[AXIS_C] = axisValue[AXIS_C1] * axisValue[AXIS_C0];
  } else {
    axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_NCHW_DIM_C], (int64_t)c0);
    axisValue[AXIS_C0] = c0;
    axisValue[AXIS_C] = originalDimVec[AXIS_NCHW_DIM_C];
  }

  axisValue[AXIS_N] = originalDimVec[AXIS_NCHW_DIM_N];
  axisValue[AXIS_H] = originalDimVec[AXIS_NCHW_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_NCHW_DIM_W];
  return true;
}

bool AxisUtil::GetAxisValueByHWCN(const vector<int64_t> &originalDimVec, const uint32_t &c0, vector<int64_t> &axisValue,
                                  vector<int64_t> &ndValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), GELOGI("Original dim vector is empty!"), return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  // TODO: temporarily modified to warning level. If modified normally, it needs complementary dimension for origin shape
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true, GELOGW("[WARNING]Parameter is invalid!"),
        return false);

  axisValue[AXIS_N] = originalDimVec[AXIS_HWCN_DIM_N];
  axisValue[AXIS_C] = originalDimVec[AXIS_HWCN_DIM_C];
  axisValue[AXIS_H] = originalDimVec[AXIS_HWCN_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_HWCN_DIM_W];
  axisValue[AXIS_C1] = DivisionCeiling(originalDimVec[AXIS_HWCN_DIM_C], (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByC1HWNCoC0(const vector<int64_t> &originalDimVec, const uint32_t &c0,
                                       vector<int64_t> &axisValue, vector<int64_t> &ndValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(originalDimVec.empty(), GELOGI("Original dim vector is empty!"), return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  CHECK(CheckParams(originalDimVec, c0, axisValue, ndValue) != true, GELOGE(GRAPH_FAILED, "[ERROR]Parameter is invalid!"),
        return false);

  axisValue[AXIS_N] = originalDimVec[AXIS_C1HWNCoC0_DIM_N];
  axisValue[AXIS_C] = originalDimVec[AXIS_C1HWNCoC0_DIM_C1] * c0;
  axisValue[AXIS_H] = originalDimVec[AXIS_C1HWNCoC0_DIM_H];
  axisValue[AXIS_W] = originalDimVec[AXIS_C1HWNCoC0_DIM_W];
  axisValue[AXIS_C1] = originalDimVec[AXIS_C1HWNCoC0_DIM_C1];
  axisValue[AXIS_Co] = originalDimVec[AXIS_C1HWNCoC0_DIM_Co];
  return true;
}

bool AxisUtil::GetAxisValueByNDHWC(const std::vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                  std::vector<int64_t>& axis_value, std::vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(original_dim_vec.empty(), GELOGI("Original dim vector is empty!"), return true);

  axis_value[AXIS_C0] = c0;
  nd_value = original_dim_vec;

  axis_value[AXIS_N] = original_dim_vec[NDHWC_DIM_N];
  int64_t axis_c_val = original_dim_vec[NDHWC_DIM_C];

  axis_value[AXIS_C] = axis_c_val;
  axis_value[AXIS_H] = original_dim_vec[NDHWC_DIM_H];
  axis_value[AXIS_W] = original_dim_vec[NDHWC_DIM_W];
  axis_value[AXIS_C1] = DivisionCeiling(axis_c_val, c0);
  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_Co] = c0;
  axis_value[AXIS_D] = original_dim_vec[NDHWC_DIM_D];
  return true;
}

bool AxisUtil::GetAxisValueByNCDHW(const std::vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                  std::vector<int64_t>& axis_value, std::vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(original_dim_vec.empty(), GELOGI("Original dim vector is empty!"), return true);

  axis_value[AXIS_C0] = c0;
  nd_value = original_dim_vec;

  axis_value[AXIS_N] = original_dim_vec[NCDHW_DIM_N];
  int64_t axis_c_val = original_dim_vec[NCDHW_DIM_C];

  axis_value[AXIS_C] = axis_c_val;
  axis_value[AXIS_H] = original_dim_vec[NCDHW_DIM_H];
  axis_value[AXIS_W] = original_dim_vec[NCDHW_DIM_W];
  axis_value[AXIS_C1] = DivisionCeiling(axis_c_val, c0);
  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_Co] = c0;
  axis_value[AXIS_D] = original_dim_vec[NCDHW_DIM_D];
  return true;
}

bool AxisUtil::GetAxisValueByDHWCN(const std::vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                  std::vector<int64_t>& axis_value, std::vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(original_dim_vec.empty(), GELOGI("Original dim vector is empty!"), return true);

  axis_value[AXIS_C0] = c0;
  nd_value = original_dim_vec;

  axis_value[AXIS_N] = original_dim_vec[DHWCN_DIM_N];
  int64_t axis_c_val = original_dim_vec[DHWCN_DIM_C];

  axis_value[AXIS_C] = axis_c_val;
  axis_value[AXIS_H] = original_dim_vec[DHWCN_DIM_H];
  axis_value[AXIS_W] = original_dim_vec[DHWCN_DIM_W];
  axis_value[AXIS_C1] = DivisionCeiling(axis_c_val, c0);
  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_Co] = c0;
  axis_value[AXIS_D] = original_dim_vec[DHWCN_DIM_D];
  return true;
}

bool AxisUtil::GetAxisValueByDHWNC(const std::vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                  std::vector<int64_t>& axis_value, std::vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(original_dim_vec.empty(), GELOGI("Original dim vector is empty!"), return true);

  axis_value[AXIS_C0] = c0;
  nd_value = original_dim_vec;

  axis_value[AXIS_N] = original_dim_vec[DHWNC_DIM_N];
  int64_t axis_c_val = original_dim_vec[DHWNC_DIM_C];

  axis_value[AXIS_C] = axis_c_val;
  axis_value[AXIS_H] = original_dim_vec[DHWNC_DIM_H];
  axis_value[AXIS_W] = original_dim_vec[DHWNC_DIM_W];
  axis_value[AXIS_C1] = DivisionCeiling(axis_c_val, c0);
  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_Co] = c0;
  axis_value[AXIS_D] = original_dim_vec[DHWNC_DIM_D];
  return true;
}
} // namespace transformer
} // namespace common
