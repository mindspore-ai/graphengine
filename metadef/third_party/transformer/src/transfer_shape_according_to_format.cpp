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

/*!
 * \file transfer_shape_according_to_format.cpp
 * \brief set shape according to original format and current format
 */
#include "transformer/inc/transfer_shape_according_to_format.h"

namespace common {
namespace transformer {
using namespace ge;
using namespace std;

ShapeTransferAccordingToFormat::ShapeTransferAccordingToFormat(void) {
  getNewShapeFuncMap = {
      {ge::FORMAT_NCHW, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNCHWShapeByAxisValue)},
      {ge::FORMAT_NHWC, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNHWCShapeByAxisValue)},
      {ge::FORMAT_NC1HWC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNC1HWC0ShapeByAxisValue)},
      {ge::FORMAT_NDC1HWC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNDC1HWC0ShapeByAxisValue)},
      {ge::FORMAT_FRACTAL_Z, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetFzShapeByAxisValue)},
      {ge::FORMAT_HWCN, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetHWCNShapeByAxisValue)},
      {ge::FORMAT_C1HWNCoC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetC1HWNCoC0ShapeByAxisValue)},
      {ge::FORMAT_FRACTAL_NZ, std::make_shared<GetNewShapeByAxisValueAndFormat>(GetNzShapeByAxisValue)}};

  mapOfDtypeAndC0 = {
      {ge::DT_FLOAT16, SHAPE_NUMBER_16}, {ge::DT_FLOAT, SHAPE_NUMBER_16},  {ge::DT_INT8, SHAPE_NUMBER_32},
      {ge::DT_INT16, SHAPE_NUMBER_16},   {ge::DT_INT32, SHAPE_NUMBER_16},  {ge::DT_INT64, SHAPE_NUMBER_16},
      {ge::DT_UINT8, SHAPE_NUMBER_16},   {ge::DT_UINT16, SHAPE_NUMBER_32}, {ge::DT_UINT32, SHAPE_NUMBER_16},
      {ge::DT_UINT64, SHAPE_NUMBER_16},  {ge::DT_BOOL, SHAPE_NUMBER_16}};
}

bool ShapeTransferAccordingToFormat::GetNDC1HWC0ShapeByAxisValue(vector<int64_t> &new_shape, const int64_t &impl_type,
                                 const std::vector<int64_t> &axis_value, const vector<int64_t> &nd_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  new_shape.push_back(axis_value[AXIS_N]);
  new_shape.push_back(axis_value[AXIS_D]);
  new_shape.push_back(axis_value[AXIS_C1]);
  new_shape.push_back(axis_value[AXIS_H]);
  new_shape.push_back(axis_value[AXIS_W]);
  new_shape.push_back(axis_value[AXIS_C0]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNCHWShapeByAxisValue(vector<int64_t>& newShape, const int64_t& implType,
                                                             const vector<int64_t>& axisValue,
                                                             const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  newShape.push_back(axisValue[AXIS_N]);
  newShape.push_back(axisValue[AXIS_C]);
  newShape.push_back(axisValue[AXIS_H]);
  newShape.push_back(axisValue[AXIS_W]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNHWCShapeByAxisValue(vector<int64_t>& newShape, const int64_t& implType,
                                                             const vector<int64_t>& axisValue,
                                                             const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  newShape.push_back(axisValue[AXIS_N]);
  newShape.push_back(axisValue[AXIS_H]);
  newShape.push_back(axisValue[AXIS_W]);
  newShape.push_back(axisValue[AXIS_C]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue(vector<int64_t>& newShape, const int64_t& implType,
                                                                const vector<int64_t>& axisValue,
                                                                const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  if (implType == EN_IMPL_HW_TBE || implType == EN_IMPL_CUSTOM_TBE || implType == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
    newShape.push_back(axisValue[AXIS_N]);
    newShape.push_back(axisValue[AXIS_C1]);
    newShape.push_back(axisValue[AXIS_H]);
    newShape.push_back(axisValue[AXIS_W]);
    newShape.push_back(axisValue[AXIS_C0]);
  } else {
    newShape.push_back(axisValue[AXIS_N]);
    newShape.push_back(axisValue[AXIS_C]);
    newShape.push_back(axisValue[AXIS_H]);
    newShape.push_back(axisValue[AXIS_W]);
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzShapeByAxisValue(vector<int64_t>& newShape, const int64_t& implType,
                                                           const vector<int64_t>& axisValue,
                                                           const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  if (ndValue.size() == SIZE_OF_CN) {
    auto sizeOfOriginalVec = ndValue.size();
    newShape = ndValue;
    /* sizeOfOriginalVec - 1 mean the last value of original vec
     * sizeOfOriginalVec - 2 mean the second last value of original vec */
    newShape[sizeOfOriginalVec - MINUS_VALUE_ONE] =
        DivisionCeiling(ndValue[sizeOfOriginalVec - MINUS_VALUE_ONE], SHAPE_NUMBER_16);
    newShape[sizeOfOriginalVec - MINUS_VALUE_TWO] =
        DivisionCeiling(ndValue[sizeOfOriginalVec - MINUS_VALUE_TWO], axisValue[AXIS_C0]);
    newShape.push_back(SHAPE_NUMBER_16);
    newShape.push_back(axisValue[AXIS_C0]);
  } else {
    if (implType == EN_IMPL_HW_TBE || implType == EN_IMPL_CUSTOM_TBE || implType == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
      int64_t hwc1 = axisValue[AXIS_C1] * axisValue[AXIS_H] * axisValue[AXIS_W];
      newShape.push_back(hwc1);
      newShape.push_back(DivisionCeiling(axisValue[AXIS_N], NI));
      newShape.push_back(NI);
      newShape.push_back(axisValue[AXIS_C0]);
    } else {
      newShape.push_back(axisValue[AXIS_N]);
      newShape.push_back(axisValue[AXIS_C]);
      newShape.push_back(axisValue[AXIS_H]);
      newShape.push_back(axisValue[AXIS_W]);
    }
  }

  return true;
}

bool ShapeTransferAccordingToFormat::GetHWCNShapeByAxisValue(vector<int64_t>& newShape, const int64_t& implType,
                                                             const vector<int64_t>& axisValue,
                                                             const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  newShape.push_back(axisValue[AXIS_H]);
  newShape.push_back(axisValue[AXIS_W]);
  newShape.push_back(axisValue[AXIS_C]);
  newShape.push_back(axisValue[AXIS_N]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetC1HWNCoC0ShapeByAxisValue(vector<int64_t>& newShape, const int64_t& implType,
                                                                  const vector<int64_t>& axisValue,
                                                                  const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  newShape.push_back(axisValue[AXIS_C1]);
  newShape.push_back(axisValue[AXIS_H]);
  newShape.push_back(axisValue[AXIS_W]);
  newShape.push_back(axisValue[AXIS_N]);
  newShape.push_back(axisValue[AXIS_Co]);
  newShape.push_back(axisValue[AXIS_C0]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNzShapeByAxisValue(vector<int64_t>& newShape, const int64_t& implType,
                                                           const vector<int64_t>& axisValue,
                                                           const vector<int64_t>& ndValue) {
  CHECK(ndValue.empty(), GELOGD("ndValue is empty!"), return true);
  CHECK(axisValue.empty() || axisValue.size() <= AXIS_C0,
        GELOGD("AxisValue is empty or its size %zu <= AXIS_C0[%u]", axisValue.size(), AXIS_C0), return true);
  uint32_t sizeOfOriginalVec = ndValue.size();
  if (sizeOfOriginalVec < MINIMUM_NZ_SHAPE_DIM_NUM) {
    GELOGD("ndValue's dim num is less than 2!");
    return true;
  }
  /* axisValue is initialized as a size 6 vector. */
  newShape = ndValue;

  /* sizeOfOriginalVec - 1 mean the last value of original vec
   * sizeOfOriginalVec - 2 mean the second last value of original vec */
  newShape[sizeOfOriginalVec - MINUS_VALUE_ONE] =
      DivisionCeiling(ndValue[sizeOfOriginalVec - MINUS_VALUE_TWO], (int64_t)SHAPE_NUMBER_16);

  newShape[sizeOfOriginalVec - MINUS_VALUE_TWO] =
      DivisionCeiling(ndValue[sizeOfOriginalVec - MINUS_VALUE_ONE], axisValue[AXIS_C0]);
  newShape.push_back(SHAPE_NUMBER_16);
  newShape.push_back(axisValue[AXIS_C0]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(ShapeAndFormat& shapeAndFormatInfo, int64_t* c) {
  /* The default new shape is old shape */
  shapeAndFormatInfo.newShape = shapeAndFormatInfo.oldShape;
  if (shapeAndFormatInfo.oldFormat >= ge::FORMAT_RESERVED || shapeAndFormatInfo.newFormat >= ge::FORMAT_RESERVED) {
    GELOGE(GRAPH_FAILED, "Old format %u or new format %u is invalid!", shapeAndFormatInfo.oldFormat,
      shapeAndFormatInfo.newFormat);
    return false;
  }

  if (shapeAndFormatInfo.currentDataType >= ge::DT_UNDEFINED) {
    GELOGE(GRAPH_FAILED, "currentDataType %u is invalid!", shapeAndFormatInfo.currentDataType);
    return false;
  }
  AxisUtil* axisutil_object = new AxisUtil();
  if (!axisutil_object->HasAxisValueFunc(shapeAndFormatInfo.oldFormat)) {
    delete axisutil_object;
    return true;
  }

  auto iterGetNewShapeFunc = getNewShapeFuncMap.find(shapeAndFormatInfo.newFormat);
  if (iterGetNewShapeFunc == getNewShapeFuncMap.end()) {
    GELOGD("Can not get new shape of new format %u!", shapeAndFormatInfo.newFormat);
    delete axisutil_object;
    return true;
  }
  GELOGD("Original format %u, new format %u", shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat);
  GetNewShapeByAxisValueAndFormatPtr getNewShapeFunc = iterGetNewShapeFunc->second;
  CHECK_NOTNULL(getNewShapeFunc);
  std::vector<int64_t> axisValue;
  for (uint32_t i = 0; i < AXIS_BOTTOM; i++) {
    axisValue.push_back(1);
  }
  std::vector<int64_t> ndValue;
  uint32_t c0;
  if (mapOfDtypeAndC0.empty()) {
    c0 = SHAPE_NUMBER_16;
  } else {
    auto iterGetC0 = mapOfDtypeAndC0.find(shapeAndFormatInfo.currentDataType);
    if (iterGetC0 == mapOfDtypeAndC0.end()) {
      GELOGE(GRAPH_FAILED, "Dtype is not support.");
      delete axisutil_object;
      return true;
    }
    c0 = iterGetC0->second;
  }

  // The value of C0 should be 4 while format is 5HD-4 or FRAZ-4
  if (shapeAndFormatInfo.newFormat == ge::FORMAT_NC1HWC0_C04) {
    c0 = SHAPE_DIM_VALUE_C04;
  }

  bool status = axisutil_object->GetAxisValueByOriginFormat(
      shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.oldShape, c0, axisValue, ndValue);
  if (status != true && shapeAndFormatInfo.newFormat != ge::FORMAT_FRACTAL_NZ) {
    delete axisutil_object;
    return true;
  }
  delete axisutil_object;

  shapeAndFormatInfo.newShape.clear();
  (*getNewShapeFunc)(shapeAndFormatInfo.newShape, shapeAndFormatInfo.opImplType, axisValue, ndValue);
  if (c != nullptr) {
    *c = axisValue[AXIS_C];
  }
  return true;
}
} // namespace transformer
} // namespace common
