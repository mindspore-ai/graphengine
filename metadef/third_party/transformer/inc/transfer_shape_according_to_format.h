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
 * \file transfer_shape_according_to_format.h
 * \brief set shape according to original format and current format
 */
#ifndef COMMON_UTILS_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_
#define COMMON_UTILS_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_

#include "transformer/inc/axis_util.h"

#include <memory.h>
#include <functional>
#include <vector>

#include "graph/types.h"
#include "graph/utils/op_desc_utils.h"

namespace common {
namespace transformer {

enum OpImplType {
  EN_IMPL_CUSTOM_CONSTANT_CCE = 0,    // custom constant op
  EN_IMPL_CUSTOM_TIK,                 // custom tik op
  EN_IMPL_CUSTOM_TBE,                 // custom tbe op
  EN_IMPL_HW_CONSTANT_CCE,            // Huawei built-in constant op
  EN_IMPL_HW_GENERAL_CCE,             // Huawei built-in cce op
  EN_IMPL_HW_TIK,                     // Huawei built-in tik op
  EN_IMPL_HW_TBE,                     // Huawei built-in tbe op
  EN_IMPL_RL,                         // RL op
  EN_IMPL_PLUGIN_TBE,                 // Huawei built-in tbe plugin op
  EN_IMPL_VECTOR_CORE_HW_TBE,         // Huawei built-in tbe op
  EN_IMPL_VECTOR_CORE_CUSTOM_TBE,     // custom tbe op
  EN_IMPL_NON_PERSISTENT_CUSTOM_TBE,  // custom tbe op
  EN_RESERVED                         // reserved value
};

const uint32_t SHAPE_NUMBER_16 = 16;
const uint32_t SHAPE_NUMBER_32 = 32;
const uint32_t SHAPE_DIM_VALUE_C04 = 4;
const uint32_t NI = 16;
const uint32_t MINUS_VALUE_ONE = 1;
const uint32_t MINUS_VALUE_TWO = 2;
const uint32_t SIZE_OF_CN = 2;
const uint32_t MINIMUM_NZ_SHAPE_DIM_NUM = 2;

/* The first parameter is axis value, second is new shape and third is
 * op implementation type. */
using GetNewShapeByAxisValueAndFormat =
    std::function<bool(vector<int64_t>&, const int64_t&, vector<int64_t>&, vector<int64_t>&)>;

using GetNewShapeByAxisValueAndFormatPtr = std::shared_ptr<GetNewShapeByAxisValueAndFormat>;

struct ShapeAndFormatInfo {
  const std::vector<int64_t> &oldShape;
  std::vector<int64_t> &newShape;
  const ge::Format &oldFormat;
  const ge::Format &newFormat;
  const ge::DataType &currentDataType;
  const int64_t &opImplType;
};

using ShapeAndFormat = struct ShapeAndFormatInfo;

class ShapeTransferAccordingToFormat {
 public:
  ShapeTransferAccordingToFormat();

  ~ShapeTransferAccordingToFormat(){};

  ShapeTransferAccordingToFormat(const ShapeTransferAccordingToFormat&) = delete;

  ShapeTransferAccordingToFormat &operator=(const ShapeTransferAccordingToFormat&) = delete;

  bool GetShapeAccordingToFormat(ShapeAndFormat &inputAndOutputInfo, int64_t* c = nullptr);

  /* ----------Below is the function of getting new shape---------------------- */
  static bool GetNDC1HWC0ShapeByAxisValue(vector<int64_t> &new_shape, const int64_t &impl_type,
                                          const std::vector<int64_t> &axis_value, const vector<int64_t> &nd_value);

  static bool GetNCHWShapeByAxisValue(vector<int64_t> &newShape, const int64_t &implType,
                                      const vector<int64_t> &axisValue, const vector<int64_t> &ndValue);

  static bool GetNHWCShapeByAxisValue(vector<int64_t> &newShape, const int64_t &implType,
                                      const vector<int64_t> &axisValue, const vector<int64_t> &ndValue);

  static bool GetNC1HWC0ShapeByAxisValue(vector<int64_t> &newShape, const int64_t &implType,
                                         const vector<int64_t> &axisValue, const vector<int64_t> &ndValue);

  static bool GetFzShapeByAxisValue(vector<int64_t> &newShape, const int64_t &implType,
                                    const vector<int64_t> &axisValue, const vector<int64_t> &ndValue);

  static bool GetHWCNShapeByAxisValue(vector<int64_t> &newShape, const int64_t &implType,
                                      const vector<int64_t> &axisValue, const vector<int64_t> &ndValue);

  static bool GetC1HWNCoC0ShapeByAxisValue(vector<int64_t> &newShape, const int64_t &implType,
                                           const vector<int64_t> &axisValue, const vector<int64_t> &ndValue);

  static bool GetNzShapeByAxisValue(vector<int64_t> &newShape, const int64_t &implType,
                                    const vector<int64_t> &axisValue, const vector<int64_t> &ndValue);

 private:
  /* map of GetAxisValueInfoByFormat, get axis value by different original
   * formats. */
  std::map<ge::Format, GetNewShapeByAxisValueAndFormatPtr> getNewShapeFuncMap;
  std::map<ge::DataType, uint32_t> mapOfDtypeAndC0;
};
} // namespace transformer
} // namespace common

#endif  // COMMON_UTILS_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_
